from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import torch

from .config import HedgeConfig
from .filtering import hmm_predict, hmm_update
from .options import black_scholes_call, black_scholes_call_delta
from .regime import sample_regimes, stationary_distribution


def _build_price_path(S0: torch.Tensor, log_returns: torch.Tensor) -> torch.Tensor:
    price = torch.empty(log_returns.shape[0] + 1, dtype=log_returns.dtype, device=log_returns.device)
    price[0] = S0
    price[1:] = S0 * torch.exp(torch.cumsum(log_returns, dim=0))
    return price


@dataclass
class Trajectory:
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    hedge_positions: torch.Tensor
    portfolio: torch.Tensor
    prices: torch.Tensor
    option_values: torch.Tensor
    beliefs: torch.Tensor
    losses: torch.Tensor
    total_loss: torch.Tensor  # equals option_value_0 - terminal portfolio value


def generate_market_path(
    cfg: HedgeConfig,
    P: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    steps = cfg.steps
    dtype = cfg.dtype
    device = cfg.device

    regimes = sample_regimes(P, steps, generator=generator).to(device=device)
    S0 = torch.tensor(cfg.S0, dtype=dtype, device=device)

    sigma_low = torch.full((steps,), cfg.sigma_L, dtype=dtype, device=device)
    sigma_high = torch.full((steps,), cfg.sigma_H, dtype=dtype, device=device)
    sigma_path = torch.where((regimes == 0), sigma_low, sigma_high)

    if generator is not None:
        z = torch.randn(steps, generator=generator, dtype=dtype)
        z = z.to(device=device)
    else:
        z = torch.randn(steps, dtype=dtype, device=device)

    drift = (cfg.mu - 0.5 * sigma_path.pow(2)) * cfg.dt
    log_returns = drift + sigma_path * cfg.sqrt_dt * z

    S_path = _build_price_path(S0, log_returns)

    return S_path, log_returns, regimes


def simulate_episode(
    cfg: HedgeConfig,
    P: torch.Tensor,
    policy: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    generator: torch.Generator | None = None,
    deterministic: bool = False,
    path_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> Trajectory:
    device = cfg.device
    dtype = cfg.dtype
    P_tensor = P.to(device=device, dtype=dtype)
    if path_data is None:
        rng = generator if generator is not None and cfg.device.type == "cpu" else generator
        S_path, log_returns, regimes = generate_market_path(cfg, P_tensor, generator=rng)
    else:
        S_path, log_returns, regimes = path_data
    dt = cfg.dt

    q0 = stationary_distribution(P_tensor)
    beliefs = torch.empty(cfg.steps, 2, dtype=dtype, device=device)
    q = q0.clone()
    rho = torch.exp(torch.tensor(cfg.r * dt, dtype=dtype, device=device))

    option_values = torch.empty(cfg.steps + 1, dtype=dtype, device=device)
    option_deltas = torch.empty(cfg.steps, dtype=dtype, device=device)
    tau = torch.linspace(cfg.T, 0.0, steps=cfg.steps + 1, dtype=dtype, device=device)
    sigma_L = torch.tensor(cfg.sigma_L, dtype=dtype, device=device)
    sigma_H = torch.tensor(cfg.sigma_H, dtype=dtype, device=device)

    def mix_price_delta(
        S_val: torch.Tensor,
        tau_val: torch.Tensor,
        belief: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        price_L = black_scholes_call(S_val, cfg.K, cfg.r, tau_val, sigma_L)
        price_H = black_scholes_call(S_val, cfg.K, cfg.r, tau_val, sigma_H)
        delta_L = black_scholes_call_delta(S_val, cfg.K, cfg.r, tau_val, sigma_L)
        delta_H = black_scholes_call_delta(S_val, cfg.K, cfg.r, tau_val, sigma_H)
        price = belief[0] * price_L + belief[1] * price_H
        delta = belief[0] * delta_L + belief[1] * delta_H
        return price, delta

    price0, _ = mix_price_delta(S_path[0], tau[0], q)
    option_values[0] = price0

    states: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    log_probs: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    hedges: List[torch.Tensor] = []
    portfolios: List[torch.Tensor] = []
    losses_per_step: List[torch.Tensor] = []
    beliefs_record: List[torch.Tensor] = []

    Pi = option_values[0].clone()
    hedge_prev = torch.zeros((), dtype=dtype, device=device)

    for t in range(cfg.steps):
        if t > 0:
            q = hmm_predict(q, P_tensor)
            q = hmm_update(q, log_returns[t - 1], cfg.mu, cfg.sigma_L, cfg.sigma_H, dt)
        q = q.detach()
        beliefs[t] = q
        beliefs_record.append(q)

        price_t, delta_t = mix_price_delta(S_path[t], tau[t], q)
        option_values[t] = price_t
        option_deltas[t] = delta_t

        portfolio_feature = torch.clamp(Pi / cfg.S0, -5.0, 5.0)

        state = torch.stack(
            [
                torch.log(S_path[t] / cfg.K),
                torch.tensor((cfg.steps - t) / cfg.steps, dtype=dtype, device=device),
                q[1],
                option_deltas[t],
                option_values[t] / cfg.S0,
                portfolio_feature,
                hedge_prev,
            ]
        ).to(dtype=dtype, device=device)

        act_vec, logp, val = policy(state, deterministic=deterministic)
        act_detached = act_vec.detach()
        act_scalar = act_detached.reshape(-1)[0]
        hedges.append(act_scalar)
        actions.append(act_detached)
        log_probs.append(logp.detach())
        values.append(val.detach())
        states.append(state.detach())

        S_next = S_path[t + 1]
        q_next = q.clone()
        if t < cfg.steps - 1:
            q_next = hmm_predict(q, P_tensor)
            q_next = hmm_update(q_next, log_returns[t], cfg.mu, cfg.sigma_L, cfg.sigma_H, dt)
            q_next = q_next.detach()
        price_next, _ = mix_price_delta(S_next, tau[t + 1], q_next)
        option_values[t + 1] = price_next

        Pi = rho * (Pi - act_scalar * S_path[t]) + act_scalar * S_next - option_values[t + 1]
        Pi = Pi.detach()
        portfolios.append(Pi)
        hedge_prev = act_scalar

    payoff = torch.clamp(S_path[-1] - cfg.K, min=0.0)
    option_values[-1] = payoff
    losses_per_step = torch.zeros(cfg.steps, dtype=dtype, device=device)
    loss_total = -(Pi - option_values[0])

    return Trajectory(
        states=torch.stack(states),
        actions=torch.stack(actions),
        log_probs=torch.stack(log_probs),
        values=torch.stack(values),
        hedge_positions=torch.stack(hedges),
        portfolio=torch.stack(portfolios),
        prices=S_path,
        option_values=option_values,
        beliefs=torch.stack(beliefs_record),
        losses=losses_per_step,
        total_loss=loss_total,
    )
