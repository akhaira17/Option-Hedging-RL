from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from .adversary import estimate_cvar, estimate_var, rectangle_corners
from .config import HedgeConfig
from .simulation import generate_market_path, simulate_episode


@dataclass
class LossSummary:
    mean: float
    var: float
    cvar: float

    def to_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "var": self.var, "cvar": self.cvar}


class _DeterministicPolicy:
    def __init__(self, action_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self._action_fn = action_fn

    def __call__(self, state: torch.Tensor, *, deterministic: bool = False):
        action = self._action_fn(state)
        zeros = torch.zeros_like(action)
        return action, zeros.squeeze(), zeros.squeeze()


def bs_delta_policy(cfg: HedgeConfig) -> _DeterministicPolicy:
    bound = cfg.max_hedge

    def _fn(state: torch.Tensor) -> torch.Tensor:
        delta = state[3].clamp(-bound, bound)
        return delta.reshape(1)

    return _DeterministicPolicy(_fn)


def no_hedge_policy(cfg: HedgeConfig) -> _DeterministicPolicy:
    def _fn(state: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, dtype=state.dtype, device=state.device)

    return _DeterministicPolicy(_fn)


def _collect_losses(
    cfg: HedgeConfig,
    P: torch.Tensor,
    policy_fn,
    episodes: int,
    *,
    deterministic: bool,
    seed_offset: int = 0,
    progress_prefix: Optional[str] = None,
    use_antithetic: bool = False,
) -> torch.Tensor:
    losses = []
    running_mean = 0.0
    idx = 0
    while idx < episodes:
        seed = cfg.seed + seed_offset + idx
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        path = generate_market_path(cfg, P, generator=generator)
        traj = simulate_episode(
            cfg,
            P,
            policy_fn,
            path_data=path,
            deterministic=deterministic,
        )
        if use_antithetic:
            S_path, log_returns, regimes = path
            log_returns_neg = -log_returns
            S_ant = torch.empty_like(S_path)
            S_ant[0] = S_path[0]
            S_ant[1:] = S_path[0] * torch.exp(torch.cumsum(log_returns_neg, dim=0))
            path_ant = (S_ant, log_returns_neg, regimes)
            traj_ant = simulate_episode(
                cfg,
                P,
                policy_fn,
                path_data=path_ant,
                deterministic=deterministic,
            )
            loss_tensor = 0.5 * (traj.total_loss.detach() + traj_ant.total_loss.detach())
        else:
            loss_tensor = traj.total_loss.detach()
        losses.append(loss_tensor)
        if progress_prefix is not None:
            loss_value = float(loss_tensor.cpu())
            running_mean += (loss_value - running_mean) / (idx + 1)
            print(
                f"\r{progress_prefix} | episode {idx + 1}/{episodes} | mean loss {running_mean:.4f}",
                end="",
                flush=True,
            )
        idx += 1
    if progress_prefix is not None:
        print()
    return torch.stack(losses)


def _summarise(losses: torch.Tensor, alpha: float) -> LossSummary:
    mean = losses.mean().item()
    var = estimate_var(losses, alpha).item()
    cvar = estimate_cvar(losses, alpha).item()
    return LossSummary(mean=mean, var=var, cvar=cvar)


def _worst_corner(
    cfg: HedgeConfig,
    policy_fn,
    corners,
    episodes: int,
    *,
    deterministic: bool,
    seed_offset: int,
    progress: bool = False,
    use_antithetic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    worst_corner = None
    worst_losses = None
    worst_cvar = None
    for idx, corner in enumerate(corners):
        prefix = None
        if progress:
            prefix = f"    Corner {idx + 1}/{len(corners)}"
        losses = _collect_losses(
            cfg,
            corner,
            policy_fn,
            episodes,
            deterministic=deterministic,
            seed_offset=seed_offset + idx * episodes,
            progress_prefix=prefix,
            use_antithetic=use_antithetic,
        )
        cvar = estimate_cvar(losses, cfg.alpha)
        if worst_cvar is None or cvar > worst_cvar:
            worst_cvar = cvar
            worst_corner = corner
            worst_losses = losses
    if worst_corner is None or worst_losses is None:
        raise RuntimeError("No corner evaluated during worst-corner search")
    return worst_corner, worst_losses


def evaluate_policy_suite(
    cfg: HedgeConfig,
    policy_fn,
    *,
    episodes: int,
    alpha: float,
    eps_multiplier: float = 1.5,
    include_baselines: bool = True,
    deterministic: bool = True,
    progress: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    baseline_policies = {}
    if include_baselines:
        baseline_policies["bs_delta"] = bs_delta_policy(cfg)
        baseline_policies["no_hedge"] = no_hedge_policy(cfg)

    def add_metrics(tag: str, P: torch.Tensor, losses_agent: torch.Tensor, scenario_seed: int) -> None:
        scenario = {"agent": _summarise(losses_agent, alpha).to_dict()}
        if include_baselines:
            for idx, (name, baseline) in enumerate(baseline_policies.items()):
                baseline_losses = _collect_losses(
                    cfg,
                    P,
                    baseline,
                    episodes,
                    deterministic=True,
                    seed_offset=scenario_seed + (idx + 1) * 10_000,
                )
                scenario[name] = _summarise(baseline_losses, alpha).to_dict()
        results[tag] = scenario

    P_bar = cfg.P_bar.to(cfg.device, dtype=cfg.dtype)
    prefix = "  P_bar" if progress else None
    losses_bar = _collect_losses(
        cfg,
        P_bar,
        policy_fn,
        episodes,
        deterministic=deterministic,
        seed_offset=0,
        progress_prefix=prefix,
        use_antithetic=cfg.use_antithetic,
    )
    add_metrics("P_bar", P_bar, losses_bar, scenario_seed=0)

    corners = rectangle_corners(P_bar, cfg.eps)
    worst_corner, losses_worst = _worst_corner(
        cfg,
        policy_fn,
        corners,
        episodes,
        deterministic=deterministic,
        seed_offset=1_000,
        progress=progress,
        use_antithetic=cfg.use_antithetic,
    )
    if progress:
        print("  Worst corner evaluation complete")
    add_metrics("worst_corner", worst_corner, losses_worst, scenario_seed=1_000)

    stress_corners = rectangle_corners(P_bar, cfg.eps * eps_multiplier)
    stress_corner, losses_stress = _worst_corner(
        cfg,
        policy_fn,
        stress_corners,
        episodes,
        deterministic=deterministic,
        seed_offset=2_000,
        progress=progress,
        use_antithetic=cfg.use_antithetic,
    )
    if progress:
        print("  Stress corner evaluation complete")
    add_metrics("stress_corner", stress_corner, losses_stress, scenario_seed=2_000)

    return results
