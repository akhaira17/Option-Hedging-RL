from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from .config import HedgeConfig
from .simulation import generate_market_path, simulate_episode


def rectangle_corners(P_bar: torch.Tensor, eps: float) -> List[torch.Tensor]:
    corners: List[torch.Tensor] = []
    device = P_bar.device
    dtype = P_bar.dtype
    base_ll = float(P_bar[0, 0])
    base_hl = float(P_bar[1, 0])
    for d_ll in (-eps, eps):
        p_ll = min(max(base_ll + d_ll, 1e-6), 1 - 1e-6)
        row0 = torch.tensor([p_ll, 1.0 - p_ll], dtype=dtype, device=device)
        for d_hl in (-eps, eps):
            p_hl = min(max(base_hl + d_hl, 1e-6), 1 - 1e-6)
            row1 = torch.tensor([p_hl, 1.0 - p_hl], dtype=dtype, device=device)
            corners.append(torch.stack([row0, row1]))
    return corners


def estimate_var(losses: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.quantile(losses, alpha)


def estimate_cvar(losses: torch.Tensor, alpha: float) -> torch.Tensor:
    var = estimate_var(losses, alpha)
    mask = losses >= var
    if mask.sum() == 0:
        return var
    return losses[mask].mean()


def select_worst_corner(
    cfg: HedgeConfig,
    policy,
    corners: Sequence[torch.Tensor],
    *,
    episodes: int,
    progress: bool = False,
    base_seed: int | None = None,
    use_antithetic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    worst_cvar = None
    worst_corner = None
    losses = None
    worst_idx = -1
    num_corners = len(corners)
    for corner_idx, corner in enumerate(corners):
        batch_losses = []
        running_mean = 0.0
        for episode_idx in range(episodes):
            seed = (base_seed or cfg.seed) + episode_idx
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            path = generate_market_path(cfg, corner, generator=generator)
            traj = simulate_episode(cfg, corner, policy, path_data=path)
            loss_tensor = traj.total_loss.detach()
            if use_antithetic:
                S_path, log_returns, regimes = path
                log_neg = -log_returns
                S_ant = torch.empty_like(S_path)
                S_ant[0] = S_path[0]
                S_ant[1:] = S_path[0] * torch.exp(torch.cumsum(log_neg, dim=0))
                path_ant = (S_ant, log_neg, regimes)
                traj_ant = simulate_episode(cfg, corner, policy, path_data=path_ant)
                loss_tensor = 0.5 * (loss_tensor + traj_ant.total_loss.detach())
            batch_losses.append(loss_tensor)
            if progress:
                loss_value = float(loss_tensor.cpu())
                running_mean += (loss_value - running_mean) / (episode_idx + 1)
                print(
                    f"\r    Corner {corner_idx + 1}/{num_corners}"
                    f" | episode {episode_idx + 1}/{episodes}"
                    f" | mean loss {running_mean:.4f}",
                    end="",
                    flush=True,
                )
        if progress:
            print()
        losses_tensor = torch.stack(batch_losses)
        cvar = estimate_cvar(losses_tensor, cfg.alpha)
        if worst_cvar is None or cvar > worst_cvar:
            worst_cvar = cvar
            worst_corner = corner
            losses = losses_tensor
            worst_idx = corner_idx
    if worst_corner is None or worst_cvar is None or losses is None:
        raise RuntimeError("Failed to evaluate adversary corners")
    return worst_corner, losses, worst_idx


def update_zeta(
    zeta: torch.Tensor,
    losses: torch.Tensor,
    *,
    alpha: float,
    lr: float,
) -> torch.Tensor:
    indicator = (losses > zeta).float()
    prob = indicator.mean()
    grad = 1.0 - (1.0 / (1.0 - alpha)) * prob
    zeta_next = zeta - lr * grad
    margin = torch.tensor(5.0, dtype=losses.dtype, device=losses.device)
    lower = losses.min() - margin
    upper = losses.max() + margin
    return torch.clamp(zeta_next, lower, upper)


def tail_losses(losses: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
    return torch.clamp(losses - zeta, min=0.0)
