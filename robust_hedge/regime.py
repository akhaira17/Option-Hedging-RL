from __future__ import annotations

import torch
from torch.distributions import Categorical


def stationary_distribution(P: torch.Tensor) -> torch.Tensor:
    """Return the stationary distribution of a 2x2 transition matrix."""
    if P.shape != (2, 2):
        raise ValueError("Only 2x2 matrices supported")
    p_lh = P[0, 1]
    p_hl = P[1, 0]
    denom = p_lh + p_hl
    if float(denom) <= 0:
        return torch.tensor([0.5, 0.5], dtype=P.dtype, device=P.device)
    weights = torch.stack([p_hl, p_lh]).clamp_min(0)
    return weights / weights.sum()


def sample_regimes(
    P: torch.Tensor,
    steps: int,
    *,
    start_state: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a latent regime path of length ``steps`` from transition matrix ``P``."""
    if P.shape != (2, 2):
        raise ValueError("P must be 2x2")
    if start_state is None:
        pi = stationary_distribution(P)
        if generator is None:
            start_state = int(torch.multinomial(pi, 1).item())
        else:
            start_state = int(torch.multinomial(pi.cpu(), 1, generator=generator).item())
    state = start_state
    path = []
    dist = Categorical
    for _ in range(steps):
        path.append(state)
        probs = P[state]
        if generator is None:
            state = int(dist(probs).sample().item())
        else:
            state = int(torch.multinomial(probs.cpu(), 1, generator=generator).item())
    return torch.tensor(path, dtype=torch.long, device=P.device)
