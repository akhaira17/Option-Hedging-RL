"""Minimal smoke test for the robust hedging stack."""

import torch

from robust_hedge.config import HedgeConfig
from robust_hedge.simulation import simulate_episode
from robust_hedge.agent import ActorCritic


def run_episode() -> None:
    cfg = HedgeConfig(steps=16)
    policy = ActorCritic(cfg).to(device=cfg.device, dtype=cfg.dtype)

    def policy_fn(state, deterministic=False):
        return policy.act(state, deterministic=deterministic)

    traj = simulate_episode(cfg, cfg.P_bar, policy_fn)
    assert traj.states.shape[0] == cfg.steps
    assert traj.actions.shape[0] == cfg.steps
    print("Episode loss:", float(traj.total_loss))


if __name__ == "__main__":
    torch.manual_seed(0)
    run_episode()
