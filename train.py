"""Entry point for training the robust CVaR PPO hedging agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from robust_hedge.config import HedgeConfig
from robust_hedge.training import RobustCVaRPPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the robust CVaR PPO hedging agent")
    parser.add_argument("--iterations", type=int, default=10, help="Number of PPO outer iterations")
    parser.add_argument("--rollout-episodes", type=int, default=64, help="Episodes per update")
    parser.add_argument("--eval-episodes", type=int, default=64, help="Episodes per adversary evaluation")
    parser.add_argument("--alpha", type=float, default=0.95, help="CVaR confidence level")
    parser.add_argument("--eps", type=float, default=0.02, help="Transition matrix rectangle half-width")
    parser.add_argument("--output", type=Path, default=Path("training_stats.json"), help="Where to dump training statistics")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, cuda, mps)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(args.device)
    else:
        device = HedgeConfig().device

    cfg = HedgeConfig(
        alpha=args.alpha,
        eps=args.eps,
        rollout_episodes=args.rollout_episodes,
        evaluation_episodes=args.eval_episodes,
        device=device,
    )

    trainer = RobustCVaRPPO(cfg)
    stats = trainer.train(args.iterations)

    payload = [
        {
            "iteration": s.iteration,
            "zeta": s.zeta,
            "loss_mean": s.loss_mean,
            "loss_cvar": s.loss_cvar,
            "loss_var": s.loss_var,
            "worst_corner_index": s.worst_corner_index,
            "actor_loss": s.actor_loss,
            "critic_loss": s.critic_loss,
            "approx_kl": s.approx_kl,
            "coverage_error": s.coverage_error,
            "env_steps": s.env_steps,
            "eval_metrics": s.eval_metrics,
        }
        for s in stats
    ]
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Training complete. Stats saved to {args.output}")


if __name__ == "__main__":
    main()
