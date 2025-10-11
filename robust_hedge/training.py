from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from .adversary import (
    estimate_cvar,
    estimate_var,
    rectangle_corners,
    select_worst_corner,
    update_zeta,
)
from .agent import ActorCritic
from .config import HedgeConfig
from .evaluation import evaluate_policy_suite
from .simulation import Trajectory, simulate_episode, generate_market_path
from .utils import seed_all


@dataclass
class TrainingStats:
    iteration: int
    zeta: float
    loss_mean: float
    loss_cvar: float
    loss_var: float
    worst_corner_index: int
    actor_loss: float
    critic_loss: float
    approx_kl: float
    coverage_error: float
    tail_probability: float
    env_steps: int
    actor_lr: float
    critic_lr: float
    entropy_coef: float
    zeta_lr: float
    eval_metrics: Dict[str, Dict[str, Dict[str, float]]]


class RobustCVaRPPO:
    def __init__(self, cfg: HedgeConfig) -> None:
        self.cfg = cfg
        seed_all(cfg.seed)
        self.device = cfg.device
        self.policy = ActorCritic(cfg).to(device=cfg.device, dtype=cfg.dtype)
        self.actor_opt = Adam(self.policy.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = Adam(self.policy.critic.parameters(), lr=cfg.critic_lr)
        self.zeta = torch.tensor(0.0, dtype=cfg.dtype, device=cfg.device)
        self.zeta_lr = cfg.zeta_lr
        self.zeta_lr_initial = cfg.zeta_lr
        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.entropy_coef = cfg.entropy_coef
        self.corners = [corner.to(device=cfg.device, dtype=cfg.dtype) for corner in rectangle_corners(cfg.P_bar.to(cfg.device), cfg.eps)]
        self.generator = cfg.rng()
        self.checkpoint_dir = None
        if cfg.save_checkpoints:
            self.checkpoint_dir = cfg.checkpoint_dir or Path("checkpoints")
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _scheduled_value(base: int, schedule: Tuple[Tuple[int, int], ...], iteration: int) -> int:
        for threshold, value in schedule:
            if iteration <= threshold:
                return value
        return base

    def _collect(self, P: Tensor, episodes: int, *, progress: bool = False) -> List[Trajectory]:
        trajectories: List[Trajectory] = []
        with self.policy.evaluation_mode():
            running_mean = 0.0
            idx = 0
            while idx < episodes:
                path = generate_market_path(self.cfg, P)
                traj = simulate_episode(
                    self.cfg,
                    P,
                    self.policy.act,
                    path_data=path,
                )
                trajectories.append(traj)
                if progress:
                    loss_value = float(traj.total_loss.detach().cpu())
                    running_mean += (loss_value - running_mean) / (idx + 1)
                    print(
                        f"\r  Rollout {idx + 1}/{episodes} | mean loss {running_mean:.4f}",
                        end="",
                        flush=True,
                    )
                idx += 1
                if self.cfg.use_antithetic and idx < episodes:
                    S_path, log_returns, regimes = path
                    log_returns_neg = -log_returns
                    S_ant = torch.empty_like(S_path)
                    S_ant[0] = S_path[0]
                    S_ant[1:] = S_path[0] * torch.exp(torch.cumsum(log_returns_neg, dim=0))
                    path_ant = (S_ant, log_returns_neg, regimes)
                    traj_ant = simulate_episode(
                        self.cfg,
                        P,
                        self.policy.act,
                        path_data=path_ant,
                    )
                    trajectories.append(traj_ant)
                    if progress:
                        loss_value = float(traj_ant.total_loss.detach().cpu())
                        running_mean += (loss_value - running_mean) / (idx + 1)
                        print(
                            f"\r  Rollout {idx + 1}/{episodes} | mean loss {running_mean:.4f}",
                            end="",
                            flush=True,
                        )
                    idx += 1
            if progress:
                print()
        return trajectories

    def _prepare_batch(
        self,
        trajectories: Sequence[Trajectory],
        zeta: Tensor,
    ) -> Dict[str, Tensor]:
        states = []
        actions = []
        log_probs = []
        targets = []
        losses = []
        for traj in trajectories:
            steps = traj.states.shape[0]
            tail_target = torch.clamp(traj.total_loss - zeta.detach(), min=0.0).detach()
            states.append(traj.states)
            actions.append(traj.actions)
            log_probs.append(traj.log_probs)
            targets.append(tail_target.repeat(steps))
            losses.append(traj.total_loss)
        batch = {
            "states": torch.cat(states),
            "actions": torch.cat(actions),
            "old_log_probs": torch.cat(log_probs),
            "targets": torch.cat(targets),
            "losses": torch.stack(losses),
        }
        return batch

    def _update_policy(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        states = batch["states"].to(device=self.device, dtype=self.cfg.dtype)
        actions = batch["actions"].to(device=self.device, dtype=self.cfg.dtype)
        old_log_probs = batch["old_log_probs"].to(device=self.device, dtype=self.cfg.dtype)
        targets = batch["targets"].to(device=self.device, dtype=self.cfg.dtype)

        with torch.no_grad():
            values = self.policy.value(states).squeeze(-1)
        advantages = targets - values
        advantages = advantages - advantages.mean()
        adv_std = advantages.std(unbiased=False)
        if torch.isfinite(adv_std) and adv_std.item() > 1e-8:
            advantages = advantages / (adv_std + 1e-8)

        num_samples = states.shape[0]
        actor_losses = []
        critic_losses = []
        kl_values = []

        early_stop = False
        for _ in range(self.cfg.ppo_epochs):
            indices = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                idx = indices[start:end]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_targets = targets[idx]

                new_log_probs, entropy, values = self.policy.evaluate_actions(batch_states, batch_actions)
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                scaled_targets = batch_targets / self.cfg.S0
                scaled_values = values / self.cfg.S0
                value_loss = F.smooth_l1_loss(scaled_values, scaled_targets)
                loss = actor_loss + self.cfg.value_coef * value_loss

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                actor_losses.append(actor_loss.detach())
                critic_losses.append(value_loss.detach())
                approx_kl = (batch_old_log_probs - new_log_probs).mean().detach()
                kl_values.append(approx_kl)
                if approx_kl > self.cfg.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break
        if early_stop:
            pass

        metrics = {
            "actor_loss": torch.stack(actor_losses).mean().item() if actor_losses else 0.0,
            "critic_loss": torch.stack(critic_losses).mean().item() if critic_losses else 0.0,
            "approx_kl": torch.stack(kl_values).mean().item() if kl_values else 0.0,
        }
        return metrics

    def train(self, iterations: int) -> List[TrainingStats]:
        stats: List[TrainingStats] = []
        print(
            f"Starting training for {iterations} iterations "
            f"(rollout base={self.cfg.rollout_episodes}, eval base={self.cfg.evaluation_episodes})"
        )
        for iteration in range(1, iterations + 1):
            self._apply_schedules(iteration, iterations)
            eval_eps = self._scheduled_value(
                self.cfg.evaluation_episodes, self.cfg.evaluation_schedule, iteration
            )
            rollout_eps = self._scheduled_value(
                self.cfg.rollout_episodes, self.cfg.rollout_schedule, iteration
            )
            print(
                f"Iteration {iteration}/{iterations}: evaluating rectangle corners "
                f"({eval_eps} episodes per corner)..."
            )
            with self.policy.evaluation_mode():
                base_seed = self.cfg.seed + iteration * 100_000
                worst_corner, eval_losses, worst_idx = select_worst_corner(
                    self.cfg,
                    self.policy.act,
                    self.corners,
                    episodes=eval_eps,
                    progress=True,
                    base_seed=base_seed,
                    use_antithetic=self.cfg.use_antithetic,
                )
                loss_mean = eval_losses.mean().item()
                loss_cvar = estimate_cvar(eval_losses, self.cfg.alpha).item()
                loss_var = estimate_var(eval_losses, self.cfg.alpha).item()

            print(
                f"Iteration {iteration}/{iterations}: collecting {rollout_eps} rollouts..."
            )
            trajectories = self._collect(
                worst_corner, rollout_eps, progress=True
            )
            batch_losses = torch.stack([traj.total_loss for traj in trajectories])
            self.zeta = update_zeta(self.zeta, batch_losses, alpha=self.cfg.alpha, lr=self.zeta_lr)
            coverage = (batch_losses > self.zeta).float().mean().item()
            coverage_error = coverage - (1.0 - self.cfg.alpha)
            abs_coverage_error = abs(coverage_error)
            if abs_coverage_error > self.cfg.coverage_target:
                self.zeta_lr = max(self.cfg.zeta_lr_min, self.zeta_lr * self.cfg.zeta_lr_decay)
            batch = self._prepare_batch(trajectories, self.zeta)
            metrics = self._update_policy(batch)

            with self.policy.evaluation_mode():
                print(
                    f"Iteration {iteration}/{iterations}: full evaluation suite "
                    f"({eval_eps} episodes per scenario)..."
                )
                eval_metrics = evaluate_policy_suite(
                    self.cfg,
                    self.policy.act,
                    episodes=eval_eps,
                    alpha=self.cfg.alpha,
                    progress=True,
                )

            print(
                f"Iteration {iteration}/{iterations}: "
                f"zeta={self.zeta.item():.4f}, "
                f"mean={loss_mean:.4f}, cvar={loss_cvar:.4f}, "
                f"actor_loss={metrics['actor_loss']:.4f}, "
                f"critic_loss={metrics['critic_loss']:.4f}, "
                f"KL={metrics['approx_kl']:.4f}, "
                f"tail_p={coverage:.4f}, error={coverage_error:.4f}, worst_corner={worst_idx}",
            )

            stats.append(
                TrainingStats(
                    iteration=iteration,
                    zeta=self.zeta.item(),
                    loss_mean=loss_mean,
                    loss_cvar=loss_cvar,
                    loss_var=loss_var,
                    worst_corner_index=worst_idx,
                    actor_loss=metrics["actor_loss"],
                    critic_loss=metrics["critic_loss"],
                    approx_kl=metrics["approx_kl"],
                    coverage_error=abs_coverage_error,
                    tail_probability=coverage,
                    env_steps=rollout_eps * self.cfg.steps * (2 if self.cfg.use_antithetic else 1),
                    actor_lr=self.actor_lr,
                    critic_lr=self.critic_lr,
                    entropy_coef=self.entropy_coef,
                    zeta_lr=self.zeta_lr,
                    eval_metrics=eval_metrics,
                )
            )

            if self.checkpoint_dir is not None and iteration % self.cfg.checkpoint_interval == 0:
                self._save_checkpoint(iteration)
        return stats

    def _save_checkpoint(self, iteration: int) -> None:
        assert self.checkpoint_dir is not None
        stem = f"iter_{iteration:03d}"
        actor_path = self.checkpoint_dir / f"policy_{stem}.pt"
        critic_path = self.checkpoint_dir / f"value_{stem}.pt"
        actor_opt_path = self.checkpoint_dir / f"actor_opt_{stem}.pt"
        critic_opt_path = self.checkpoint_dir / f"critic_opt_{stem}.pt"
        torch.save(self.policy.actor.state_dict(), actor_path)
        torch.save(self.policy.critic.state_dict(), critic_path)
        torch.save(self.actor_opt.state_dict(), actor_opt_path)
        torch.save(self.critic_opt.state_dict(), critic_opt_path)
        print(
            f"  Saved checkpoint to {self.checkpoint_dir} (iteration {iteration})"
        )

    def _apply_schedules(self, iteration: int, iterations: int) -> None:
        denom = max(iterations - 1, 1)
        frac = (iteration - 1) / denom
        self.actor_lr = self.cfg.actor_lr + frac * (self.cfg.actor_lr_final - self.cfg.actor_lr)
        self.critic_lr = self.cfg.critic_lr + frac * (self.cfg.critic_lr_final - self.cfg.critic_lr)
        self.entropy_coef = self.cfg.entropy_coef + frac * (self.cfg.entropy_coef_final - self.cfg.entropy_coef)
        for group in self.actor_opt.param_groups:
            group["lr"] = self.actor_lr
        for group in self.critic_opt.param_groups:
            group["lr"] = self.critic_lr
