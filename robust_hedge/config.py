from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch


@dataclass
class HedgeConfig:
    """Central configuration for the robust hedging environment and agent."""

    T: float = 1.0
    steps: int = 252
    mu: float = 0.0
    r: float = 0.0
    sigma_L: float = 0.2
    sigma_H: float = 0.5
    S0: float = 100.0
    K: float = 100.0
    seed: int = 1337
    known_regime: bool = False
    eps: float = 0.02
    alpha: float = 0.95
    P_bar: Optional[torch.Tensor] = None
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
    )
    dtype: torch.dtype = torch.float32

    actor_hidden: int = 128
    critic_hidden: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    actor_lr_final: float = 1e-4
    critic_lr_final: float = 1e-4
    zeta_lr: float = 5e-3
    zeta_lr_min: float = 1e-4
    zeta_lr_decay: float = 0.9
    coverage_target: float = 0.03
    clip_epsilon: float = 0.2
    entropy_coef: float = 1e-3
    entropy_coef_final: float = 3e-4
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float = 0.03
    rollout_episodes: int = 64
    evaluation_episodes: int = 256
    rollout_schedule: Tuple[Tuple[int, int], ...] = ((2, 16), (4, 32))
    evaluation_schedule: Tuple[Tuple[int, int], ...] = ((2, 64), (4, 128))
    minibatch_size: int = 32
    ppo_epochs: int = 2
    log_std_init: float = -0.5
    min_log_std: float = -2.0
    max_log_std: float = 1.0
    state_dim: int = 7
    action_dim: int = 1
    max_hedge: float = 5.0
    save_checkpoints: bool = False
    checkpoint_dir: Optional[Path] = None
    checkpoint_interval: int = 2
    use_antithetic: bool = True

    def __post_init__(self) -> None:
        if self.P_bar is None:
            self.P_bar = torch.tensor([[0.98, 0.02], [0.04, 0.96]], dtype=self.dtype)
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)

    @property
    def dt(self) -> float:
        return self.T / self.steps

    @property
    def sqrt_dt(self) -> float:
        return float(self.dt) ** 0.5

    def rng(self) -> torch.Generator:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        return gen
