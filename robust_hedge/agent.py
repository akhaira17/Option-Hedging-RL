from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal

from .config import HedgeConfig


def _atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class TanhNormal:
    def __init__(self, mu: torch.Tensor, log_std: torch.Tensor, bound: float) -> None:
        self.mu = mu
        self.log_std = log_std
        self.std = torch.exp(log_std)
        self.base = Normal(mu, self.std)
        self.bound = bound

    def rsample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.base.rsample()
        a_unit = torch.tanh(u)
        a = self.bound * a_unit
        return a, u, a_unit

    def log_prob(
        self,
        a_scaled: torch.Tensor,
        *,
        u: torch.Tensor | None = None,
        a_unit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if a_unit is None:
            a_unit = a_scaled / self.bound
        a_unit = a_unit.clamp(-0.999999, 0.999999)
        if u is None:
            u = _atanh(a_unit)
        log_prob_u = self.base.log_prob(u)
        log_det = math.log(self.bound) + torch.log1p(-a_unit.pow(2) + 1e-6)
        return (log_prob_u - log_det).sum(dim=-1)

    def entropy(self, *, a_unit: torch.Tensor) -> torch.Tensor:
        base_entropy = torch.sum(
            self.log_std + 0.5 * (1.0 + math.log(2.0 * math.pi)), dim=-1
        )
        log_det = math.log(self.bound) + torch.log1p(-a_unit.pow(2) + 1e-6)
        return base_entropy + log_det.sum(dim=-1)


class GaussianActor(nn.Module):
    def __init__(self, cfg: HedgeConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.actor_hidden),
            nn.Tanh(),
            nn.Linear(cfg.actor_hidden, cfg.actor_hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(cfg.actor_hidden, cfg.action_dim)
        self.log_std_head = nn.Linear(cfg.actor_hidden, cfg.action_dim)
        self.min_log_std = cfg.min_log_std
        self.max_log_std = cfg.max_log_std
        with torch.no_grad():
            self.log_std_head.weight.zero_()
            self.log_std_head.bias.fill_(cfg.log_std_init)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(self.min_log_std, self.max_log_std)
        return mu, log_std


class ValueNet(nn.Module):
    def __init__(self, cfg: HedgeConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.critic_hidden),
            nn.Tanh(),
            nn.Linear(cfg.critic_hidden, cfg.critic_hidden),
            nn.Tanh(),
            nn.Linear(cfg.critic_hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ActorCritic(nn.Module):
    def __init__(self, cfg: HedgeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.actor = GaussianActor(cfg)
        self.critic = ValueNet(cfg)

    def distribution(self, state: torch.Tensor) -> TanhNormal:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        mu, log_std = self.actor(state)
        return TanhNormal(mu, log_std, self.cfg.max_hedge)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.critic(state)

    def act(
        self,
        state: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(state)
        if deterministic:
            mu = dist.mu
            action_unit = torch.tanh(mu)
            action = self.cfg.max_hedge * action_unit
            log_prob = torch.zeros(action.shape[:-1], device=action.device, dtype=action.dtype)
        else:
            action, u, a_unit = dist.rsample()
            log_prob = dist.log_prob(action, u=u, a_unit=a_unit)
        value = self.value(state).squeeze(-1)
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(states)
        a_unit = (actions / self.cfg.max_hedge).clamp(-0.999999, 0.999999)
        u = _atanh(a_unit)
        log_probs = dist.log_prob(actions, u=u, a_unit=a_unit)
        entropy = dist.entropy(a_unit=a_unit)
        values = self.value(states).squeeze(-1)
        return log_probs, entropy, values

    @contextmanager
    def evaluation_mode(self):
        training = self.training
        try:
            self.eval()
            with torch.no_grad():
                yield
        finally:
            if training:
                self.train()
            else:
                self.eval()
