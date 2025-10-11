from __future__ import annotations

import math

import torch


def hmm_predict(q_prev: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    if q_prev.shape != (2,) or P.shape != (2, 2):
        raise ValueError("q_prev must be (2,), P must be (2,2)")
    q_pred = P.T @ q_prev
    q_pred = q_pred.clamp_min(0)
    total = q_pred.sum()
    if float(total) <= 0:
        return torch.full_like(q_prev, 0.5)
    return q_pred / total


def _normal_logpdf(x: torch.Tensor, mean: float, var: float) -> torch.Tensor:
    const = -0.5 * math.log(2.0 * math.pi)
    return const - 0.5 * math.log(var) - 0.5 * ((x - mean) ** 2) / var


def hmm_update(
    q_pred: torch.Tensor,
    x: torch.Tensor,
    mu: float,
    sigma_L: float,
    sigma_H: float,
    dt: float,
) -> torch.Tensor:
    if q_pred.shape != (2,):
        raise ValueError("q_pred must be (2,)")
    var_L = sigma_L ** 2 * dt
    var_H = sigma_H ** 2 * dt
    mean_L = (mu - 0.5 * sigma_L ** 2) * dt
    mean_H = (mu - 0.5 * sigma_H ** 2) * dt
    loglik_L = _normal_logpdf(x, mean_L, var_L)
    loglik_H = _normal_logpdf(x, mean_H, var_H)
    logs = torch.stack([loglik_L, loglik_H]) + torch.log(q_pred.clamp_min(1e-32))
    maxlog = torch.max(logs)
    exps = torch.exp(logs - maxlog)
    q_post = exps / exps.sum()
    q_post = q_post.clamp_min(1e-9)
    return q_post / q_post.sum()


def run_filter(
    xs: torch.Tensor,
    P: torch.Tensor,
    q0: torch.Tensor,
    mu: float,
    sigma_L: float,
    sigma_H: float,
    dt: float,
) -> torch.Tensor:
    beliefs = []
    q = q0
    for x in xs.reshape(-1):
        q = hmm_predict(q, P)
        q = hmm_update(q, x, mu, sigma_L, sigma_H, dt)
        beliefs.append(q.clone())
    if beliefs:
        return torch.stack(beliefs)
    return torch.empty(0, 2, dtype=q0.dtype, device=q0.device)
