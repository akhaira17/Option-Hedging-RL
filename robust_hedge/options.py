from __future__ import annotations

import math

import torch


def _std_norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _std_norm_pdf(x: torch.Tensor) -> torch.Tensor:
    return (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * x**2)


def black_scholes_call(
    S: torch.Tensor,
    K: float,
    r: float,
    tau: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    tiny = 1e-8
    tau = torch.clamp(tau, min=0.0)
    sigma = torch.clamp(sigma, min=1e-8)
    sqrt_tau = torch.sqrt(torch.clamp(tau, min=tiny))
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau + tiny)
    d2 = d1 - sigma * sqrt_tau
    price = S * _std_norm_cdf(d1) - K * torch.exp(-r * tau) * _std_norm_cdf(d2)
    intrinsic = torch.clamp(S - K, min=0.0)
    return torch.where(tau <= tiny, intrinsic, price)


def black_scholes_call_delta(
    S: torch.Tensor,
    K: float,
    r: float,
    tau: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    tiny = 1e-8
    tau = torch.clamp(tau, min=0.0)
    sigma = torch.clamp(sigma, min=1e-8)
    sqrt_tau = torch.sqrt(torch.clamp(tau, min=tiny))
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau + tiny)
    delta = _std_norm_cdf(d1)
    intrinsic_delta = (S > K).to(S.dtype)
    return torch.where(tau <= tiny, intrinsic_delta, delta)
