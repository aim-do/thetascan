from __future__ import annotations
import torch


def rmsnorm(a: torch.Tensor, eps: float = 1e-6):
    """Returns (normed, inv_rms). inv_rms [..,1] is the scalar derivative factor."""
    inv = torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + eps)
    return a * inv, inv


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
