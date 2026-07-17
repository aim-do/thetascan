from __future__ import annotations
import torch


def rope_cache(T: int, rot_dim: int, base: float, device, dtype):
    inv = 1.0 / (base ** (torch.arange(0, rot_dim, 2, device=device, dtype=dtype) / rot_dim))
    t = torch.arange(T, device=device, dtype=dtype)
    f = torch.outer(t, inv)                      # [T, rot_dim/2]
    return f.cos(), f.sin()


def apply_rope(x: torch.Tensor, mode: str, frac: float, base: float) -> torch.Tensor:
    """x [B,H,T,d]. Rotates the first rot_dim channels pairwise; rest untouched."""
    if mode == "none":
        return x
    d = x.shape[-1]
    max_rot = d // 2 * 2
    if max_rot == 0:
        return x
    rot = max_rot if mode == "full" else min(max_rot, max(2, int(d * frac) // 2 * 2))
    cos, sin = rope_cache(x.shape[2], rot, base, x.device, x.dtype)   # [T, rot/2]
    xr, xp = x[..., :rot], x[..., rot:]
    x1, x2 = xr[..., 0::2], xr[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    out = torch.stack((o1, o2), dim=-1).flatten(-2)
    return torch.cat((out, xp), dim=-1)
