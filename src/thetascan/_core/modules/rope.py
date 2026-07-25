from __future__ import annotations
from functools import lru_cache

import torch


@lru_cache(maxsize=32)
def _rope_cache_cached(
    T: int,
    rot_dim: int,
    base: float,
    device_type: str,
    device_index: int | None,
    dtype: torch.dtype,
):
    device = (
        torch.device(device_type)
        if device_index is None
        else torch.device(device_type, device_index)
    )
    # Position indices must not be formed in BF16/FP16.  Above their exact
    # integer ranges adjacent positions collapse to the same value before the
    # phase is even evaluated (catastrophic for long-context RoPE).  Match the
    # usual mixed-precision RoPE contract: form phases and trigonometric values
    # in FP32, then cast the cached rotations to the activation dtype.
    phase_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    inv = 1.0 / (
        base
        ** (
            torch.arange(
                0, rot_dim, 2, device=device, dtype=phase_dtype
            )
            / rot_dim
        )
    )
    t = torch.arange(T, device=device, dtype=phase_dtype)
    f = torch.outer(t, inv)                      # [T, rot_dim/2]
    return f.cos().to(dtype=dtype), f.sin().to(dtype=dtype)


def rope_cache(T: int, rot_dim: int, base: float, device, dtype):
    """Return one immutable RoPE table shared by q/k, layers and recomputes."""
    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _rope_cache_cached(
        T,
        rot_dim,
        float(base),
        resolved.type,
        index,
        dtype,
    )


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
