from __future__ import annotations
import torch

FP8_MAX = 448.0  # float8_e4m3fn


def qcast(w: torch.Tensor, mode: str) -> torch.Tensor:
    """Fake-quantized view of a master-fp32 parameter; gradients flow (STE for fp8)."""
    if mode == "fp32":
        return w
    if mode == "fp16":
        return w.to(torch.float16).to(w.dtype)
    if mode == "bf16":
        return w.to(torch.bfloat16).to(w.dtype)
    if mode == "fp8_e4m3":
        s = w.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / FP8_MAX
        q = (w.detach() / s).to(torch.float8_e4m3fn).to(w.dtype) * s
        return w + (q - w).detach()  # STE
    raise ValueError(mode)
