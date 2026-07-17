from __future__ import annotations
import torch


def linattn(q, k, v, decay=None):
    """Masked-matmul dual form, O(T^2) but matmul-shaped. Scalar decay only."""
    T = q.shape[2]
    att = q @ k.transpose(-1, -2)                          # [B,H,T,T]
    mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    if decay is not None:
        if decay.shape[-1] != 1:
            raise NotImplementedError("quad backend: scalar decay only (v1)")
        L = decay.squeeze(-1).cumsum(-1)                   # [B,H,T]
        att = att * (L.unsqueeze(-1) - L.unsqueeze(-2)).clamp(max=0.0).exp()
    att = att.masked_fill(~mask, 0.0)
    return att @ v
