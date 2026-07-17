from __future__ import annotations
import torch


def linattn(q, k, v, decay=None):
    """Loop oracle. q,k [B,H,T,Dk], v [B,H,T,Dv], decay None | log-alpha
    [B,H,T,1] (scalar) | [B,H,T,Dk] (per key channel)."""
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    S = q.new_zeros(B, H, Dk, Dv)
    out = q.new_zeros(B, H, T, Dv)
    for t in range(T):
        if decay is not None:
            a = decay[:, :, t].exp()                       # [B,H,1] or [B,H,Dk]
            S = S * a.unsqueeze(-1)
        S = S + k[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
        out[:, :, t] = (q[:, :, t].unsqueeze(-1) * S).sum(-2)
    return out
