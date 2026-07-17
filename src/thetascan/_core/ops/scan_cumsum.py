from __future__ import annotations
import torch

CLAMP = 60.0  # exp range guard; with strong decay over long T quad/fla are the honest paths


def linattn(q, k, v, decay=None):
    """O(T) prefix sums over the outer-product state. Memory O(T*Dk*Dv). Scalar decay only."""
    if decay is None:
        S = (k.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(2)   # [B,H,T,Dk,Dv]
        return torch.einsum("bhtk,bhtkv->bhtv", q, S)
    if decay.shape[-1] != 1:
        raise NotImplementedError("cumsum backend: scalar decay only (v1)")
    L = decay.squeeze(-1).cumsum(-1)                        # [B,H,T], <= 0, decreasing
    w = (-L).clamp(max=CLAMP).exp().unsqueeze(-1)           # 1/prod(alpha)
    kv = (k * w).unsqueeze(-1) * v.unsqueeze(-2)
    S = kv.cumsum(2)
    out = torch.einsum("bhtk,bhtkv->bhtv", q, S)
    return out * L.clamp(min=-CLAMP).exp().unsqueeze(-1)
