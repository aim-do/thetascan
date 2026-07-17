from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HeadLinear(nn.Module):
    """d_model -> [B,H,T,out] projection used by all gates."""
    def __init__(self, d_model: int, n_heads: int, out: int, bias_init: float):
        super().__init__()
        self.h, self.o = n_heads, out
        self.lin = nn.Linear(d_model, n_heads * out, bias=True)
        nn.init.zeros_(self.lin.weight)
        nn.init.constant_(self.lin.bias, bias_init)

    def forward(self, x):                                  # [B,T,D]
        B, T, _ = x.shape
        return self.lin(x).view(B, T, self.h, self.o).transpose(1, 2)  # [B,H,T,out]


class DecayGate(nn.Module):
    """Data-dependent scalar write-side decay: one log(alpha) <= 0 stream per
    head, shared by both LA key kinds."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.g = _HeadLinear(d_model, n_heads, 1, bias_init=-4.0)   # alpha ~ 0.982

    def forward(self, x):
        la = -F.softplus(self.g(x))
        return la, la                       # same scalar stream for both LAs


class OutGate(nn.Module):
    def __init__(self, d_model, n_heads, d):
        super().__init__()
        self.g = _HeadLinear(d_model, n_heads, d, bias_init=1.0)

    def forward(self, x):
        return F.silu(self.g(x))                # [B,H,T,d]
