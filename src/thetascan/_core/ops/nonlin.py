from __future__ import annotations
import torch
import torch.nn.functional as F

# Each entry: act(pre) -> h ; dact(pre, h) -> elementwise sigma'.
# softmax_hidden has a structured (non-diagonal) Jacobian, exposed functionally below.
# swiglu is two-branch and handled directly by the engine.


def _relu2(pre):
    r = F.relu(pre)
    return r * r


def _drelu2(pre, h):
    return 2.0 * F.relu(pre)


def _silu(pre):
    return F.silu(pre)


def _dsilu(pre, h):
    s = torch.sigmoid(pre)
    return s * (1 + pre * (1 - s))


ELEMENTWISE = {
    "relu2": (_relu2, _drelu2),
    "silu": (_silu, _dsilu),
}


def silu_d(pre):
    """silu'(pre) — used by the swiglu gate branch."""
    s = torch.sigmoid(pre)
    return s * (1 + pre * (1 - s))


def softmax_S_apply(h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """(diag(h) - h h^T) u — symmetric, used for both S and S^T."""
    return h * u - h * (h * u).sum(-1, keepdim=True)


def softmax_W2S(W2: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """W2 (diag(h) - h h^T) -> [B,H,T,d,m]. W2 [H,d,m] (H may be 1), h [B,H,T,m]."""
    if W2.shape[0] == 1 and h.shape[1] != 1:
        W2 = W2.expand(h.shape[1], -1, -1)
    W2h = torch.einsum("hdm,bhtm->bhtd", W2, h)
    return W2.unsqueeze(0).unsqueeze(2) * h.unsqueeze(-2) - W2h.unsqueeze(-1) * h.unsqueeze(-2)
