from __future__ import annotations

import torch

from . import scan_naive, scan_quad, scan_cumsum, scan_fla

_REGISTRY = {"naive": scan_naive.linattn, "quad": scan_quad.linattn, "cumsum": scan_cumsum.linattn}


def resolve_backend(
    name: str,
    decay_gate: str = "off",
    accumulation: str = "sum",
    device: torch.device | str | None = None,
    require_scalar_decay: bool = False,
) -> str:
    if name != "auto":
        return name
    target = None if device is None else torch.device(device)
    if (target is None or target.type == "cuda") and torch.cuda.is_available():
        try:
            if scan_fla.supports(
                decay_gate,
                require_scalar_decay=require_scalar_decay,
            ):
                return "fla"
        except Exception:
            pass
    # Non-CUDA input (or no FLA): quad is the default matmul form.
    return "quad"


def _fn(backend: str):
    if backend == "fla":
        return scan_fla.linattn
    return _REGISTRY[backend]


def _roll(x: torch.Tensor, n: int) -> torch.Tensor:
    """Shift right along T by n, zero-fill the front."""
    if n == 0:
        return x
    pad = x.new_zeros(*x.shape[:2], n, *x.shape[3:])
    return torch.cat([pad, x[:, :, :-n]], dim=2)


def ema_cumsum(x: torch.Tensor, log_alpha: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Discounted causal cumsum along T: z_t = sum_{i<=t} alpha^(t-i) x_i, with a
    per-head STATIC alpha = exp(log_alpha) [H]. Exact chunked form: inside a chunk
    a lower-triangular Toeplitz matmul (exponents clamped >= 0 BEFORE exp — no
    1/alpha^i rescaling, so small alphas cannot overflow, unlike scan_cumsum's
    CLAMP trick); the carry re-enters each chunk through alpha^(offset).
    Differentiable in log_alpha. x [B,H,T,...]."""
    B, H, T = x.shape[:3]
    la = log_alpha.to(dtype=x.dtype, device=x.device).view(H)
    # No indexed writes into a shared output: the carry is a VIEW of the chunk it
    # came from, and any later in-place write into a shared z would version-bump
    # the storage autograd saved for the carry's backward when T spans chunks.
    chunks = []
    carry = None
    for t0 in range(0, T, chunk):
        C = min(T, t0 + chunk) - t0
        i = torch.arange(C, device=x.device, dtype=x.dtype)
        expo = (i.view(C, 1) - i.view(1, C)).clamp(min=0.0)
        A = torch.exp(la.view(H, 1, 1) * expo).tril()          # [H,C,C]: alpha^(t-i)
        zc = torch.einsum("hij,bhj...->bhi...", A, x[:, :, t0:t0 + C])
        if carry is not None:
            dec = torch.exp(la.view(1, H, 1) * (i + 1.0))      # [1,H,C]
            zc = zc + carry.unsqueeze(2) * dec.view(1, H, C, *([1] * (x.dim() - 3)))
        chunks.append(zc)
        carry = zc[:, :, -1]
    return torch.cat(chunks, dim=2)


class Accumulator:
    """Per-mixer-pass accumulation context shared by every LA call."""

    def __init__(self, backend: str, decay_d=None, decay_m=None,
                 eps: float = 1e-6):
        self.backend = backend
        self.decay = {"d": decay_d, "m": decay_m}   # log-alpha streams keyed by key-dim kind
        self.eps = eps

    def mass_cum(self, keys: torch.Tensor) -> torch.Tensor:
        """Cumulative key mass for the read_norm denominator. Plain prefix sum —
        Sum uses a plain prefix. EMA applies its decay to this mass too; fade
        overrides this method with its own fast/stale masses."""
        decay = self.decay["m"]
        if decay is None:
            return keys.cumsum(dim=2)
        # A unit query/key scan returns the decayed prefix of each key
        # coordinate, exactly matching the filter on the memory numerator.
        ones = keys.new_ones(*keys.shape[:3], 1)
        return _fn(self.backend)(ones, ones, keys, decay)

    def __call__(self, q, k, v, key_kind: str):
        return _fn(self.backend)(q, k, v, self.decay[key_kind])


class FadeStale:
    """read_fade's STALE-memory view: corr(S_slow - S_fast), where S_slow is the
    plain sum and S_fast the alpha-discounted sum of the SAME streams. Per-token
    weight is 1 - alpha^(t-i): the current token contributes ZERO, old tokens
    tend to full weight. mass_cum returns the stale key mass, so under read_norm the
    stale read is the normalized estimate over the stale distribution."""

    def __init__(self, slow: Accumulator, fast: Accumulator,
                 log_alpha: torch.Tensor, chunk: int = 64):
        self.slow, self.fast = slow, fast
        self.log_alpha = log_alpha
        self.chunk = chunk

    def __call__(self, q, k, v, key_kind: str):
        return self.slow(q, k, v, key_kind) - self.fast(q, k, v, key_kind)

    def mass_cum(self, keys: torch.Tensor) -> torch.Tensor:
        return keys.cumsum(dim=2) - ema_cumsum(keys, self.log_alpha, self.chunk)


class FadeFast:
    """read_fade_mode='fast': the RECENCY read — corr(S_fast) with the
    alpha-discounted key mass as the read_norm denominator (the normalized read over the
    recency-weighted distribution). Blended at the output in block.py:
    y = y_slow + eta*(y_fast − y_slow). For the linear (non-normalized) read this
    is algebraically the stale form; they differ exactly where Read is nonlinear
    in the state (read_norm / LA1); the current token carries full
    weight (alpha^0 = 1)."""

    def __init__(self, fast: Accumulator, log_alpha: torch.Tensor, chunk: int = 64):
        self.fast = fast
        self.log_alpha = log_alpha
        self.chunk = chunk

    def __call__(self, q, k, v, key_kind: str):
        return self.fast(q, k, v, key_kind)

    def mass_cum(self, keys: torch.Tensor) -> torch.Tensor:
        return ema_cumsum(keys, self.log_alpha, self.chunk)


class NullAccumulator:
    """Zero-memory view: every LA correction (and the read_norm mass) is zero, so
    dual_read returns the READ-side f_theta0(q) exactly — including learn_thresh /
    value_centers read paths, which _forward_theta0 (the WRITE-side forward) does
    not evaluate. Used by read_fade to keep the theta0 base function out of the
    fade term."""

    def __call__(self, q, k, v, key_kind: str):
        return v.new_zeros(*q.shape[:3], v.shape[-1])

    def mass_cum(self, keys: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(keys)

