from __future__ import annotations

import torch

from functools import partial

from . import scan_naive, scan_quad, scan_cumsum, scan_chunk, scan_fla

_REGISTRY = {"naive": scan_naive.linattn, "quad": scan_quad.linattn, "cumsum": scan_cumsum.linattn}


def resolve_backend(
    name: str,
    decay_gate: str = "off",
    accumulation: str = "sum",
    device: torch.device | str | None = None,
    require_scalar_decay: bool = False,
) -> str:
    """Pick a scan implementation for ``backend='auto'``.

    Always ``chunk``.  It is matmul-shaped like ``quad`` but retains
    O(T * chunk) activations instead of a full causal score matrix, it is exact
    for both promoted temporal views, and it needs no optional dependency.

    ``auto`` deliberately never selects FLA, for two measured reasons.  Its
    chunked gated backward is refused outright for any retention on Hopper with
    Triton >= 3.4, so every reference preset would fail on its first backward.
    And for the one case it does support, the ungated plain sum, its kernel
    wrapper is closed to the compiler: a compiled model breaks into several
    graphs around it and measured slower than the portable tile at every length
    tried, while also holding more memory at short context.  FLA stays available
    as an explicit choice for an eager long-sequence plain sum.

    A caller who wants the legacy execution contract asks for ``quad``.
    """
    if name != "auto":
        return name
    return "chunk"


def _fn(backend: str, chunk: int = scan_chunk.DEFAULT_CHUNK):
    if backend == "fla":
        return scan_fla.linattn
    if backend == "chunk":
        return partial(scan_chunk.linattn, chunk=chunk)
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


def _retention_stream(retention, like: torch.Tensor) -> torch.Tensor | None:
    """Widen a compact per-head retention to the stream a scan kernel wants."""
    if retention is None or retention.ndim == 4:
        return retention
    heads = like.shape[1]
    return retention.to(dtype=like.dtype, device=like.device).view(
        1, heads, 1, 1
    ).expand(like.shape[0], heads, like.shape[2], 1)


def scan_views(views, q, k, v, key_kind: str) -> tuple[torch.Tensor, ...]:
    """Scan one stream set through several temporal views.

    Every view is a signed combination of plain scans of the SAME streams, which
    is what makes this shareable: the distinct retentions are scanned once and
    each view is assembled from them.  On the chunk backend that single call also
    forms each score tile once for all of them, which is where the saving is --
    the score is about 70% of a view's multiply-accumulates at the reference
    width.  Other backends have no tile to share, so they scan each distinct
    retention separately; the result is identical either way.

    A one-view call is exactly one scan, so the ordinary single-view read is
    unchanged rather than routed through a second implementation.
    """
    terms = [view.scan_terms(key_kind) for view in views]
    slots: dict[object, int] = {}
    retentions: list[object] = []
    for view_terms in terms:
        for _, retention in view_terms:
            slot = None if retention is None else id(retention)
            if slot not in slots:
                slots[slot] = len(retentions)
                retentions.append(retention)

    def zeros() -> torch.Tensor:
        return v.new_zeros(*q.shape[:3], v.shape[-1])

    if not retentions:
        return tuple(zeros() for _ in views)

    # Take the execution context from a view that actually scans: a zero-memory
    # view contributes no retention and must not decide the backend.
    contributing = next(
        view for view, view_terms in zip(views, terms) if view_terms
    )
    backend = contributing.backend
    chunk = getattr(contributing, "chunk", scan_chunk.DEFAULT_CHUNK)
    if backend == "chunk":
        scanned = scan_chunk.linattn_views(q, k, v, tuple(retentions), chunk)
    else:
        fn = _fn(backend, chunk)
        scanned = tuple(
            fn(q, k, v, _retention_stream(retention, q))
            for retention in retentions
        )

    outputs = []
    for view_terms in terms:
        if not view_terms:
            outputs.append(zeros())
            continue
        total = None
        for coefficient, retention in view_terms:
            piece = scanned[slots[None if retention is None else id(retention)]]
            scaled = piece if coefficient == 1.0 else coefficient * piece
            total = scaled if total is None else total + scaled
        outputs.append(total)
    return tuple(outputs)


class Accumulator:
    """Per-mixer-pass accumulation context shared by every LA call.

    ``static_log_alpha`` is the per-head ``[H]`` log-retention that produced the
    ``decay_*`` streams, when there is one.  The streams stay authoritative for
    every backend; the chunk backend additionally uses the compact form to build
    one ``[H, chunk, chunk]`` retention tile instead of a per-batch tile.
    """

    def __init__(self, backend: str, decay_d=None, decay_m=None,
                 eps: float = 1e-6, chunk: int = scan_chunk.DEFAULT_CHUNK,
                 static_log_alpha=None):
        self.backend = backend
        self.decay = {"d": decay_d, "m": decay_m}   # log-alpha streams keyed by key-dim kind
        self.eps = eps
        self.chunk = chunk
        self.static_log_alpha = static_log_alpha

    def mass_cum(self, keys: torch.Tensor) -> torch.Tensor:
        """Cumulative key mass for the read_norm denominator. Plain prefix sum —
        Sum uses a plain prefix. EMA applies its decay to this mass too; fade
        overrides this method with its own fast/stale masses."""
        decay = self.decay["m"]
        if decay is None:
            return keys.cumsum(dim=2)
        if self.static_log_alpha is not None:
            # Same recurrence, but the payload is the key itself: this is the
            # scan with a unit query/key, which ema_cumsum computes directly.
            return ema_cumsum(keys, self.static_log_alpha, self.chunk)
        # A unit query/key scan returns the decayed prefix of each key
        # coordinate, exactly matching the filter on the memory numerator.
        ones = keys.new_ones(*keys.shape[:3], 1)
        return _fn(self.backend, self.chunk)(ones, ones, keys, decay)

    def scan_terms(self, key_kind: str) -> tuple[tuple[float, object], tuple, ...]:
        """The signed retentions whose scans compose this view.

        A view is a linear combination of plain scans of the same streams, so
        stating it as ``(coefficient, retention)`` terms lets several views share
        one score tile.  The compact per-head form is preferred when present,
        because that is what makes the tile batch-independent.
        """
        decay = self.decay[key_kind]
        if decay is None:
            return ((1.0, None),)
        if self.static_log_alpha is not None:
            return ((1.0, self.static_log_alpha),)
        return ((1.0, decay),)

    def __call__(self, q, k, v, key_kind: str):
        return scan_views((self,), q, k, v, key_kind)[0]


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

    @property
    def backend(self) -> str:
        return self.slow.backend

    def scan_terms(self, key_kind: str):
        """S_slow - S_fast, so the stale view needs no scan of its own."""
        return self.slow.scan_terms(key_kind) + tuple(
            (-coefficient, retention)
            for coefficient, retention in self.fast.scan_terms(key_kind)
        )

    def __call__(self, q, k, v, key_kind: str):
        return scan_views((self,), q, k, v, key_kind)[0]

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

    @property
    def backend(self) -> str:
        return self.fast.backend

    def scan_terms(self, key_kind: str):
        return self.fast.scan_terms(key_kind)

    def __call__(self, q, k, v, key_kind: str):
        return scan_views((self,), q, k, v, key_kind)[0]

    def mass_cum(self, keys: torch.Tensor) -> torch.Tensor:
        return ema_cumsum(keys, self.log_alpha, self.chunk)


class NullAccumulator:
    """Zero-memory view: every LA correction (and the read_norm mass) is zero, so
    dual_read returns the READ-side f_theta0(q) exactly — including learn_thresh /
    value_centers read paths, which _forward_theta0 (the WRITE-side forward) does
    not evaluate. Used by read_fade to keep the theta0 base function out of the
    fade term."""

    backend = "chunk"
    chunk = scan_chunk.DEFAULT_CHUNK

    def scan_terms(self, key_kind: str):
        return ()

    def __call__(self, q, k, v, key_kind: str):
        return v.new_zeros(*q.shape[:3], v.shape[-1])

    def mass_cum(self, keys: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(keys)

