"""Chunkwise linear-attention scan: matmul-shaped with O(T * chunk) activations.

Same contract as :func:`scan_quad.linattn` -- an inclusive causal scan with an
optional scalar log-retention -- but the score matrix is formed one
``[chunk, chunk]`` tile at a time and the cross-chunk dependency travels through
an explicit ``[Dk, Dv]`` state instead of a global ``[T, T]`` matrix.  The two
temporal views it supports are exactly the two this library promotes: the plain
causal sum, and the per-head static retention of the recency bank.

Why this is the default rather than the quadratic form.  At the reference shape
``T=1024``, ``Dk=192``, ``Dv=65`` the quadratic form spends about
``T^2 (Dk + Dv)`` multiply-accumulates and retains a ``[B, H, T, T]`` score plus
a second tensor of the same shape once a retention is active.  The chunked form
spends about ``T * chunk * (Dk + Dv) + 2 * T * Dk * Dv`` -- several times fewer
-- and retains ``[B, H, T, chunk]`` tiles plus one state per chunk boundary.
Half of the quadratic score is discarded by the causal mask; the tiled form
never computes it.

Retained bytes are ``T * chunk`` for the tiles and ``(T / chunk) * Dk * Dv`` for
the states, so the footprint minimum sits near ``chunk = sqrt(Dk * Dv)``: 111 to
158 across the reference widths.  Sequence length does not move that optimum,
and both terms are linear in ``T`` rather than quadratic.

The default of 512 is deliberately above that footprint optimum.  Measured on an
H100 in bfloat16, small tiles lose badly to per-tile launch overhead, and
``torch.compile`` does not close that gap; a small tile also raises compile time,
because the loop unrolls statically.  The footprint reduction holds at every
length -- 1.17x-1.54x at ``T=1024`` and 2.80x-3.82x at ``T=4096``.  Lower
``scan_chunk`` toward ``sqrt(Dk * Dv)`` only when memory is the binding
constraint.

A single tile is not a wasted case.  At ``chunk >= T`` this backend still beats
``quad`` on the retained views, because a static retention builds one
``[1, H, T, T]`` weight shared across the batch where ``quad`` builds
``[B, H, T, T]`` from a per-token cumulative sum, and because a triangular
select replaces an explicit boolean mask tensor.

Retention handling matches :mod:`scan_quad` structurally: a weight is always the
DIFFERENCE of cumulative logs clamped at zero before ``exp``, so no ``1/alpha^i``
factor is ever materialized and small retentions cannot overflow.  Because the
cumulative log is local to a tile, the exponents stay small; that is a different
rounding order from the global cumulative sum in :mod:`scan_quad`, so parity
with the legacy backend is bounded closeness, not bitwise identity -- except at
``chunk >= T``, where the two constructions coincide and the backends agree bit
for bit.

For a static per-head retention the intra-tile weight is a Toeplitz matrix that
depends on neither the batch nor the tile index, so it is built once per call as
``[H, chunk, chunk]`` instead of ``[B, H, chunk, chunk]``.

Precision policy: every matmul runs in the stream dtype, exactly as the
quadratic backend does, so autocast behaviour is unchanged and a float64 oracle
stays exact.  Only the running state -- the one value that accumulates across
the whole sequence, and the one place a chunked scan could lose accuracy that a
single-matmul form keeps -- is promoted to at least float32.
"""
from __future__ import annotations

import torch

DEFAULT_CHUNK = 512


def _static_weights(
    log_alpha: torch.Tensor, chunk: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intra-tile retention ``[H,C,C]`` and incoming-state carry ``[H,C]``.

    ``tile[h, i, j] = alpha_h^(i - j)`` for ``j <= i`` and zero above the
    diagonal; ``carry[h, i] = alpha_h^(i + 1)`` is the weight that the state
    entering the tile carries at local position ``i``.  Exponents are clamped
    before ``exp``, which is what stops a small ``alpha`` from producing an
    infinite weight.
    """
    index = torch.arange(chunk, device=device, dtype=dtype)
    exponent = (index.view(chunk, 1) - index.view(1, chunk)).clamp(min=0.0)
    la = log_alpha.to(device=device, dtype=dtype).reshape(-1, 1, 1)
    tile = torch.exp(la * exponent).tril()
    carry = torch.exp(la.view(-1, 1) * (index + 1.0))
    return tile, carry


def _view_weights(
    retention: torch.Tensor | None,
    *,
    start: int,
    width: int,
    heads: int,
    static: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return this view's intra-tile weight and incoming-state carry.

    ``None`` for both means the plain causal sum, where every retained weight is
    exactly one.  A static per-head retention reuses one batch-independent
    Toeplitz tile; a per-token stream builds its own from the tile-local
    cumulative log, which is the same construction the quadratic backend uses.
    """
    if retention is None:
        return None, None
    if static is not None:
        tile, carry = static
        return (
            tile[:, :width, :width].unsqueeze(0),
            carry[:, :width].reshape(1, heads, width, 1),
        )
    local = retention[:, :, start:start + width, 0].cumsum(-1)      # [B,H,C]
    tile = (
        local.unsqueeze(-1) - local.unsqueeze(-2)
    ).clamp(max=0.0).exp()                                          # [B,H,C,C]
    return tile, local.exp().unsqueeze(-1)


def linattn_views(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    retentions: tuple[torch.Tensor | None, ...],
    chunk: int = DEFAULT_CHUNK,
) -> tuple[torch.Tensor, ...]:
    """Scan one stream set through several temporal views at once.

    Each entry of ``retentions`` selects one view: ``None`` is the plain causal
    sum, a ``[H]`` tensor is a static per-head log-retention, and a
    ``[B,H,T,1]`` tensor is a per-token scalar log-retention stream.

    The views share a query and a key stream, so the score tile ``q @ k^T`` is
    formed once per tile and each view only applies its own retention to it.
    That is where the saving is: at the reference width the score is about 70% of
    a view's multiply-accumulates, while the payload contraction and the carried
    state -- the parts that genuinely differ per view -- are the rest.  Each view
    keeps its own state, so nothing about the result changes.

    This is the single scan implementation; :func:`linattn` is the one-view
    convenience wrapper over it.
    """
    if chunk < 1:
        raise ValueError("chunk backend: chunk must be >= 1")
    if not retentions:
        return ()
    heads, T = q.shape[1], q.shape[2]
    width0 = min(chunk, T)

    statics: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    for retention in retentions:
        if retention is None or retention.ndim == 4:
            if retention is not None and retention.shape[-1] != 1:
                raise NotImplementedError(
                    "chunk backend: scalar decay only (v1)"
                )
            statics.append(None)
            continue
        if retention.numel() != heads:
            raise ValueError(
                "chunk backend: a static retention needs one entry per head, "
                f"got {tuple(retention.shape)} for q {tuple(q.shape)}"
            )
        statics.append(_static_weights(retention, width0, q.dtype, q.device))

    state_dtype = torch.promote_types(q.dtype, torch.float32)
    outputs: list[list[torch.Tensor]] = [[] for _ in retentions]
    states: list[torch.Tensor | None] = [None] * len(retentions)

    for start in range(0, T, chunk):
        stop = min(T, start + chunk)
        width = stop - start
        qc = q[:, :, start:stop]
        kc = k[:, :, start:stop]
        vc = v[:, :, start:stop]
        scores = qc @ kc.transpose(-1, -2)          # formed once for all views

        for index, retention in enumerate(retentions):
            tile, carry = _view_weights(
                retention, start=start, width=width, heads=heads,
                static=statics[index],
            )
            weighted = scores if tile is None else scores * tile
            out = weighted.tril() @ vc
            state = states[index]
            if state is not None:
                cross = qc @ state.to(qc.dtype)
                out = out + (
                    cross if carry is None else cross * carry.to(out.dtype)
                )
            outputs[index].append(out)

            if stop < T:
                # The weight of key j in the state leaving this tile is
                # alpha^(width - 1 - j): the last row of the intra tile.
                keyed = kc if tile is None else kc * tile[
                    :, :, width - 1, :
                ].unsqueeze(-1).to(kc.dtype)
                update = (keyed.transpose(-1, -2) @ vc).to(state_dtype)
                if state is None:
                    states[index] = update
                elif carry is None:
                    states[index] = state + update
                else:
                    # carry at the last local position is alpha^width: the whole
                    # tile's retention applied to everything already stored.
                    # Keep the sliced axis so it broadcasts over [B,H,Dk,Dv].
                    closing = carry[:, :, width - 1: width, :].to(state_dtype)
                    states[index] = state * closing + update

    return tuple(torch.cat(chunks, dim=2) for chunks in outputs)


def linattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor | None = None,
    chunk: int = DEFAULT_CHUNK,
    log_alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inclusive causal scan of ``v`` addressed by ``q`` against keys ``k``.

    ``q``/``k`` are ``[B,H,T,Dk]`` and ``v`` is ``[B,H,T,Dv]``.  Supply at most
    one retention: ``log_alpha`` is a static per-head ``[H]`` log-retention (the
    recency bank), ``decay`` is a per-token scalar log-retention stream
    ``[B,H,T,1]``.  ``log_alpha`` takes precedence.  Both are differentiable.
    """
    retention = log_alpha if log_alpha is not None else decay
    return linattn_views(q, k, v, (retention,), chunk)[0]
