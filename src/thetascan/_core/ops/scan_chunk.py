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

Every tile is issued at once, not in a Python loop.  The tiles are a leading
batch axis of one ``[B, H, T/chunk, chunk, chunk]`` score, so the whole scan is
a handful of batched matmuls plus one scan over the tile axis, whatever the
sequence length.  That matters because the per-tile work at the reference width
is far below what one GPU can execute concurrently: a loop over 256 tiles of a
131,072-token sequence issues thousands of kernels that each leave the device
mostly idle, and wall time then tracks the dispatch count rather than the
arithmetic.  Batching also makes ``scan_chunk`` a throughput knob rather than a
launch-count knob, so it can be raised to trade the ``T * chunk`` score term
against nothing but memory.

The tile axis carries the cross-chunk state, and for the two promoted temporal
views that recurrence is associative, so it is computed in parallel rather than
stepped:

* plain sum -- ``S_n = sum_{j<n} U_j`` is an exclusive cumulative sum;
* static per-head retention -- ``S_n = sum_{j<n} alpha^(chunk (n-1-j)) U_j`` is
  the same sum against a fixed geometric Toeplitz weight over the tile axis,
  which is tiny because there are ``T/chunk`` tiles, not ``T`` tokens.

A per-token retention stream has a data-dependent closing factor per tile, so
its state is still stepped over the tile axis -- but only the state is, at two
operations per tile instead of a full tile of work, and every score, payload and
cross term stays batched.  Stepping it keeps the guarantee that no ``1/alpha^i``
factor is ever materialized, which is what a log-space parallel form would give
up.

A single tile is not a wasted case.  At ``chunk >= T`` this backend still avoids
some retained-view overhead, because a static retention builds one
``[1, H, T, T]`` weight shared across the batch where ``quad`` builds
``[B, H, T, T]`` from a per-token cumulative sum, and because a triangular
select replaces an explicit boolean mask tensor.  That case also needs no tile
axis and is evaluated directly.  The plain sum retains the same arithmetic as
``quad``; a retained reduced-precision scan can differ because this backend
forms its phases in at least float32 before casting the finished weights.

Retention handling matches :mod:`scan_quad` structurally: a weight is always the
DIFFERENCE of cumulative logs clamped at zero before ``exp``, so no ``1/alpha^i``
factor is ever materialized and small retentions cannot overflow.  Because the
cumulative log is local to a tile, the exponents stay small; that is a different
rounding order from the global cumulative sum in :mod:`scan_quad`, so parity
with the legacy backend is bounded closeness, not bitwise identity.  Even in one
tile, the retained path intentionally avoids the low-precision cumulative-log
aliasing that remains in the legacy backend.

For a static per-head retention the intra-tile weight is a Toeplitz matrix that
depends on neither the batch nor the tile index, so it is built once per call as
``[H, chunk, chunk]`` instead of ``[B, H, chunk, chunk]``.

Precision policy: every matmul runs in the stream dtype, exactly as the
quadratic backend does, so autocast behaviour is unchanged and a float64 oracle
stays exact.  Retention coordinates, cumulative logs and exponentials are
formed in at least float32 before the finished weights are cast to the stream
dtype; otherwise BF16 aliases positions above 256 inside a 512-wide tile.  The
running state -- the one value that accumulates across the whole sequence, and
the one place a chunked scan could lose accuracy that a single-matmul form
keeps -- is also promoted to at least float32.
"""
from __future__ import annotations

import torch

DEFAULT_CHUNK = 512


def _retention_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Use exact-enough coordinates for retention phases.

    BF16 cannot represent every integer above 256.  Forming a 512-wide
    Toeplitz tile directly in the stream dtype therefore aliases adjacent
    positions before ``exp`` (the same failure mode as low-precision RoPE
    positions).  Retention phases are cheap relative to the score tile, so
    evaluate them in FP32 and round only the finished weights to the stream
    dtype.  Float64 remains the exact oracle path.
    """
    return torch.float64 if dtype == torch.float64 else torch.float32


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
    phase_dtype = _retention_compute_dtype(dtype)
    index = torch.arange(chunk, device=device, dtype=phase_dtype)
    exponent = (index.view(chunk, 1) - index.view(1, chunk)).clamp(min=0.0)
    la = log_alpha.to(device=device, dtype=phase_dtype).reshape(-1, 1, 1)
    tile = torch.exp(la * exponent).tril().to(dtype=dtype)
    carry = torch.exp(la.view(-1, 1) * (index + 1.0)).to(dtype=dtype)
    return tile, carry


def _tile_weights(
    retention: torch.Tensor | None,
    *,
    tiles: int,
    width: int,
    heads: int,
    static: tuple[torch.Tensor, torch.Tensor] | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return this view's intra-tile weight and incoming-state carry.

    Both are shaped to broadcast against a tiled ``[B,H,NT,C,*]`` stream.
    ``None`` for both means the plain causal sum, where every retained weight is
    exactly one.  A static per-head retention reuses one batch- and
    tile-independent Toeplitz tile; a per-token stream builds its own from the
    tile-local cumulative log, which is the same construction the quadratic
    backend uses.
    """
    if retention is None:
        return None, None
    if static is not None:
        tile, carry = static
        return (
            tile.view(1, heads, 1, width, width),
            carry.view(1, heads, 1, width, 1),
        )
    phase_dtype = _retention_compute_dtype(dtype)
    local = retention.view(-1, heads, tiles, width).to(
        dtype=phase_dtype
    ).cumsum(-1)                                                    # [B,H,NT,C]
    tile = (
        local.unsqueeze(-1) - local.unsqueeze(-2)
    ).clamp(max=0.0).exp().to(dtype=dtype)                          # [B,H,NT,C,C]
    return tile, local.exp().to(dtype=dtype).unsqueeze(-1)


def _pad_tiles(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Zero-extend along T so every tile is full width.

    Padded keys and values contribute nothing to any score, payload or state,
    and padded queries only produce output rows that the caller slices away.
    """
    if pad == 0:
        return x
    tail = x.new_zeros(*x.shape[:2], pad, *x.shape[3:])
    return torch.cat([x, tail], dim=2)


def _tile_states(
    update: torch.Tensor,
    *,
    retention: torch.Tensor | None,
    static_log_alpha: torch.Tensor | None,
    closing: torch.Tensor | None,
    width: int,
) -> torch.Tensor:
    """Exclusive scan of per-tile updates: ``S_n = sum_{j<n} decay(n,j) U_j``.

    ``update`` is ``[B,H,NT,Dk,Dv]``.  The plain sum and the static retention
    are associative in closed form and are evaluated in parallel; a per-token
    stream keeps the sequential recurrence, because the alternative is a
    log-space form that materializes the reciprocal decay this module refuses to
    form.
    """
    tiles = update.shape[2]
    if retention is None:
        # Exclusive prefix sum, by shifting the inclusive one rather than
        # subtracting each tile's own term from it.  Subtracting is algebraically
        # identical and one kernel cheaper, but not identical in floating point:
        # ``(U0 + U1) - U1`` is ``U0`` only up to rounding, which makes the state
        # entering a tile depend faintly on that tile's own tokens -- a causal
        # scan that is causal only to within round-off.  The shift has no such
        # term, so a position's output depends on no later position at all.
        inclusive = update.cumsum(2)
        return torch.cat(
            [torch.zeros_like(update[:, :, :1]), inclusive[:, :, :-1]], dim=2
        )
    if static_log_alpha is not None:
        # One [H,NT,NT] geometric weight over the tile axis.  Exponents are
        # clamped before exp, so a small alpha cannot overflow, and the strictly
        # lower triangle keeps the scan exclusive.
        index = torch.arange(tiles, device=update.device, dtype=update.dtype)
        exponent = (index.view(tiles, 1) - index.view(1, tiles) - 1.0).clamp(min=0.0)
        la = static_log_alpha.to(device=update.device, dtype=update.dtype)
        weight = torch.exp(la.reshape(-1, 1, 1) * float(width) * exponent).tril(-1)
        return torch.einsum("hnj,bhjkv->bhnkv", weight, update)
    assert closing is not None
    states = [torch.zeros_like(update[:, :, 0])]
    for index in range(tiles - 1):
        states.append(states[-1] * closing[:, :, index] + update[:, :, index])
    return torch.stack(states, dim=2)


def _single_tile(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    retentions: tuple[torch.Tensor | None, ...],
    statics: list[tuple[torch.Tensor, torch.Tensor] | None],
) -> tuple[torch.Tensor, ...]:
    """The ``chunk >= T`` case, evaluated without a tile axis.

    This keeps the plain-sum arithmetic aligned with :mod:`scan_quad`. Retained
    phases still use this backend's at-least-FP32 precision policy, so a
    reduced-precision result may intentionally differ from the legacy backend.
    """
    scores = q @ k.transpose(-1, -2)
    outputs = []
    for retention, static in zip(retentions, statics):
        if retention is None:
            tile = None
        elif static is not None:
            tile = static[0].unsqueeze(0)
        else:
            phase_dtype = _retention_compute_dtype(q.dtype)
            local = retention[..., 0].to(dtype=phase_dtype).cumsum(-1)
            tile = (
                local.unsqueeze(-1) - local.unsqueeze(-2)
            ).clamp(max=0.0).exp().to(dtype=q.dtype)
        weighted = scores if tile is None else scores * tile
        outputs.append(weighted.tril() @ v)
    return tuple(outputs)


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
    B, heads, T = q.shape[0], q.shape[1], q.shape[2]
    width = min(chunk, T)
    tiles = -(-T // width)

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
        statics.append(_static_weights(retention, width, q.dtype, q.device))

    if tiles == 1:
        return _single_tile(q, k, v, retentions, statics)

    pad = tiles * width - T
    qt = _pad_tiles(q, pad).view(B, heads, tiles, width, -1)
    kt = _pad_tiles(k, pad).view(B, heads, tiles, width, -1)
    vt = _pad_tiles(v, pad).view(B, heads, tiles, width, -1)
    scores = qt @ kt.transpose(-1, -2)          # formed once for all views
    state_dtype = torch.promote_types(q.dtype, torch.float32)
    plain_update: torch.Tensor | None = None

    outputs = []
    for index, retention in enumerate(retentions):
        stream = None
        if retention is not None and statics[index] is None:
            stream = _pad_tiles(retention, pad)
        tile, carry = _tile_weights(
            stream if stream is not None else retention,
            tiles=tiles, width=width, heads=heads, static=statics[index],
            dtype=q.dtype,
        )
        weighted = scores if tile is None else scores * tile
        out = weighted.tril() @ vt

        if tile is None:
            if plain_update is None:
                plain_update = (kt.transpose(-1, -2) @ vt).to(state_dtype)
            update = plain_update
        else:
            # The weight of key j in the state leaving a tile is
            # alpha^(width - 1 - j): the last row of that tile's intra weight.
            keyed = kt * tile[..., width - 1, :].unsqueeze(-1).to(kt.dtype)
            update = (keyed.transpose(-1, -2) @ vt).to(state_dtype)
        states = _tile_states(
            update,
            retention=retention,
            static_log_alpha=None if statics[index] is None else retention,
            # carry at the last local position is alpha^width: the whole tile's
            # retention applied to everything already stored.
            closing=None if carry is None else carry[:, :, :, width - 1:width],
            width=width,
        )
        cross = qt @ states.to(qt.dtype)
        out = out + (cross if carry is None else cross * carry.to(out.dtype))
        outputs.append(out.reshape(B, heads, tiles * width, -1)[:, :, :T])

    return tuple(outputs)


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
