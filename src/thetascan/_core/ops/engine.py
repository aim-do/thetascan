"""Math engine: closed-form per-token writes, Jacobi-preconditioned solve, dual-form read.

Write (per head, per block, at theta0):
    a = W1 x;  h = sigma(norm(a));  o = W2 h;  r = v - f(k)
    lambda = r / diag(G)  (Jacobi preconditioner on the output-space Gram)
    delta W2 = (s*lambda) h^T ;  delta W1 = (s*S^T W2^T lambda) x^T
Read (dual form): every accumulated delta applied to a query is causal linear
attention — LA1 corrects pre-activations (keys x, values g), LA2 corrects block
outputs (keys h, values lam_hat). Depth-L ResNet: x_{l+1} = x_l + s*u_l, s=1/sqrt(L).
Experimental read_norm_w1 replaces only the LA1 read with a normalized positive
h0-feature read of the same g values, then reapplies sigma before normalized LA2;
it preserves the GN write payload but is not a literal read of sum(delta W1).

Approximations (documented design choices, validated against the oracle):
- Block writes treat downstream blocks as identity (near-identity ResNet
  justification); the output-space Gram sums block-local contributions / L.
- RMSNorm enters derivatives only through its scalar 1/rms factor (rank-one term
  dropped) — consistently in g, Gram, and both read paths.
- write_iters=2: the second Gauss-Newton iteration linearizes at theta0 + this
  token's own delta (Option A preserved).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .nonlin import ELEMENTWISE, silu_d, softmax_S_apply, softmax_W2S
from . import kernel_features
from ..modules.norms import rmsnorm


# ------------------------------------------------------------------ helpers

def _ein_w(W, x, pat):
    """einsum with H=1 broadcast for shared-head weights. W [Hw,*,*], x [B,H,T,*]."""
    if W.shape[0] == 1 and x.shape[1] != 1:
        W = W.expand(x.shape[1], -1, -1)
    return torch.einsum(pat, W, x)


def _gain_for(softmax_gain, cfg, reference: torch.Tensor):
    """Broadcast scalar, ``[H]``, or ``[H,M]`` gain over a head-major tensor."""
    gain = cfg.softmax_gain if softmax_gain is None else softmax_gain
    if not torch.is_tensor(gain):
        return gain
    gain = gain.to(device=reference.device, dtype=reference.dtype)
    if gain.ndim == 0:
        return gain
    if gain.ndim == 1 and gain.shape[0] == reference.shape[1]:
        return gain.view(1, gain.shape[0], *([1] * (reference.ndim - 2)))
    if (
        gain.ndim == 2
        and gain.shape[0] == reference.shape[1]
        and gain.shape[1] == reference.shape[-1]
    ):
        return gain.view(1, gain.shape[0], *([1] * (reference.ndim - 3)), gain.shape[1])
    raise ValueError(
        "learned softmax gain must have shape [n_heads] or "
        "[n_heads, n_features], got "
        f"{tuple(gain.shape)} for reference {tuple(reference.shape)}"
    )


def _shift(x: torch.Tensor, n: int) -> torch.Tensor:
    """Shift right along T (dim 2) by n, zero-fill the front."""
    if n == 0:
        return x
    pad = x.new_zeros(*x.shape[:2], n, *x.shape[3:])
    return torch.cat([pad, x[:, :, :-n]], dim=2)


def _hrot_stream(h, hrot):
    """Feature-space RoPE on an LA2 stream tensor [B,H,T,m]: rotate the first
    rot_dim channels in 2x2 pairs by the per-position angles. The scan dot then
    carries the relative rotation between write key and read query."""
    if hrot is None:
        return h
    cos, sin, rot = hrot
    hr, hp = h[..., :rot], h[..., rot:]
    h1, h2 = hr[..., 0::2], hr[..., 1::2]
    o1 = h1 * cos - h2 * sin
    o2 = h1 * sin + h2 * cos
    out = torch.stack((o1, o2), dim=-1).flatten(-2)
    return torch.cat([out, hp], dim=-1)


def _hrot_pos(h, hrot, t):
    """Single-position variant for the oracle: h [B,H,m], angles taken at t."""
    if hrot is None:
        return h
    cos, sin, rot = hrot
    return _hrot_stream(h.unsqueeze(2),
                        (cos[:, :, t:t + 1], sin[:, :, t:t + 1], rot)).squeeze(2)


@dataclass
class BlockCtx:
    x: torch.Tensor                     # block input [B,H,T,d]
    pre: torch.Tensor                   # normed pre-activation (a branch) [B,H,T,m]
    h: torch.Tensor                     # activation [B,H,T,m]
    inv_a: torch.Tensor                 # 1/rms(a) [B,H,T,1] (ones if act_norm off)
    preb: torch.Tensor | None = None    # swiglu gate branch (normed)
    inv_b: torch.Tensor | None = None


@dataclass
class Streams:
    """Outer-product streams, iter-major: index [it*L + l]."""
    la1: list = field(default_factory=list)   # (keys x [B,H,T,d], values g [B,H,T,m|2m])
    la2: list = field(default_factory=list)   # (keys h [B,H,T,m], values lam_hat [B,H,T,d])


# ------------------------------------------------------------------ forward at theta0

def _block_forward(W1, W2, Wg, cfg, x, delta=None, softmax_gain=None,
                   kernel_controls=None):
    """One memory block. delta: this-token local write (iter-2 path) or None."""
    a = _ein_w(W1, x, "hmd,bhtd->bhtm")
    if delta is not None:
        a = a + delta["g_a"] * (delta["x0"] * x).sum(-1, keepdim=True)
    if cfg.nonlin == "swiglu":
        b = _ein_w(Wg, x, "hmd,bhtd->bhtm")
        if delta is not None:
            b = b + delta["g_b"] * (delta["x0"] * x).sum(-1, keepdim=True)
        pre, inv_a = rmsnorm(a, cfg.eps) if cfg.act_norm else (a, torch.ones_like(a[..., :1]))
        preb, inv_b = rmsnorm(b, cfg.eps) if cfg.act_norm else (b, torch.ones_like(b[..., :1]))
        h = F.silu(preb) * pre
        ctx = BlockCtx(x=x, pre=pre, h=h, inv_a=inv_a, preb=preb, inv_b=inv_b)
    else:
        pre, inv_a = rmsnorm(a, cfg.eps) if cfg.act_norm else (a, torch.ones_like(a[..., :1]))
        if cfg.learn_thresh:
            # Wg slot carries τ [Hw,m,1]; subtract BEFORE ctx so every downstream
            # sigma' evaluates at the shifted pre (τ is constant wrt x — the
            # Jacobian chain is untouched). Cast keeps autocast dtype parity.
            pre = pre - Wg[..., 0].unsqueeze(0).unsqueeze(2).to(pre.dtype)
        if cfg.write_rule == "kernel":
            controls = kernel_controls or {}
            h = kernel_features.feature_map(
                pre,
                cfg,
                softmax_gain=softmax_gain,
                spline_scale=controls.get("spline_scale"),
                score_bias=controls.get("score_bias"),
                relu2_threshold=controls.get("relu2_threshold"),
                relative_threshold=controls.get("relative_threshold"),
                sparse_blend_alpha=controls.get("sparse_blend_alpha"),
            )
        elif cfg.nonlin == "softmax_hidden":
            h = torch.softmax(pre * _gain_for(softmax_gain, cfg, pre), dim=-1)
        else:
            h = ELEMENTWISE[cfg.nonlin][0](pre)
        ctx = BlockCtx(x=x, pre=pre, h=h, inv_a=inv_a)
    u = _ein_w(W2, ctx.h, "hdm,bhtm->bhtd")
    if delta is not None:
        u = u + delta["lam_hat"] * (delta["h0"] * ctx.h).sum(-1, keepdim=True)
    return u, ctx


def _forward_theta0(weights, cfg, k, deltas=None, softmax_gain=None,
                    kernel_controls=None):
    s = 1.0 / math.sqrt(cfg.depth)
    x = k
    ctxs = []
    for l, (W1, W2, Wg) in enumerate(weights):
        u, ctx = _block_forward(
            W1, W2, Wg, cfg, x,
            None if deltas is None else deltas[l],
            softmax_gain=softmax_gain,
            kernel_controls=kernel_controls,
        )
        ctxs.append(ctx)
        x = x + s * u
    return x, ctxs


# ------------------------------------------------------------------ derivative pieces

def _g_from_lambda(W2, cfg, ctx, lam):
    """S^T (W2^T lam): LA1 raw values. [B,H,T,m] or [B,H,T,2m] (swiglu: a then b)."""
    u2 = _ein_w(W2, lam, "hdm,bhtd->bhtm")
    if cfg.nonlin == "softmax_hidden":
        assert cfg.softmax_gain_mode == "fixed", (
            "Jacobian writes use the scalar softmax_gain; learned kernel gains "
            "are valid only for normalized kernel memory"
        )
        # d softmax(g·pre)/d pre = g·[diag(h) − h hᵀ] — the gain scales the Jacobian
        return softmax_S_apply(ctx.h, u2) * (ctx.inv_a * cfg.softmax_gain)
    if cfg.nonlin == "swiglu":
        g_a = F.silu(ctx.preb) * u2 * ctx.inv_a
        g_b = ctx.pre * silu_d(ctx.preb) * u2 * ctx.inv_b
        return torch.cat([g_a, g_b], dim=-1)
    dsig = ELEMENTWISE[cfg.nonlin][1](ctx.pre, ctx.h)
    return dsig * u2 * ctx.inv_a


def _w2s(W2, cfg, ctx) -> torch.Tensor:
    """Q = W2 * dh/d(pre-raw) -> [B,H,T,d,m] (or [..,d,2m] swiglu). Includes 1/rms."""
    if cfg.nonlin == "softmax_hidden":
        assert cfg.softmax_gain_mode == "fixed", (
            "Jacobian writes use the scalar softmax_gain; learned kernel gains "
            "are valid only for normalized kernel memory"
        )
        return softmax_W2S(W2, ctx.h) * (ctx.inv_a * cfg.softmax_gain).unsqueeze(-2)
    W2b = W2.unsqueeze(0).unsqueeze(2)                       # [1,Hw,1,d,m]
    if cfg.nonlin == "swiglu":
        Qa = W2b * (F.silu(ctx.preb) * ctx.inv_a).unsqueeze(-2)
        Qb = W2b * (ctx.pre * silu_d(ctx.preb) * ctx.inv_b).unsqueeze(-2)
        return torch.cat([Qa, Qb], dim=-1)
    dsig = ELEMENTWISE[cfg.nonlin][1](ctx.pre, ctx.h)
    return W2b * (dsig * ctx.inv_a).unsqueeze(-2)


def _lambda_solve(weights, cfg, ctxs, r):
    """Jacobi (diagonal-Gram) preconditioner: elementwise r / diag(G), no solve.

    diag(G) = sum_l [ ||h_l||^2 + ||x_l||^2 * rowsq(Q_l) ] / L
    """
    diag = None
    for l, (W1, W2, Wg) in enumerate(weights):
        c = ctxs[l]
        hh = (c.h * c.h).sum(-1, keepdim=True)
        xx = (c.x * c.x).sum(-1, keepdim=True)
        if cfg.nonlin == "softmax_hidden":
            assert cfg.softmax_gain_mode == "fixed", (
                "Jacobian writes require a fixed scalar softmax_gain"
            )
            # softmax Jacobian J = g(diag h − h hᵀ) is diagonal + rank-1, so the
            # row norms factor exactly: Q[i,:] = g·inv·h⊙(W2[i,:] − c_i) with
            # c = W2·h ⇒ rowsq_i = g²inv²(A_i − 2c_i B_i + c_i² S), A=(W2²)s2,
            # B=W2·s2, S=Σs2, s2=h² — three matvecs, no [d,m] materialization
            # This avoids materializing a large softmax Jacobian tensor.
            s2 = c.h * c.h
            A = _ein_w(W2 * W2, s2, "hdm,bhtm->bhtd")
            Bv = _ein_w(W2, s2, "hdm,bhtm->bhtd")
            cvec = _ein_w(W2, c.h, "hdm,bhtm->bhtd")
            S = s2.sum(-1, keepdim=True)
            gain2 = (c.inv_a * cfg.softmax_gain) ** 2
            rowsq = gain2 * (A - 2.0 * cvec * Bv + cvec * cvec * S)
        else:
            # Elementwise nonlins: Q = W2 * col-scale s, so rowsq(Q) = (W2^2)(s^2)
            # exactly — never materialize the [B,H,T,d,m] Q, which is costly.
            if cfg.nonlin == "swiglu":
                sa = F.silu(c.preb) * c.inv_a
                sb = c.pre * silu_d(c.preb) * c.inv_b
                s2 = sa * sa + sb * sb
            else:
                dsig = ELEMENTWISE[cfg.nonlin][1](c.pre, c.h)
                s = dsig * c.inv_a
                s2 = s * s
            rowsq = _ein_w(W2 * W2, s2, "hdm,bhtm->bhtd")
        term = hh + xx * rowsq
        diag = term if diag is None else diag + term
    return r / (diag / cfg.depth + cfg.eps)


# ------------------------------------------------------------------ write streams

def write_streams(weights, cfg, k, v, hrot=None, softmax_gain=None,
                  kernel_controls=None):
    """Compute per-token write streams.

    Returns (Streams, ctxs at theta0, o = f_theta0(k)).
    fast_w1=False: no dW1/dWg — la1 entries are None (read skips them); sigma' is
    never evaluated (the write is linear in the fast params).
    hrot=(cos, sin, rot_dim): feature-space RoPE — the LA2 keys are rotated so the
    scan dot carries the relative rotation."""
    s = 1.0 / math.sqrt(cfg.depth)
    m = weights[0][1].shape[-1]
    o, ctxs = _forward_theta0(
        weights, cfg, k, softmax_gain=softmax_gain,
        kernel_controls=kernel_controls,
    )
    hks = [_hrot_stream(c.h, hrot) for c in ctxs] if hrot is not None else None
    if cfg.write_rule == "kernel":
        # S += a·v^T: the raw-value deposit of classic kernel regression /
        # Direct kernel payload (pair with read_norm for the normalized read).
        # No residual or lambda solve — each center accumulates the values written
        # near it; the slow W2 (zero-init) trains as a per-center prior on top.
        # dtype: match the LA key stream — autocast runs softmax in fp32 while v
        # arrives bf16, and fla kernels require k/v dtype parity (the GN path
        # launders lam through the solve, kernel memory must cast explicitly).
        if cfg.value_centers:
            # Joint-distribution memory: deposit the value's softmax-partition code r_t (a
            # softmax mixture over the shared codebook C_v = Wg slot) instead of
            # the raw v — S accumulates the key-value joint histogram.
            Cv = weights[0][2]
            Cve = Cv if Cv.shape[0] == v.shape[1] else Cv.expand(v.shape[1], -1, -1)
            vn = v / v.norm(dim=-1, keepdim=True).clamp_min(cfg.eps)
            # Value-codebook temperature is a distinct representation choice.
            # Learned kernel sharpness applies only to key/query addressing, so
            # preserve the configured scalar for this value-side encoder.
            logits = torch.einsum("hvd,bhtd->bhtv", Cve, vn)
            r = torch.softmax(cfg.softmax_gain * logits, -1)
            lam = r.to(v.dtype)
        else:
            lam = v
        # scan dtype: run the LA streams in the COMPUTE dtype (v's — bf16 under
        # autocast, fp64 in the oracle). Softmax h arrives fp32 under autocast;
        # casting it DOWN (below, at the stream append) instead of casting values
        # UP keeps the scans bf16, matching the established GN stream precision;
        # FLA accumulates tl.dot in fp32 regardless.
    else:
        r = v - o
        lam = _lambda_solve(weights, cfg, ctxs, r)
    streams = Streams()
    deltas = []
    for l, (W1, W2, Wg) in enumerate(weights):
        lam_hat = s * lam
        if cfg.fast_w1:
            g = s * _g_from_lambda(W2, cfg, ctxs[l], lam)
            streams.la1.append((ctxs[l].x, g))
            deltas.append({"x0": ctxs[l].x, "h0": ctxs[l].h, "lam_hat": lam_hat,
                           "g_a": g[..., :m],
                           "g_b": g[..., m:] if g.shape[-1] > m else None})
        else:
            streams.la1.append(None)
        key_h = ctxs[l].h if hks is None else hks[l]
        if cfg.write_rule == "kernel":
            key_h = key_h.to(lam_hat.dtype)
        streams.la2.append((key_h, lam_hat))
    if cfg.write_iters == 2:
        o2, ctxs2 = _forward_theta0(
            weights, cfg, k, deltas=deltas, softmax_gain=softmax_gain,
            kernel_controls=kernel_controls,
        )
        r2 = v - o2
        lam2 = _lambda_solve(weights, cfg, ctxs2, r2)
        # The second-step LA2 keys must see the same h-space rotation as the
        # first step: the read rotates its query once, so an unrotated second
        # stream would be scored at absolute rather than relative positions.
        hks2 = [_hrot_stream(c.h, hrot) for c in ctxs2] \
            if hrot is not None else None
        for l, (W1, W2, Wg) in enumerate(weights):
            g2 = s * _g_from_lambda(W2, cfg, ctxs2[l], lam2)
            streams.la1.append((ctxs2[l].x, g2))
            key_h2 = ctxs2[l].h if hks2 is None else hks2[l]
            streams.la2.append((key_h2, s * lam2))
    return streams, ctxs, o


# ------------------------------------------------------------------ dual-form read

def dual_read(weights, cfg, q, streams, acc, hrot=None,
              softmax_gain=None, kernel_controls=None):
    """f_{theta_t}(q_t) in dual form; acc is an ops.interface.Accumulator.
    fast_w1=False: la1 entries are None -> skipped.
    hrot: the LA2 query h~ is rotated for the scan dot only — the base term
    W2 h~ stays unrotated."""
    s = 1.0 / math.sqrt(cfg.depth)
    L, m = cfg.depth, weights[0][1].shape[-1]
    n_iter = cfg.write_iters
    x = q
    for l in range(L):
        W1, W2, Wg = weights[l]
        a = _ein_w(W1, x, "hmd,bhtd->bhtm")
        b = _ein_w(Wg, x, "hmd,bhtd->bhtm") if cfg.nonlin == "swiglu" else None
        feature_mass = None
        if cfg.read_norm_w1:
            # Experimental two-stage GN read.  Address the existing g_i payloads
            # with the same positive slow-reference ReLU-squared features p_i that
            # key LA2, instead of applying the literal signed sum(delta W1) to q.
            # The matched mass makes this a scan-exact normalized kernel read for
            # sum, EMA and the fast/stale fade views.  After adding c1 below we
            # evaluate the nonlinearity again; LA2 therefore sees the corrected h.
            pre0 = rmsnorm(a, cfg.eps)[0] if cfg.act_norm else a
            if cfg.learn_thresh:
                pre0 = pre0 - Wg[..., 0].unsqueeze(0).unsqueeze(2).to(pre0.dtype)
            h0 = ELEMENTWISE["relu2"][0](pre0)
            h0q = _hrot_stream(h0, hrot) if hrot is not None else h0
            feature_keys, _ = streams.la2[l]
            entry = streams.la1[l]
            if entry is None:  # guarded by config; keeps private callers explicit
                raise RuntimeError("read_norm_w1 requires an LA1 GN stream")
            _, gvals = entry
            feature_mass = acc.mass_cum(feature_keys.to(h0q.dtype))
            corr1 = acc(h0q.to(feature_keys.dtype), feature_keys, gvals, "m")
            denom1 = (h0q * feature_mass).sum(-1, keepdim=True)
            a = a + corr1[..., :m] / (denom1 + cfg.eps)
        else:
            for it in range(n_iter):
                entry = streams.la1[it * L + l]
                if entry is None:
                    continue
                keys, vals = entry
                corr = acc(x, keys, vals, "d")
                a = a + corr[..., :m]
                if b is not None:
                    b = b + corr[..., m:]
        if cfg.nonlin == "swiglu":
            pre = rmsnorm(a, cfg.eps)[0] if cfg.act_norm else a
            preb = rmsnorm(b, cfg.eps)[0] if cfg.act_norm else b
            h = F.silu(preb) * pre
        else:
            pre = rmsnorm(a, cfg.eps)[0] if cfg.act_norm else a
            if cfg.learn_thresh:
                pre = pre - Wg[..., 0].unsqueeze(0).unsqueeze(2).to(pre.dtype)
            if cfg.write_rule == "kernel":
                controls = kernel_controls or {}
                h = kernel_features.feature_map(
                    pre,
                    cfg,
                    softmax_gain=softmax_gain,
                    spline_scale=controls.get("spline_scale"),
                    score_bias=controls.get("score_bias"),
                    relu2_threshold=controls.get("relu2_threshold"),
                    relative_threshold=controls.get("relative_threshold"),
                    sparse_blend_alpha=controls.get("sparse_blend_alpha"),
                )
            else:
                gq = _gain_for(softmax_gain, cfg, pre)
                h = torch.softmax(pre * gq, -1) if cfg.nonlin == "softmax_hidden" \
                    else ELEMENTWISE[cfg.nonlin][0](pre)
        u = _ein_w(W2, h, "hdm,bhtm->bhtd")
        hq = _hrot_stream(h, hrot) if hrot is not None else h
        if cfg.read_norm:
            # linear-attention denominator: y_mem = Σw_i λ̂_i / (Σw_i + eps) with
            # w_i = <h_q, h_i> — softmax's convex-combination semantics, exact via
            # a causal key-mass scan with the same temporal weighting as LA2.
            keys, vals = streams.la2[l]
            # hq cast: fla needs q/k/v dtype parity (softmax hq is fp32 under
            # autocast, kernel streams are bf16); the mass cumsum runs at hq's
            # WIDER dtype — the denominator is cheap, keep it accurate.
            # mass_cum dispatches on the accumulator: sum/EMA key mass normally,
            # stale mass (cumsum − ema_cumsum) under read_fade, zeros for the
            # NullAccumulator base read.
            # both_feature_mass uses the same write features and temporal weights
            # for both stages, so one accumulated mass lane serves both denominators.
            kcum = feature_mass if feature_mass is not None \
                else acc.mass_cum(keys.to(hq.dtype))
            if cfg.read_norm_mode == "feature_mass":
                normalized_query = hq / (kcum + cfg.eps)
                mem = acc(normalized_query.to(keys.dtype), keys, vals, "m")
                denom = None
            else:
                corr = acc(hq.to(keys.dtype), keys, vals, "m")
                denom = (hq * kcum).sum(-1, keepdim=True)
                mem = corr / (denom + cfg.eps)
            if cfg.value_centers:
                # decode the retrieved value-center mixture through the codebook
                # (v1: linear decode = convex combination of centers)
                Cve = Wg if Wg.shape[0] == q.shape[1] else Wg.expand(q.shape[1], -1, -1)
                mem = torch.einsum("hvd,bhtv->bhtd", Cve.to(mem.dtype), mem)
            u = u + mem
        else:
            for it in range(n_iter):
                keys, vals = streams.la2[it * L + l]
                u = u + acc(hq, keys, vals, "m")
        x = x + s * u
    return x


# ------------------------------------------------------------------ naive oracle

@torch.no_grad()
def naive_read(weights, cfg, k, v, q, decay_d=None, decay_m=None,
               hrot=None, fade=None, softmax_gain=None, kernel_controls=None):
    """Materializes per-position parameter deltas from the DEFINITIONS (brute-force
    re-sum per position). Never calls dual_read/Accumulator. Small dims only.
    hrot: feature-space RoPE — keys enter via write_streams; the query-side h~ is
    rotated per position for the DW2 term only (base W2 h~ unrotated).
    fade=(eta [H], log_alpha [H]): read_fade — materialize S_fast with weights
    alpha^(t-i) alongside S_slow and emit
        y = Read(S_slow) − eta·(Read(S_slow − S_fast) − f_theta0_read(q)),
    each Read a full brute-force evaluation at that state (sum accumulation only)."""
    streams, _, _ = write_streams(
        weights, cfg, k, v, hrot=hrot, softmax_gain=softmax_gain,
        kernel_controls=kernel_controls,
    )
    sets = [streams]
    B, H, T = q.shape[:3]
    d = v.shape[-1]
    L, m = cfg.depth, weights[0][1].shape[-1]
    s = 1.0 / math.sqrt(L)

    def eval_q(xq, DW1, DWg, DW2, t, Smass=None, N1=None):
        for l in range(L):
            W1, W2, Wg = weights[l]
            W2e = W2 if W2.shape[0] == H else W2.expand(H, -1, -1)
            W1e = W1 if W1.shape[0] == H else W1.expand(H, -1, -1)
            a = torch.einsum("hmd,bhd->bhm", W1e, xq)
            if cfg.read_norm_w1:
                # N1 materializes sum_i omega_i p_i outer-product g_i.  This is
                # intentionally independent of the literal DW1 oracle below.
                an0 = rmsnorm(a, cfg.eps)[0] if cfg.act_norm else a
                if cfg.learn_thresh:
                    an0 = an0 - Wg[..., 0].unsqueeze(0).to(an0.dtype)
                h0 = ELEMENTWISE["relu2"][0](an0)
                h0q = _hrot_pos(h0, hrot, t) if hrot is not None else h0
                corr1 = torch.einsum("bhvk,bhk->bhv", N1[l], h0q)
                den1 = (Smass[l] * h0q).sum(-1, keepdim=True)
                a = a + corr1 / (den1 + cfg.eps)
            elif DW1 is not None:
                a = a + torch.einsum("bhmd,bhd->bhm", DW1[l], xq)
            if cfg.nonlin == "swiglu":
                Wge = Wg if Wg.shape[0] == H else Wg.expand(H, -1, -1)
                bb = torch.einsum("hmd,bhd->bhm", Wge, xq)
                if DWg is not None:
                    bb = bb + torch.einsum("bhmd,bhd->bhm", DWg[l], xq)
                an = rmsnorm(a, cfg.eps)[0] if cfg.act_norm else a
                bn = rmsnorm(bb, cfg.eps)[0] if cfg.act_norm else bb
                h = F.silu(bn) * an
            else:
                an = rmsnorm(a, cfg.eps)[0] if cfg.act_norm else a
                if cfg.learn_thresh:
                    an = an - Wg[..., 0].unsqueeze(0).to(an.dtype)
                if cfg.write_rule == "kernel":
                    controls = kernel_controls or {}
                    h = kernel_features.feature_map(
                        an,
                        cfg,
                        softmax_gain=softmax_gain,
                        spline_scale=controls.get("spline_scale"),
                        score_bias=controls.get("score_bias"),
                        relu2_threshold=controls.get("relu2_threshold"),
                        relative_threshold=controls.get("relative_threshold"),
                        sparse_blend_alpha=controls.get("sparse_blend_alpha"),
                    )
                else:
                    gq = _gain_for(softmax_gain, cfg, an)
                    h = torch.softmax(an * gq, -1) if cfg.nonlin == "softmax_hidden" \
                        else ELEMENTWISE[cfg.nonlin][0](an)
            hq = _hrot_pos(h, hrot, t) if hrot is not None else h
            mem = torch.einsum("bhdm,bhm->bhd", DW2[l], hq)
            if cfg.read_norm:
                if cfg.read_norm_mode == "feature_mass":
                    mem = torch.einsum(
                        "bhdm,bhm->bhd",
                        DW2[l] / (Smass[l].unsqueeze(-2) + cfg.eps),
                        hq,
                    )
                    den = None
                else:
                    den = (Smass[l] * hq).sum(-1, keepdim=True)
                    mem = mem / (den + cfg.eps)
            if cfg.value_centers:
                Cve = Wg if Wg.shape[0] == mem.shape[1] else Wg.expand(mem.shape[1], -1, -1)
                mem = torch.einsum("hvd,bhv->bhd", Cve, mem)
            u = torch.einsum("hdm,bhm->bhd", W2e, h) + mem
            xq = xq + s * u
        return xq

    out = q.new_zeros(B, H, T, d)

    has_la1 = any(e is not None for e in streams.la1)
    Ld = decay_d.cumsum(2) if decay_d is not None else None
    Lm = decay_m.cumsum(2) if decay_m is not None else None
    fe = fl = None
    if fade is not None:
        fe, fl = fade                                # eta [H], log_alpha [H]
    for t in range(T):
        lo = 0
        n = t - lo + 1
        coef = q.new_ones(B, H, n)
        wd = (Ld[:, :, t:t + 1] - Ld[:, :, lo:t + 1]).exp() if Ld is not None else None
        wm = (Lm[:, :, t:t + 1] - Lm[:, :, lo:t + 1]).exp() if Lm is not None else None
        DW1 = ([q.new_zeros(B, H, m, q.shape[-1]) for _ in range(L)]
               if has_la1 and not cfg.read_norm_w1 else None)
        DWg = ([q.new_zeros(B, H, m, q.shape[-1]) for _ in range(L)]
               if has_la1 and not cfg.read_norm_w1 else None)
        N1 = ([q.new_zeros(B, H, m, m) for _ in range(L)]
              if cfg.read_norm_w1 else None)
        # Value dimension derives from the stream: value_centers deposits M_v-dim
        # codes rather than raw d-dimensional values.
        dvs = streams.la2[0][1].shape[-1]
        DW2 = [v.new_zeros(B, H, dvs, m) for _ in range(L)]
        Smass = [None] * L
        if fade is not None:
            # S_fast: the SAME deltas discounted by alpha^(t-i) (weight on the key
            # side, mirroring the decay convention). coef rides the shared value
            # side, so slow and fast weight the same tokens and the stale
            # difference zeroes the current token exactly (1 − alpha^0).
            idx = torch.arange(lo, t + 1, device=q.device, dtype=q.dtype)
            wf = torch.exp(fl.to(q.dtype).view(1, -1, 1, 1)
                           * (t - idx).view(1, 1, -1, 1))          # [1,H,n,1]
            DW1f = ([q.new_zeros(B, H, m, q.shape[-1]) for _ in range(L)]
                    if has_la1 and not cfg.read_norm_w1 else None)
            DWgf = ([q.new_zeros(B, H, m, q.shape[-1]) for _ in range(L)]
                    if has_la1 and not cfg.read_norm_w1 else None)
            N1f = ([q.new_zeros(B, H, m, m) for _ in range(L)]
                   if cfg.read_norm_w1 else None)
            DW2f = [v.new_zeros(B, H, dvs, m) for _ in range(L)]
            Smassf = [None] * L
        for st in sets:
            for j, (kh, lv) in enumerate(st.la2):
                l = j % L                        # iter-major stream layout
                hs = kh[:, :, lo:t + 1] * (wm if wm is not None else 1.0)
                lh = lv[:, :, lo:t + 1] * coef.unsqueeze(-1)
                DW2[l] += torch.einsum("bhnd,bhnm->bhdm", lh, hs)
                if fade is not None:
                    DW2f[l] += torch.einsum("bhnd,bhnm->bhdm", lh,
                                            kh[:, :, lo:t + 1] * wf)
                if cfg.read_norm:
                    # Apply the same temporal coefficients as the LA2 numerator.
                    # This is essential for a normalized EMA memory.
                    mass = (kh[:, :, lo:t + 1] * (wm if wm is not None else 1.0)).sum(2)
                    Smass[l] = mass if Smass[l] is None else Smass[l] + mass
                    if fade is not None:
                        massf = (kh[:, :, lo:t + 1] * wf).sum(2)
                        Smassf[l] = massf if Smassf[l] is None else Smassf[l] + massf
                entry = st.la1[j]
                if entry is None:
                    continue
                kx, gv = entry
                ga = gv[:, :, lo:t + 1, :m] * coef.unsqueeze(-1)
                if cfg.read_norm_w1:
                    # Use the positive LA2 feature key and its m-lane weights,
                    # preserving g_i as the GN value payload.
                    N1[l] += torch.einsum("bhnv,bhnk->bhvk", ga, hs)
                    if fade is not None:
                        N1f[l] += torch.einsum(
                            "bhnv,bhnk->bhvk", ga, kh[:, :, lo:t + 1] * wf)
                else:
                    xs = kx[:, :, lo:t + 1] * (wd if wd is not None else 1.0)
                    DW1[l] += torch.einsum("bhnm,bhnd->bhmd", ga, xs)
                    if fade is not None:
                        DW1f[l] += torch.einsum("bhnm,bhnd->bhmd", ga,
                                                kx[:, :, lo:t + 1] * wf)
                    if gv.shape[-1] > m:
                        gb = gv[:, :, lo:t + 1, m:] * coef.unsqueeze(-1)
                        DWg[l] += torch.einsum("bhnm,bhnd->bhmd", gb, xs)
                        if fade is not None:
                            DWgf[l] += torch.einsum("bhnm,bhnd->bhmd", gb,
                                                    kx[:, :, lo:t + 1] * wf)
        y_t = eval_q(q[:, :, t], DW1, DWg, DW2, t, Smass, N1)
        if fade is not None and cfg.read_fade_mode == "fast":
            # y = Read(S_slow) + eta·(Read(S_fast) − Read(S_slow)): blend toward
            # the recency read (under read_norm: normalized over the alpha-weighted
            # distribution). Linear read: algebraically == the stale form.
            y_f = eval_q(q[:, :, t], DW1f, DWgf, DW2f, t, Smassf, N1f)
            y_t = y_t + fe.to(y_t.dtype).view(1, -1, 1) * (y_f - y_t)
        elif fade is not None:
            # y = Read(S_slow) − eta·(Read(S_stale) − f_theta0_read(q)): each term
            # a full evaluation at that state (LA1 corrections enter the nonlin, so
            # Read is NOT linear in the state — no output-blend shortcut).
            DW1s = None if DW1 is None else [a - b for a, b in zip(DW1, DW1f)]
            DWgs = None if DWg is None else [a - b for a, b in zip(DWg, DWgf)]
            N1s = None if N1 is None else [a - b for a, b in zip(N1, N1f)]
            DW2s = [a - b for a, b in zip(DW2, DW2f)]
            Smass_s = [None if a is None else a - b for a, b in zip(Smass, Smassf)]
            y_s = eval_q(q[:, :, t], DW1s, DWgs, DW2s, t, Smass_s, N1s)
            zeros2 = [torch.zeros_like(w2) for w2 in DW2]
            zmass = [None if a is None else torch.zeros_like(a) for a in Smass]
            zeros1 = None if N1 is None else [torch.zeros_like(n1) for n1 in N1]
            y_0 = eval_q(q[:, :, t], None, None, zeros2, t, zmass, zeros1)
            y_t = y_t - fe.to(y_t.dtype).view(1, -1, 1) * (y_s - y_0)
        out[:, :, t] = y_t
    return out
