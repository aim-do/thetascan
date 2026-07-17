from __future__ import annotations
import torch

_fla = None


def _load():
    global _fla
    if _fla is None:
        import importlib
        mods = {}
        for name, path, fn in [
            ("linear", "fla.ops.linear_attn", "chunk_linear_attn"),
            ("simple_gla", "fla.ops.simple_gla", "chunk_simple_gla"),
            ("gla", "fla.ops.gla", "chunk_gla"),
        ]:
            try:
                mods[name] = getattr(importlib.import_module(path), fn)
            except Exception:
                mods[name] = None
        _fla = mods
    return _fla


def supports(
    decay_gate: str = "off",
    *,
    require_scalar_decay: bool = False,
) -> bool:
    """Return whether the kernels required by an ``auto`` run can be imported.

    Importing the top-level :mod:`fla` package is insufficient: installations
    can omit one or more optional kernel modules.  Resolve those modules before
    selecting FLA so ``backend='auto'`` can retain its portable fallback.
    Explicit ``backend='fla'`` remains fail-fast at the actual call site.
    """
    kernels = _load()
    if decay_gate == "channel":
        return kernels["gla"] is not None
    if decay_gate in ("scalar", "static") or require_scalar_decay:
        return kernels["simple_gla"] is not None
    return kernels["linear"] is not None or kernels["simple_gla"] is not None


def _unwrap(o):
    return o[0] if isinstance(o, tuple) else o


def linattn(q, k, v, decay=None):
    """flash-linear-attention Triton kernels. GPU only.

    fla expects [B, T, H, D] (batch, seqlen, heads, dim); our streams are
    [B, H, T, D], so transpose H<->T -- .contiguous(), fla's Triton kernels
    require it -- on the way in and back out, and on the decay/gate stream too
    (else the gate is applied along the wrong axis: silently wrong output).
    scale=1.0: attention-style 1/sqrt(d) scaling must NOT touch our streams."""
    if not q.is_cuda:
        raise RuntimeError("fla backend requires CUDA tensors")
    fla = _load()
    qt = q.transpose(1, 2).contiguous()
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()
    if decay is None:
        if fla["linear"] is not None:
            o = fla["linear"](qt, kt, vt, scale=1.0, normalize=False)
        elif fla["simple_gla"] is not None:
            g = torch.zeros(q.shape[0], q.shape[2], q.shape[1], device=q.device, dtype=q.dtype)
            o = fla["simple_gla"](qt, kt, vt, g=g, scale=1.0)
        else:
            raise RuntimeError("no usable fla kernel found")
    elif decay.shape[-1] == 1:
        if fla["simple_gla"] is None:
            raise RuntimeError("no usable simple_gla kernel found")
        g = decay.squeeze(-1).transpose(1, 2).contiguous()
        o = fla["simple_gla"](qt, kt, vt, g=g, scale=1.0)
    else:
        if fla["gla"] is None:
            raise RuntimeError("no usable gla kernel found")
        g = decay.transpose(1, 2).contiguous()
        o = fla["gla"](qt, kt, vt, g=g, scale=1.0)
    return _unwrap(o).transpose(1, 2).contiguous()      # [B,T,H,D] -> [B,H,T,D]
