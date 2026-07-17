from __future__ import annotations
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ThetaScanConfig
from ..quant import qcast
from ..ops import engine
from ..ops.interface import (Accumulator, FadeFast, FadeStale, NullAccumulator,
                             resolve_backend)
from .rope import apply_rope, rope_cache
from .norms import l2norm
from .gates import DecayGate, OutGate


_EXPANSION_MAP_DOMAIN = "thetascan-expansion-v1"
_EXPANSION_FINGERPRINT_MAGIC = b"TSXF"
_EXPANSION_FINGERPRINT_SCHEMA_VERSION = 1


def _inverse_softplus(value: float) -> float:
    """Stable scalar inverse of softplus for a strictly positive value."""
    return value + math.log(-math.expm1(-value))


def _row_correlation_penalty(
    weight: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Scale-invariant mean squared correlation between distinct rows."""
    rows = weight / weight.norm(dim=-1, keepdim=True).clamp_min(eps)
    gram = rows @ rows.t()
    count = gram.shape[0]
    if count < 2:
        return weight.new_zeros(())
    off_diagonal = gram * (
        1.0 - torch.eye(count, dtype=gram.dtype, device=gram.device)
    )
    return off_diagonal.square().sum() / (count * (count - 1))


def _expand_kv_groups(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """Expand GQA groups in deterministic [g0,g0,g1,g1,...] order."""
    return x.repeat_interleave(repeats, dim=1)


def _fixed_rademacher(
    shape: tuple[int, ...], *, key: str, role: str
) -> torch.Tensor:
    """Return an RNG-independent fixed sign map derived only from its key.

    SHAKE256 makes the construction stable across global PyTorch RNG state,
    devices, and process order.  The buffers produced from these maps are
    deliberately non-persistent: a checkpoint recreates them from the config
    key instead of spending artifact bytes on non-trainable random bits.
    """
    count = math.prod(shape)
    domain = _fixed_rademacher_domain(shape, key=key, role=role)
    raw = bytearray(hashlib.shake_256(domain).digest(count))
    bits = torch.frombuffer(raw, dtype=torch.uint8).clone()
    signs = bits.bitwise_and(1).to(torch.float32).mul_(2.0).sub_(1.0)
    return signs.reshape(shape)


def _fixed_rademacher_domain(
    shape: tuple[int, ...], *, key: str, role: str
) -> bytes:
    """Return the versioned domain that identifies one fixed sign stream."""
    return (
        f"{_EXPANSION_MAP_DOMAIN}\0"
        f"{key}\0{role}\0{','.join(str(value) for value in shape)}"
    ).encode("utf-8")


def _expansion_fingerprint(
    *, key: str, effective_hidden: int, feature_expansion: int, depth: int
) -> torch.Tensor:
    """Compact identity of every derived expansion map in a module.

    The maps themselves are intentionally absent from checkpoints.  This
    versioned fingerprint makes their derivation part of the strict state-dict
    contract without adding artifact-sized persistent buffers.  Any change to
    the map construction must use a new map domain and fingerprint schema.
    """
    if feature_expansion <= 1:
        raise ValueError("an expansion fingerprint requires feature_expansion > 1")
    base_hidden = effective_hidden // feature_expansion
    digest = hashlib.sha256()
    digest.update(b"thetascan-expansion-checkpoint\0")
    digest.update(_EXPANSION_MAP_DOMAIN.encode("ascii"))
    for layer in range(depth):
        descriptors = (
            _fixed_rademacher_domain(
                (1, effective_hidden, base_hidden),
                key=key,
                role=f"w1-expand-layer-{layer}",
            ),
            _fixed_rademacher_domain(
                (1, base_hidden, effective_hidden),
                key=key,
                role=f"w2-expand-layer-{layer}",
            ),
        )
        for descriptor in descriptors:
            digest.update(len(descriptor).to_bytes(8, "big"))
            digest.update(descriptor)
    encoded = (
        _EXPANSION_FINGERPRINT_MAGIC
        + _EXPANSION_FINGERPRINT_SCHEMA_VERSION.to_bytes(4, "big")
        + digest.digest()
    )
    return torch.tensor(tuple(encoded), dtype=torch.uint8)


def _fixed_row_map(
    shape: tuple[int, ...], *, key: str, role: str, row_norm: float = 1.0,
    center_rows: bool = True,
) -> torch.Tensor:
    """Deterministic rows with a prescribed L2 norm.

    Centering must not be used for an expansion map: centering every row of an
    ``m x base`` expansion makes the all-ones base direction an exact null
    vector and silently wastes one counted trainable direction.
    """
    matrix = _fixed_rademacher(shape, key=key, role=role)
    if center_rows:
        matrix = matrix - matrix.mean(dim=-1, keepdim=True)
    fallback = _fixed_rademacher(shape, key=key, role=f"{role}-fallback")
    norm = matrix.norm(dim=-1, keepdim=True)
    matrix = torch.where(norm > 0.0, matrix, fallback)
    return matrix * (float(row_norm) / matrix.norm(dim=-1, keepdim=True))


class ThetaScan(nn.Module):
    """Drop-in token mixer: x [B,T,D] -> y [B,T,D].

    Trainable ("slow") parameters: qkv/out projections, theta0 = per-block memory
    weights, gates. Fast weights (per-token deltas) are runtime activations."""

    def __init__(self, cfg: ThetaScanConfig):
        super().__init__()
        self.cfg = cfg
        D, H, d, m_total, L = (
            cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.mem_hidden, cfg.depth
        )
        m = m_total
        if cfg.softmax_gain_mode == "learned_per_head":
            raw0 = _inverse_softplus(float(cfg.softmax_gain))
            self.kernel_sharpness_raw = nn.Parameter(torch.full((H,), raw0))
        elif cfg.softmax_gain_mode == "learned_per_feature":
            raw0 = _inverse_softplus(float(cfg.softmax_gain))
            self.kernel_sharpness_raw = nn.Parameter(torch.full((H, m), raw0))
        else:
            self.register_parameter("kernel_sharpness_raw", None)
        if cfg.kernel_score_bias:
            self.kernel_score_bias = nn.Parameter(torch.zeros(H, m))
        else:
            self.register_parameter("kernel_score_bias", None)
        if cfg.kernel_relu2_threshold_mode == "learned_per_head":
            # One unconstrained threshold per head. Zero initialization exactly
            # reproduces the ordinary relu2_ridge map.
            self.kernel_relu2_threshold = nn.Parameter(torch.zeros(H))
        else:
            self.register_parameter("kernel_relu2_threshold", None)
        if cfg.kernel_sparsity != "none":
            rho0 = float(cfg.kernel_relative_threshold_init)
            raw0 = math.log(rho0 / (1.0 - rho0))
            self.kernel_relative_threshold_raw = nn.Parameter(
                torch.full((H,), raw0)
            )
        else:
            self.register_parameter("kernel_relative_threshold_raw", None)
        if cfg.kernel_sparsity == "relative_st_blend":
            self.kernel_sparse_blend_raw = nn.Parameter(
                torch.full((H,), float(cfg.kernel_sparse_blend_init))
            )
        else:
            self.register_parameter("kernel_sparse_blend_raw", None)
        if cfg.bspline_scale_mode == "learned_per_head":
            scale0 = _inverse_softplus(float(cfg.bspline_scale))
            self.bspline_scale_raw = nn.Parameter(torch.full((H,), scale0))
        else:
            self.register_parameter("bspline_scale_raw", None)
        if cfg.key_value_heads is not None:
            # Transformer-GQA layout: every query head has its own projection,
            # while grouped K/V projections are repeated over contiguous head
            # groups. For H=8,G=4 this is pairwise official GQA.
            G = cfg.key_value_heads
            self.proj_qkv = None
            self.proj_q = nn.Linear(D, H * d, bias=False)
            self.proj_k = nn.Linear(D, G * d, bias=False)
            self.proj_v = nn.Linear(D, G * d, bias=False)
        elif cfg.share_kq:
            # ONE k/q group shared by every head (the Mamba-3 ngroups=1 structure):
            # slim d-dim q,k projections broadcast across heads + per-head static
            # biases (the B_bias/C_bias analog); values stay per-head. Zero-init
            # biases reproduce exact head symmetry at init.
            self.proj_qkv = None
            self.proj_q1 = nn.Linear(D, d, bias=False)
            self.proj_k1 = nn.Linear(D, d, bias=False)
            self.proj_v = nn.Linear(D, H * d, bias=False)
            self.kq_bias = nn.Parameter(torch.zeros(2, H, d))
        else:
            self.proj_qkv = nn.Linear(D, 3 * H * d, bias=False)
        self.proj_out = nn.Linear(H * d, D, bias=False)
        Hw = H

        def w(*shape, fan):
            return nn.Parameter(torch.randn(*shape) / math.sqrt(fan))

        def theta0(hidden: int):
            """One memory net: (W1, W2, Wg) ParameterLists over depth."""
            din = d
            projection_hidden = (
                hidden // cfg.bspline_basis_count
                if cfg.kernel_kind == "projected_bspline"
                else hidden
            )
            W1 = nn.ParameterList(
                [w(H, projection_hidden, din, fan=din) for _ in range(L)]
            )
            # W2 = 0: blocks start near-identity, the write residual r = v
            # exactly, and the read starts as pure associative retrieval of v.
            W2 = nn.ParameterList(
                [nn.Parameter(torch.zeros(Hw, d, hidden)) for _ in range(L)]
            )
            if cfg.nonlin == "swiglu":
                Wg = nn.ParameterList([w(Hw, hidden, d, fan=d) for _ in range(L)])
            elif cfg.learn_thresh:
                # τ rides the (otherwise free) Wg slot: per-unit learnable threshold
                # sigma(pre − τ), zero-init = the exact base nonlin at step 0.
                Wg = nn.ParameterList([nn.Parameter(torch.zeros(H, hidden, 1))
                                       for _ in range(L)])
            elif cfg.value_centers:
                # C_v value codebook rides the Wg slot: head-SHARED [1, M_v, d]
                # (share_kq spirit — one value vocabulary, heads differ by keys).
                Wg = nn.ParameterList([w(1, cfg.value_centers, d, fan=d)
                                       for _ in range(L)])
            else:
                Wg = None
            return W1, W2, Wg

        self.W1, self.W2, self.Wg = theta0(m)
        # Random feature expansion: trainable W1/W2 stay at the narrow base
        # width m/f while fixed, key-derived sign maps expand them to the full
        # effective width m in _weights().  Fast state and every learned
        # control (thresholds, sharpness) live at the effective width; the
        # buffers are non-persistent and are derived from expansion_key when
        # the target module is constructed.
        self._w1_expansion_buffer_names: tuple[str, ...] = ()
        self._w2_expansion_buffer_names: tuple[str, ...] = ()
        # None leaves dense checkpoints bit-for-bit unchanged. Expanded
        # checkpoints carry only this 40-byte persistent identity, never the
        # potentially large maps themselves.
        self.register_buffer("_expansion_fingerprint", None)
        if cfg.feature_expansion > 1:
            base_hidden = m // cfg.feature_expansion
            din = d
            self.W1 = nn.ParameterList(
                [w(H, base_hidden, din, fan=din) for _ in range(L)]
            )
            self.W2 = nn.ParameterList(
                [nn.Parameter(torch.zeros(Hw, d, base_hidden)) for _ in range(L)]
            )
            w1_names = []
            w2_names = []
            for layer in range(L):
                w1_name = f"_expand_w1_{layer}"
                w2_name = f"_expand_w2_{layer}"
                self.register_buffer(
                    w1_name,
                    _fixed_row_map(
                        (1, m, base_hidden),
                        key=cfg.expansion_key,
                        role=f"w1-expand-layer-{layer}",
                        center_rows=False,
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    w2_name,
                    _fixed_row_map(
                        (1, base_hidden, m),
                        key=cfg.expansion_key,
                        role=f"w2-expand-layer-{layer}",
                        center_rows=False,
                    ),
                    persistent=False,
                )
                w1_names.append(w1_name)
                w2_names.append(w2_name)
            self._w1_expansion_buffer_names = tuple(w1_names)
            self._w2_expansion_buffer_names = tuple(w2_names)
            self._expansion_fingerprint = _expansion_fingerprint(
                key=cfg.expansion_key,
                effective_hidden=m,
                feature_expansion=cfg.feature_expansion,
                depth=L,
            )
        self.decay = DecayGate(D, H) if cfg.decay_gate != "off" else None
        # value_mlp: head-shared residual value featurizer on the WRITE deposit;
        # zero-init out -> the deposit is exactly v at step 0 (identity-at-init).
        self.vmlp_in = self.vmlp_out = None
        if cfg.value_mlp_mult > 0:
            vh = max(1, int(cfg.value_mlp_mult * d))
            self.vmlp_in = nn.Linear(d, vh)
            self.vmlp_out = nn.Linear(vh, d)
            nn.init.zeros_(self.vmlp_out.weight)
            nn.init.zeros_(self.vmlp_out.bias)
        self.ogate = OutGate(D, H, d) if cfg.out_gate else None
        # read_fade: one or two per-head EMA retentions (sigmoid-parameterized)
        # plus zero-init blends. The default one-branch configuration deliberately
        # keeps the historical [H] state-dict shapes and initialization path.
        if cfg.read_fade:
            alphas = cfg.resolved_fade_alphas
            if cfg.fade_branches == 1:
                a0 = math.log(alphas[0] / (1.0 - alphas[0]))
                self.fade_alpha = nn.Parameter(torch.full((H,), a0))
                self.fade_eta = nn.Parameter(torch.zeros(H))
            else:
                raw = [math.log(alpha / (1.0 - alpha)) for alpha in alphas]
                self.fade_alpha = nn.Parameter(
                    torch.tensor(raw).view(-1, 1).expand(-1, H).clone()
                )
                self.fade_eta = nn.Parameter(torch.zeros(cfg.fade_branches, H))
        else:
            self.fade_alpha = None
            self.fade_eta = None
        self._backend = None
        self._backend_device_type = None

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Validate compact identities before loading expanded parameters.

        ``strict=False`` deliberately remains an escape hatch for legacy
        checkpoints that have no fingerprint.  A checkpoint that does contain
        a malformed or mismatched fingerprint is never accepted: silently
        combining its learned factors with this module's different fixed maps
        would change the represented model.
        """
        fingerprint = self._expansion_fingerprint
        fingerprint_key = prefix + "_expansion_fingerprint"
        if fingerprint is not None and fingerprint_key in state_dict:
            loaded = state_dict[fingerprint_key]
            expected_size = fingerprint.numel()
            loaded_bytes = None
            if (
                torch.is_tensor(loaded)
                and loaded.dtype == torch.uint8
                and loaded.ndim == 1
                and loaded.numel() == expected_size
            ):
                try:
                    loaded_bytes = bytes(loaded.detach().cpu().tolist())
                except (RuntimeError, TypeError, ValueError):
                    loaded_bytes = None

            expected_bytes = bytes(fingerprint.detach().cpu().tolist())
            if loaded_bytes != expected_bytes:
                if loaded_bytes is None:
                    reason = "malformed compact fingerprint"
                elif loaded_bytes[:4] != _EXPANSION_FINGERPRINT_MAGIC:
                    reason = "unknown compact fingerprint encoding"
                else:
                    schema_version = int.from_bytes(loaded_bytes[4:8], "big")
                    if schema_version != _EXPANSION_FINGERPRINT_SCHEMA_VERSION:
                        reason = (
                            "unsupported fingerprint schema version "
                            f"{schema_version} (expected "
                            f"{_EXPANSION_FINGERPRINT_SCHEMA_VERSION})"
                        )
                    else:
                        reason = (
                            "fingerprint mismatch (different expansion_key, "
                            "map shapes, or map derivation)"
                        )
                error_msgs.append(
                    f'{fingerprint_key}: cannot load random feature expansion: '
                    f"{reason}"
                )
                # Abort before PyTorch copies any direct parameters or recurses
                # into W1/W2. Catching the load error must never leave learned
                # factors from one expansion paired with another key's maps.
                raise RuntimeError(error_msgs[-1])

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def kernel_sharpness(self) -> torch.Tensor | None:
        """Return positive learned head/feature sharpness, or ``None`` in fixed mode.

        The fixed path deliberately returns ``None`` so the engine continues to
        use the scalar config path bit-for-bit. Softplus keeps the
        learned values positive and is well behaved for both small and large
        gains; the tiny floor protects reduced-precision underflow.
        """
        if self.kernel_sharpness_raw is None:
            return None
        value = F.softplus(self.kernel_sharpness_raw)
        return value.clamp_min(torch.finfo(value.dtype).tiny)

    def kernel_relative_threshold(self) -> torch.Tensor | None:
        """Return the learned relative-to-maximum cutoff in ``(0, 1)``."""
        if self.kernel_relative_threshold_raw is None:
            return None
        return torch.sigmoid(self.kernel_relative_threshold_raw)

    def kernel_sparse_blend_alpha(self) -> torch.Tensor | None:
        """Return the bounded dense-to-sparse blend with a recoverable gradient.

        The forward value is ``clamp(raw, 0, 1)`` and is therefore exactly zero
        at the default initialization. Backward follows ``sigmoid(raw)`` so a
        gate that is at either clamp boundary can move back into the open range.
        """
        if self.kernel_sparse_blend_raw is None:
            return None
        bounded = self.kernel_sparse_blend_raw.clamp(0.0, 1.0)
        surrogate = torch.sigmoid(self.kernel_sparse_blend_raw)
        return bounded.detach() + surrogate - surrogate.detach()

    def bspline_scale(self) -> torch.Tensor | None:
        """Return a positive learned coordinate scale, or ``None`` when fixed."""
        raw = self.bspline_scale_raw
        if raw is None:
            return None
        value = F.softplus(raw)
        return value.clamp_min(torch.finfo(value.dtype).tiny)

    def kernel_controls(self) -> dict[str, torch.Tensor | None]:
        """Controls shared by key and query feature-map evaluation."""
        return {
            "spline_scale": self.bspline_scale(),
            "score_bias": self.kernel_score_bias,
            "relu2_threshold": self.kernel_relu2_threshold,
            "relative_threshold": self.kernel_relative_threshold(),
            "sparse_blend_alpha": self.kernel_sparse_blend_alpha(),
        }

    def key_query_feature_parameters(self):
        """Yield the parameters defining the shared key/query feature map.

        This explicit ownership boundary avoids relying on private parameter
        name prefixes when the public API freezes kernel feature parameters.
        """
        seen: set[int] = set()
        for parameter_list_name in ("W1",):
            parameter_list = getattr(self, parameter_list_name, None)
            if parameter_list is not None:
                for parameter in parameter_list:
                    if id(parameter) not in seen:
                        seen.add(id(parameter))
                        yield parameter
        for name in (
            "kernel_sharpness_raw", "kernel_score_bias",
            "kernel_relu2_threshold",
            "kernel_relative_threshold_raw", "kernel_sparse_blend_raw",
            "bspline_scale_raw",
        ):
            parameter = getattr(self, name, None)
            if parameter is not None and id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter

    def fade_blends(self) -> torch.Tensor | None:
        """Return the actual signed blend(s), preserving the stored raw parameter.

        One-branch mode returns ``[H]``; two-branch
        mode returns ``[2,H]``. ``tanh`` constrains each signed blend to (-1, 1)
        without changing its zero initialization or its initial derivative.
        """
        if self.fade_eta is None:
            return None
        if self.cfg.fade_blend_mode == "tanh":
            return torch.tanh(self.fade_eta)
        return self.fade_eta

    def ortho_loss(self) -> torch.Tensor:
        """Optional regularizer (add to the training loss): scale-invariant
        decorrelation of theta0 — pairwise row-cosine^2 inside each memory matrix
        (hidden units spread out instead of collapsing onto one neuron) and pairwise
        cosine^2 of flattened per-head matrices (heads stay diverse). Cosines, not
        ||WW^T - I||: norms stay free (W2 is zero-initialized; W1 has m > d rows,
        which can only form a tight frame, not an orthonormal set)."""
        cfg = self.cfg
        out = torch.zeros((), device=self.W2[0].device, dtype=self.W2[0].dtype)
        if (cfg.ortho_intra == 0.0 and cfg.ortho_inter == 0.0
                and cfg.value_mlp_ortho == 0.0):
            return out
        # Wg stores an actual feature/value matrix for SwiGLU and value anchors,
        # but in learned-threshold mode its last axis has width one and contains
        # scalar thresholds. Penalizing those scalars as feature directions is
        # meaningless, so omit only that storage role.
        matrix_lists = [self.W1, self.W2]
        if not cfg.learn_thresh:
            matrix_lists.append(self.Wg)
        mats = [W for lst in matrix_lists if lst is not None for W in lst]
        if cfg.ortho_intra:
            acc = 0.0
            eligible = [W for W in mats if W.shape[-2] > 1]
            for W in eligible:
                Wn = W / W.norm(dim=-1, keepdim=True).clamp_min(cfg.eps)
                G = Wn @ Wn.transpose(-1, -2)                     # [Hw, r, r]
                r = G.shape[-1]
                off = G * (1.0 - torch.eye(r, device=G.device, dtype=G.dtype))
                acc = acc + off.square().sum() / (G.shape[0] * r * (r - 1))
            if eligible:
                out = out + cfg.ortho_intra * acc / len(eligible)
        if cfg.ortho_inter and cfg.n_heads > 1:
            acc = 0.0
            eligible = [W for W in mats if W.shape[0] > 1]
            for W in eligible:
                V = W.flatten(1)                                  # [H, r*c]
                Vn = V / V.norm(dim=-1, keepdim=True).clamp_min(cfg.eps)
                Gh = Vn @ Vn.t()
                Hh = Gh.shape[0]
                off = Gh * (1.0 - torch.eye(Hh, device=Gh.device, dtype=Gh.dtype))
                acc = acc + off.square().sum() / (Hh * (Hh - 1))
            if eligible:
                out = out + cfg.ortho_inter * acc / len(eligible)
        if cfg.value_mlp_ortho and self.vmlp_in is not None:
            out = out + cfg.value_mlp_ortho * (
                _row_correlation_penalty(self.vmlp_in.weight, cfg.eps)
                + _row_correlation_penalty(self.vmlp_out.weight, cfg.eps)
            ) / 2.0
        return out

    def _weights(self):
        pd = self.cfg.param_dtype
        W1l, W2l, Wgl = (self.W1, self.W2, self.Wg)
        if self.cfg.feature_expansion > 1:
            out = []
            for l in range(self.cfg.depth):
                W1 = torch.matmul(
                    qcast(getattr(self, self._w1_expansion_buffer_names[l]), pd),
                    qcast(W1l[l], pd),
                )
                W2 = torch.matmul(
                    qcast(W2l[l], pd),
                    qcast(getattr(self, self._w2_expansion_buffer_names[l]), pd),
                )
                out.append((
                    W1, W2, qcast(Wgl[l], pd) if Wgl is not None else None
                ))
            return out
        return [(qcast(W1l[l], pd) if W1l is not None else None,
                 qcast(W2l[l], pd),
                 qcast(Wgl[l], pd) if Wgl is not None else None)
                for l in range(self.cfg.depth)]

    def _lin(self, x, weight):
        pd = self.cfg.param_dtype
        return F.linear(x, weight if pd == "fp32" else qcast(weight, pd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, T, _ = x.shape
        H, d = cfg.n_heads, cfg.head_dim
        if cfg.key_value_heads is not None:
            G = cfg.key_value_heads
            repeats = H // G
            q = self._lin(x, self.proj_q.weight).view(B, T, H, d).transpose(1, 2)
            k = self._lin(x, self.proj_k.weight).view(B, T, G, d).transpose(1, 2)
            v = self._lin(x, self.proj_v.weight).view(B, T, G, d).transpose(1, 2)
            k = _expand_kv_groups(k, repeats)
            v = _expand_kv_groups(v, repeats)
        elif cfg.share_kq:
            # shared k/q group -> broadcast to heads + per-head biases (cast to the
            # compute dtype: fp32 bias + bf16 activations must not upcast the path)
            kqb = self.kq_bias.to(dtype=x.dtype)
            q = self._lin(x, self.proj_q1.weight).unsqueeze(1) + kqb[0][None, :, None, :]
            k = self._lin(x, self.proj_k1.weight).unsqueeze(1) + kqb[1][None, :, None, :]
            v = self._lin(x, self.proj_v.weight).view(B, T, H, d).transpose(1, 2)
        else:
            qkv = self._lin(x, self.proj_qkv.weight)
            q, k, v = qkv.view(B, T, 3, H, d).permute(2, 0, 3, 1, 4)    # each [B,H,T,d]
        if cfg.rope_placement == "input":
            q = apply_rope(q, cfg.rope, cfg.rope_frac, cfg.rope_base)
            k = apply_rope(k, cfg.rope, cfg.rope_frac, cfg.rope_base)
        if cfg.qk_norm:
            q, k = l2norm(q, cfg.eps), l2norm(k, cfg.eps)
        k_w, v_w = k, v
        if self.vmlp_in is not None:
            v_w = v_w + self.vmlp_out(F.silu(self.vmlp_in(v_w)))
        hrot = None
        if cfg.rope_placement == "feature" and cfg.rope != "none":
            m = cfg.mem_hidden
            max_rot = m // 2 * 2
            feature_rot = max_rot if cfg.rope == "full" else min(
                max_rot, max(2, int(m * cfg.rope_frac) // 2 * 2)
            )
            if feature_rot:
                hcos, hsin = rope_cache(
                    T, feature_rot, cfg.rope_base, x.device, x.dtype
                )
                # Keep the common [B,H,T,pairs] hrot shape used by the
                # data-dependent rotation path and the direct-form oracle.
                hcos = hcos.view(1, 1, T, -1).expand(B, H, T, -1)
                hsin = hsin.view(1, 1, T, -1).expand(B, H, T, -1)
                hrot = (hcos, hsin, feature_rot)
        decay_d = decay_m = None
        if self.decay is not None:
            decay_d, decay_m = self.decay(x)
        if self._backend is None or self._backend_device_type != x.device.type:
            self._backend = resolve_backend(
                cfg.backend,
                cfg.decay_gate,
                cfg.accumulation,
                device=x.device,
                require_scalar_decay=cfg.read_fade,
            )
            self._backend_device_type = x.device.type
        acc = Accumulator(self._backend, decay_d=decay_d, decay_m=decay_m,
                          eps=cfg.eps)
        weights = self._weights()
        kernel_sharpness = self.kernel_sharpness()
        kernel_controls = self.kernel_controls()
        streams, _, _ = engine.write_streams(
            weights, cfg, k_w, v_w, hrot=hrot,
            softmax_gain=kernel_sharpness,
            kernel_controls=kernel_controls,
        )
        fade_log_alphas: list[torch.Tensor] = []
        fade_fasts: list[Accumulator] = []
        fade_etas: list[torch.Tensor] = []
        if cfg.read_fade:
            alpha_raw = self.fade_alpha
            eta_actual = self.fade_blends()
            if alpha_raw.ndim == 1:
                alpha_raw = alpha_raw.unsqueeze(0)
                eta_actual = eta_actual.unsqueeze(0)
            for branch in range(cfg.fade_branches):
                log_alpha = (-F.softplus(-alpha_raw[branch])).to(dtype=x.dtype)
                la_s = log_alpha.view(1, H, 1, 1).expand(B, H, T, 1)
                fade_log_alphas.append(log_alpha)
                fade_etas.append(eta_actual[branch])
                fade_fasts.append(Accumulator(
                    self._backend,
                    decay_d=la_s,
                    decay_m=la_s,
                    eps=cfg.eps,
                ))

        def read_with_temporal(read_weights, query, read_streams, *,
                               controls=kernel_controls):
            """Read one stream set through the configured slow/fade view."""
            out = engine.dual_read(
                read_weights, cfg, query, read_streams, acc,
                hrot=hrot,
                softmax_gain=kernel_sharpness,
                kernel_controls=controls,
            )
            if not cfg.read_fade:
                return out

            if cfg.read_fade_mode == "fast":
                result = out
                for fast_acc, log_alpha, eta_head in zip(
                    fade_fasts, fade_log_alphas, fade_etas
                ):
                    facc = FadeFast(
                        fast_acc, log_alpha, chunk=min(cfg.chunk_size, 64)
                    )
                    out_fast = engine.dual_read(
                        read_weights, cfg, query, read_streams, facc,
                        hrot=hrot,
                        softmax_gain=kernel_sharpness,
                        kernel_controls=controls,
                    )
                    eta = eta_head.view(1, -1, 1, 1).to(dtype=out.dtype)
                    result = result + eta * (out_fast - out)
                return result

            # The stale view already has zero self weight.  Subtract the base
            # theta0 read so fade changes memory content only.
            out_base = engine.dual_read(
                read_weights, cfg, query, read_streams, NullAccumulator(),
                hrot=hrot,
                softmax_gain=kernel_sharpness,
                kernel_controls=controls,
            )
            result = out
            for fast_acc, log_alpha, eta_head in zip(
                fade_fasts, fade_log_alphas, fade_etas
            ):
                out_stale = engine.dual_read(
                    read_weights, cfg, query, read_streams,
                    FadeStale(acc, fast_acc, log_alpha),
                    hrot=hrot,
                    softmax_gain=kernel_sharpness,
                    kernel_controls=controls,
                )
                eta = eta_head.view(1, -1, 1, 1).to(dtype=out.dtype)
                result = result - eta * (out_stale - out_base)
            return result

        y = read_with_temporal(weights, q, streams)
        if self.ogate is not None:
            y = y * self.ogate(x)
        y = y.transpose(1, 2).reshape(B, T, H * d)
        return self._lin(y, self.proj_out.weight)
