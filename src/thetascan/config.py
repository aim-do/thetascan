"""Small, public configuration surface for ThetaScan.

The implementation deliberately keeps the research-era switches behind a private
adapter.  Users choose an algorithm family first and then configure only options
which have a stable, documented meaning for that family.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any, Literal, Mapping


Family = Literal["gn", "kernel"]
GNNonlinearity = Literal["relu2", "relu2_threshold", "silu", "swiglu"]
GNReadNormalization = Literal[
    "none", "w2_feature_mass", "both_feature_mass"
]
KernelFeatureMap = Literal["softmax_partition", "relu2_ridge", "projected_bspline"]
KernelValueRepresentation = Literal["raw", "value_anchors", "value_mlp"]
KernelSparsity = Literal[
    "none", "relative_soft", "relative_st", "relative_st_blend"
]
KernelReadNormalization = Literal["none", "key_mass", "feature_mass"]
KernelSharpnessMode = Literal["fixed", "learned_per_head", "learned_per_feature"]
KernelReLU2ThresholdMode = Literal["none", "learned_per_head"]
KernelBSplineScaleMode = Literal["fixed", "learned_per_head"]
TemporalMode = Literal["sum", "ema", "bank"]
BankMode = Literal["fast", "stale"]
BlendMode = Literal["free", "tanh"]
Backend = Literal["auto", "naive", "quad", "chunk", "cumsum", "fla"]
RoPEMode = Literal["none", "partial", "full"]
RoPEPlacement = Literal["input", "feature"]


def _require_strict_int(name: str, value: object) -> None:
    """Reject bools and float-shaped integers in structural fields."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")


def _require_strict_bool(name: str, value: object) -> None:
    """Reject integer/string truthiness for public boolean switches."""
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")


def _require_finite_number(name: str, value: object) -> None:
    """Require a real, finite scalar before applying range constraints."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


@dataclass
class GNConfig:
    """Options specific to Gauss--Newton fast-weight memory.

    ``w2_feature_mass`` normalizes only the read of the fast W2 updates.  The
    ``both_feature_mass`` mode first retrieves a normalized hidden update from
    the positive slow-memory ReLU-squared features, applies the nonlinearity,
    and then performs the normalized W2 read.  The latter is scan-friendly but
    is no longer a literal read of ``sum(delta W1)``.
    """

    nonlinearity: GNNonlinearity = "relu2"
    jacobian_steps: int = 1
    read_normalization: GNReadNormalization = "none"

    def validate(self) -> None:
        _require_strict_int("GNConfig.jacobian_steps", self.jacobian_steps)
        if self.nonlinearity not in ("relu2", "relu2_threshold", "silu", "swiglu"):
            raise ValueError(f"unknown GN nonlinearity: {self.nonlinearity!r}")
        if self.jacobian_steps not in (1, 2):
            raise ValueError("GNConfig.jacobian_steps must be 1 or 2")
        if self.read_normalization not in (
            "none", "w2_feature_mass", "both_feature_mass"
        ):
            raise ValueError(
                f"unknown GN read normalization: {self.read_normalization!r}"
            )
        if (self.read_normalization != "none"
                and self.nonlinearity not in ("relu2", "relu2_threshold")):
            raise ValueError(
                "GN feature-mass normalization requires non-negative relu2 or "
                "relu2_threshold features"
            )
        if self.read_normalization != "none" and self.jacobian_steps != 1:
            raise ValueError(
                "GN feature-mass normalization requires jacobian_steps=1"
            )


@dataclass
class KernelConfig:
    """Options for normalized positive-feature kernel memory.

    ``feature_map`` selects the feature geometry. ``softmax_partition`` encodes keys and queries as
    ``softmax(kernel_sharpness * RMSNorm(W1 @ L2Norm(key)))``. Larger values
    produce more concentrated partition weights. The sharpness is not a literal
    radial/Gaussian bandwidth because the learned rows of W1 are not constrained to unit
    norm; ``8.0`` is the validated default. ``kernel_sharpness_mode='fixed'``
    keeps that scalar constant. ``'learned_per_head'`` treats it as the positive
    initialization of one independently learned sharpness per memory head.
    ``'learned_per_feature'`` learns a positive scale for every partition logit
    in every head. This is a diagonal logit calibration rather than a single
    temperature; the same scales are used for keys and queries.

    ``feature_parameters_trainable=False`` freezes W1 and every learned control
    that participates in the shared key/query feature map between optimizer
    steps. W1 is always fixed *within* a sequence in kernel-memory mode; this
    flag controls slow training, which is a distinct choice. It does not freeze
    a value codebook selected by ``value_anchors``.

    ``sparsity='relative_st'`` adds a learned per-head relative threshold on top
    of the softmax sharpness.  A feature remains active when its weight exceeds
    ``relative_threshold_init * max(weight)``.  The forward pass is exactly
    sparse while a sigmoid straight-through surrogate keeps gradients available.
    ``'relative_st_blend'`` convexly blends that exact-forward sparse address
    with the dense softmax using a learned bounded per-head scalar. Its default
    zero blend is exactly the historical dense address at initialization;
    ``sparse_blend_init`` selects another initial mixture in ``[0, 1]``.

    ``relu2_ridge`` applies a positive squared-ReLU ridge map and writes values
    with the inexpensive kernel numerator/mass rule. At zero threshold each
    pre-normalization feature has half-space support.
    ``relu2_threshold_mode='learned_per_head'`` subtracts one learned,
    zero-initialized scalar threshold after score RMS normalization, so a
    nonzero threshold need not retain a literal hyperplane boundary.
    The same threshold is used for key writes and query reads. This is distinct
    from softmax relative sparsity, which compares each weight with the largest
    weight in its address. ``projected_bspline`` maps each learned scalar
    projection through a fixed open-uniform B-spline partition. Each cell is
    local along one projection and global in all orthogonal directions.
    """

    feature_map: KernelFeatureMap = "softmax_partition"
    value_representation: KernelValueRepresentation = "raw"
    value_anchors: int = 8
    value_mlp_multiplier: float = 1.0
    kernel_sharpness: float = 8.0
    kernel_sharpness_mode: KernelSharpnessMode = "fixed"
    score_bias: bool = False
    relu2_threshold_mode: KernelReLU2ThresholdMode = "none"
    sparsity: KernelSparsity = "none"
    sparse_blend_init: float = 0.0
    relative_threshold_init: float = 0.01
    threshold_temperature: float = 0.25
    bspline_basis_count: int = 8
    bspline_degree: int = 3
    bspline_bound: float = 3.0
    bspline_scale: float = 1.0
    bspline_scale_mode: KernelBSplineScaleMode = "fixed"
    feature_parameters_trainable: bool = True
    read_normalization: KernelReadNormalization = "key_mass"

    def validate(self) -> None:
        for name, value in (
            ("KernelConfig.score_bias", self.score_bias),
            (
                "KernelConfig.feature_parameters_trainable",
                self.feature_parameters_trainable,
            ),
        ):
            _require_strict_bool(name, value)
        for name, value in (
            ("KernelConfig.value_anchors", self.value_anchors),
            ("KernelConfig.bspline_basis_count", self.bspline_basis_count),
            ("KernelConfig.bspline_degree", self.bspline_degree),
        ):
            _require_strict_int(name, value)
        for name, value in (
            ("KernelConfig.value_mlp_multiplier", self.value_mlp_multiplier),
            ("KernelConfig.kernel_sharpness", self.kernel_sharpness),
            ("KernelConfig.sparse_blend_init", self.sparse_blend_init),
            ("KernelConfig.relative_threshold_init", self.relative_threshold_init),
            ("KernelConfig.threshold_temperature", self.threshold_temperature),
            ("KernelConfig.bspline_bound", self.bspline_bound),
            ("KernelConfig.bspline_scale", self.bspline_scale),
        ):
            _require_finite_number(name, value)
        if self.feature_map not in (
            "softmax_partition", "relu2_ridge", "projected_bspline"
        ):
            raise ValueError(f"unknown kernel feature map: {self.feature_map!r}")
        if self.value_representation not in ("raw", "value_anchors", "value_mlp"):
            raise ValueError(f"unknown kernel value representation: {self.value_representation!r}")
        if self.read_normalization not in ("none", "key_mass", "feature_mass"):
            raise ValueError(f"unknown kernel read normalization: {self.read_normalization!r}")
        if self.value_representation == "value_anchors" and self.value_anchors < 2:
            raise ValueError("KernelConfig.value_anchors must be at least 2")
        if self.value_representation == "value_mlp" and self.value_mlp_multiplier <= 0:
            raise ValueError("KernelConfig.value_mlp_multiplier must be positive")
        if self.kernel_sharpness <= 0:
            raise ValueError("KernelConfig.kernel_sharpness must be positive")
        if self.kernel_sharpness_mode not in (
            "fixed", "learned_per_head", "learned_per_feature"
        ):
            raise ValueError(
                "KernelConfig.kernel_sharpness_mode must be 'fixed', "
                "'learned_per_head', or 'learned_per_feature'"
            )
        if self.sparsity not in (
            "none", "relative_soft", "relative_st", "relative_st_blend"
        ):
            raise ValueError(f"unknown kernel sparsity: {self.sparsity!r}")
        if not 0.0 <= self.sparse_blend_init <= 1.0:
            raise ValueError("KernelConfig.sparse_blend_init must be in [0, 1]")
        if self.sparsity != "relative_st_blend" and self.sparse_blend_init != 0.0:
            raise ValueError(
                "KernelConfig.sparse_blend_init is active only with "
                "sparsity='relative_st_blend'"
            )
        if self.relu2_threshold_mode not in ("none", "learned_per_head"):
            raise ValueError(
                "KernelConfig.relu2_threshold_mode must be 'none' or "
                "'learned_per_head'"
            )
        if self.relu2_threshold_mode != "none" \
                and self.feature_map != "relu2_ridge":
            raise ValueError(
                "learned relu2_threshold_mode requires "
                "feature_map='relu2_ridge'"
            )
        if self.relu2_threshold_mode != "none" and self.score_bias:
            raise ValueError(
                "relu2_threshold_mode and score_bias are redundant affine "
                "controls; enable only one"
            )
        if not 0.0 < self.relative_threshold_init < 1.0:
            raise ValueError("KernelConfig.relative_threshold_init must be in (0, 1)")
        if self.threshold_temperature <= 0.0:
            raise ValueError("KernelConfig.threshold_temperature must be positive")
        if self.sparsity != "none" and self.feature_map != "softmax_partition":
            raise ValueError(
                "learned relative sparsity is currently defined for "
                "feature_map='softmax_partition' only"
            )
        if self.kernel_sharpness_mode != "fixed" \
                and self.feature_map != "softmax_partition":
            raise ValueError(
                "learned kernel_sharpness_mode is defined for "
                "feature_map='softmax_partition' only"
            )
        if self.bspline_basis_count <= self.bspline_degree:
            raise ValueError("kernel B-spline basis count must exceed its degree")
        if self.bspline_degree != 3:
            raise ValueError("kernel projected_bspline currently supports degree=3 only")
        if self.bspline_bound <= 0.0 or self.bspline_scale <= 0.0:
            raise ValueError("kernel B-spline bound and scale must be positive")
        if self.bspline_scale_mode not in ("fixed", "learned_per_head"):
            raise ValueError(
                "KernelConfig.bspline_scale_mode must be 'fixed' or 'learned_per_head'"
            )
        if self.feature_map != "projected_bspline" \
                and self.bspline_scale_mode != "fixed":
            raise ValueError(
                "learned bspline_scale_mode requires "
                "feature_map='projected_bspline'"
            )


@dataclass
class RoPEConfig:
    """Fixed rotary positional features applied to memory keys and queries.

    ``partial`` rotates the first ``fraction`` of a feature dimension; ``full``
    rotates every paired dimension. ``placement="input"`` is the established
    path and rotates projected keys/queries before the nonlinear feature map.
    ``placement="feature"`` is an experimental signed-kernel ablation that rotates
    nonlinear features immediately before their scans. It can make an otherwise
    non-negative normalized kernel signed, so it is not a positive-kernel read.
    ``none`` leaves positional encoding to the surrounding model and is the default.
    """

    mode: RoPEMode = "none"
    fraction: float = 0.5
    base: float = 10_000.0
    placement: RoPEPlacement = "input"

    def validate(self) -> None:
        _require_finite_number("RoPEConfig.fraction", self.fraction)
        _require_finite_number("RoPEConfig.base", self.base)
        if self.mode not in ("none", "partial", "full"):
            raise ValueError(f"unknown RoPE mode: {self.mode!r}")
        if self.placement not in ("input", "feature"):
            raise ValueError(f"unknown RoPE placement: {self.placement!r}")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("RoPEConfig.fraction must be in (0, 1]")
        if self.base <= 0.0:
            raise ValueError("RoPEConfig.base must be positive")


@dataclass
class TemporalConfig:
    """How writes are combined over time.

    ``ema`` is a learned data-dependent write-side decay.  ``bank`` selects a
    temporal-mode bank: it keeps the sum state and
    adds one or two independently learned recency branches. Each
    zero-initialized branch contributes
    ``eta_j * (fast_j - slow)`` in ``bank_mode="fast"``.

    ``retention_inits`` specifies the EMA retention (alpha) directly;
    ``half_life_inits`` is the equivalent, more interpretable token
    half-life (``alpha = 2**(-1 / half_life)``).  They are mutually exclusive
    and must contain one value per branch.  With two branches and neither field
    set, the defaults are retentions ``(0.9, 0.99)`` (about 6.6 and 69 tokens).
    ``retention_init`` initializes the default one-branch configuration.
    ``blend_mode="free"`` preserves the unconstrained
    signed eta; ``"tanh"`` uses ``eta=tanh(raw_eta)`` for a signed range (-1, 1).
    """

    mode: TemporalMode = "sum"
    bank_mode: BankMode = "fast"
    retention_init: float = 0.9
    recency_branches: int = 1
    retention_inits: tuple[float, ...] | None = None
    half_life_inits: tuple[float, ...] | None = None
    blend_mode: BlendMode = "free"

    def resolved_retentions(self) -> tuple[float, ...]:
        """Return one EMA retention for each configured recency branch."""
        if self.retention_inits is not None:
            return tuple(float(value) for value in self.retention_inits)
        if self.half_life_inits is not None:
            return tuple(
                math.pow(2.0, -1.0 / float(value))
                for value in self.half_life_inits
            )
        if self.recency_branches == 2:
            return (float(self.retention_init), 0.99)
        return (float(self.retention_init),)

    def validate(self) -> None:
        _require_strict_int(
            "TemporalConfig.recency_branches", self.recency_branches
        )
        _require_finite_number(
            "TemporalConfig.retention_init", self.retention_init
        )
        if self.mode not in ("sum", "ema", "bank"):
            raise ValueError(f"unknown temporal mode: {self.mode!r}")
        if self.bank_mode not in ("fast", "stale"):
            raise ValueError(f"unknown bank mode: {self.bank_mode!r}")
        if not 0.0 < self.retention_init < 1.0:
            raise ValueError("TemporalConfig.retention_init must be in (0, 1)")
        if self.recency_branches not in (1, 2):
            raise ValueError("TemporalConfig.recency_branches must be 1 or 2")
        if self.blend_mode not in ("free", "tanh"):
            raise ValueError(
                "TemporalConfig.blend_mode must be 'free' or 'tanh'"
            )
        if (self.retention_inits is not None
                and self.half_life_inits is not None):
            raise ValueError(
                "set only one of retention_inits and half_life_inits"
            )
        if self.retention_inits is not None:
            for index, value in enumerate(self.retention_inits):
                _require_finite_number(
                    f"TemporalConfig.retention_inits[{index}]", value
                )
        if self.half_life_inits is not None:
            for index, value in enumerate(self.half_life_inits):
                _require_finite_number(
                    f"TemporalConfig.half_life_inits[{index}]", value
                )
            if any(value <= 0.0 for value in self.half_life_inits):
                raise ValueError("half-lives must be positive")
        retentions = self.resolved_retentions()
        if len(retentions) != self.recency_branches:
            raise ValueError(
                "retention/half-life initializers must contain exactly "
                "recency_branches values"
            )
        if any(not 0.0 < value < 1.0 for value in retentions):
            raise ValueError("retentions must be in (0, 1)")
        if self.recency_branches == 2 and retentions[0] == retentions[1]:
            raise ValueError(
                "two recency branches require distinct initial timescales"
            )


@dataclass
class RegularizationConfig:
    """Weights for optional cosine-orthogonality penalties."""

    feature_weight: float = 0.0
    head_weight: float = 0.0
    value_mlp_weight: float = 0.0

    def validate(self) -> None:
        named_values = (
            ("RegularizationConfig.feature_weight", self.feature_weight),
            ("RegularizationConfig.head_weight", self.head_weight),
            ("RegularizationConfig.value_mlp_weight", self.value_mlp_weight),
        )
        for name, value in named_values:
            _require_finite_number(name, value)
        values = tuple(value for _, value in named_values)
        if any(value < 0.0 for value in values):
            raise ValueError("regularization weights must be non-negative")


@dataclass
class RuntimeConfig:
    """Execution selection.  ``auto`` is the recommended setting: it picks
    ``chunk`` on every device, and FLA only for the ungated plain-sum scan on
    CUDA when the package is installed.

    ``chunk`` is the default implementation.  It forms the causal score one
    ``[scan_chunk, scan_chunk]`` tile at a time and carries an explicit
    ``[Dk, Dv]`` state across tiles, so its activation footprint grows linearly
    in sequence length instead of quadratically, and it performs about six times
    fewer multiply-accumulates than the quadratic form at the reference width.
    It is exact for both temporal modes and needs no compiled extension.

    The explicit portable backends compute the same scan and exist for
    debugging, profiling and parity: ``naive`` is the sequential reference loop;
    ``quad`` is the masked-matmul dual form, which materializes the full
    causal score matrix (O(T^2) memory) and trades that for one large matmul --
    it is the reference the published v0.1.0 measurements used, and ``chunk``
    with ``scan_chunk >= T`` reproduces it bit for bit; ``cumsum`` is the
    prefix-scan form.
    ``cumsum`` supports ``temporal.mode='sum'`` only: its decay handling
    divides by a clamped cumulative product, which silently loses precision
    once the accumulated decay of a long sequence exceeds the guard range.
    ``auto`` never selects it.

    ``fla`` requires the optional ``flash-linear-attention`` dependency, CUDA
    tensors, and a Triton build whose gated backward is trustworthy on the
    target architecture.  Validate a forward and a backward against ``naive``
    on the target stack before selecting it explicitly; ``auto`` deliberately
    keeps retained views away from that kernel.

    ``scan_chunk`` tunes the ``chunk`` tile.  Retained bytes are
    ``T * scan_chunk`` for the score tiles and ``(T / scan_chunk) * Dk * Dv``
    for the carried states, so the footprint minimum is near
    ``sqrt(Dk * Dv)``; the default suits the reference memory width.  Other
    backends ignore it.
    """

    backend: Backend = "auto"
    scan_chunk: int = 512

    def validate(self) -> None:
        if self.backend not in ("auto", "naive", "quad", "chunk", "cumsum", "fla"):
            raise ValueError(
                "RuntimeConfig.backend must be one of 'auto', 'naive', 'quad', "
                "'chunk', 'cumsum', 'fla'"
            )
        if not isinstance(self.scan_chunk, bool) and isinstance(self.scan_chunk, int):
            if self.scan_chunk < 1:
                raise ValueError("RuntimeConfig.scan_chunk must be >= 1")
        else:
            raise TypeError("RuntimeConfig.scan_chunk must be an int")


@dataclass
class ThetaScanConfig:
    """Public ThetaScan configuration.

    Select ``family`` first. ``gn`` exposes a nonlinear Gauss--Newton memory
    with one or two Jacobian steps. ``kernel`` exposes normalized kernel
    regression with softmax-partition, ReLU-squared ridge, or projected B-spline
    features.

    ``share_key_query`` uses one key and one query projection for all heads,
    plus learned per-head biases that start at zero; values and the memory
    weights stay per-head. ``key_value_heads`` instead selects a Transformer-GQA
    projection layout: queries remain per-head while each key/value group is
    repeated over an equal-sized contiguous head group. The two options are
    mutually exclusive. ``output_gate`` multiplies the mixer output by a learned
    input-dependent gate. The validated benchmark presets select their own
    projection layout.
    """

    d_model: int = 512
    n_heads: int = 8
    head_dim: int | None = None
    memory_multiplier: int = 2
    feature_expansion: int = 1
    expansion_key: str = "thetascan"
    depth: int = 1
    share_key_query: bool = False
    key_value_heads: int | None = None
    output_gate: bool = True
    family: Family = "gn"
    gn: GNConfig = field(default_factory=GNConfig)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    rope: RoPEConfig = field(default_factory=RoPEConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        for name, value in (
            ("ThetaScanConfig.share_key_query", self.share_key_query),
            ("ThetaScanConfig.output_gate", self.output_gate),
        ):
            _require_strict_bool(name, value)
        for name, value in (
            ("ThetaScanConfig.d_model", self.d_model),
            ("ThetaScanConfig.n_heads", self.n_heads),
            ("ThetaScanConfig.memory_multiplier", self.memory_multiplier),
            ("ThetaScanConfig.feature_expansion", self.feature_expansion),
            ("ThetaScanConfig.depth", self.depth),
        ):
            _require_strict_int(name, value)
        if self.head_dim is not None:
            _require_strict_int("ThetaScanConfig.head_dim", self.head_dim)
        if self.family not in ("gn", "kernel"):
            raise ValueError("ThetaScanConfig.family must be 'gn' or 'kernel'")
        if self.d_model <= 0 or self.n_heads <= 0 or self.memory_multiplier <= 0:
            raise ValueError("d_model, n_heads, and memory_multiplier must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.head_dim is not None and self.head_dim * self.n_heads != self.d_model:
            raise ValueError("head_dim * n_heads must equal d_model")
        if self.key_value_heads is not None:
            if type(self.key_value_heads) is not int:
                raise TypeError(
                    "ThetaScanConfig.key_value_heads must be an integer or None"
                )
            if not 1 <= self.key_value_heads <= self.n_heads:
                raise ValueError("key_value_heads must be in [1, n_heads]")
            if self.n_heads % self.key_value_heads:
                raise ValueError("n_heads must be divisible by key_value_heads")
            if self.share_key_query:
                raise ValueError(
                    "key_value_heads and share_key_query=True are mutually exclusive"
                )
        if self.depth < 1:
            raise ValueError("depth must be at least 1")
        if self.feature_expansion < 1:
            raise ValueError("ThetaScanConfig.feature_expansion must be >= 1")
        if self.feature_expansion > 1:
            if not isinstance(self.expansion_key, str) or not self.expansion_key:
                raise ValueError(
                    "ThetaScanConfig.expansion_key must be a non-empty string"
                )
            memory_features = self.memory_multiplier * (
                self.head_dim or self.d_model // self.n_heads
            )
            if memory_features % self.feature_expansion:
                raise ValueError(
                    "feature_expansion must divide the effective memory width "
                    "(memory_multiplier * head_dim)"
                )
            if self.family == "gn" and self.gn.nonlinearity not in (
                "relu2", "relu2_threshold"
            ):
                raise ValueError(
                    "GN feature_expansion requires nonlinearity 'relu2' or "
                    "'relu2_threshold'"
                )
            if (self.family == "kernel"
                    and self.kernel.feature_map != "relu2_ridge"):
                raise ValueError(
                    "kernel feature_expansion requires feature_map='relu2_ridge'"
                )
        self.temporal.validate()
        self.rope.validate()
        self.regularization.validate()
        self.runtime.validate()
        if self.runtime.backend == "cumsum" and self.temporal.mode != "sum":
            raise ValueError(
                "backend='cumsum' supports temporal.mode='sum' only: its "
                "decayed form divides by a clamped cumulative product and "
                "silently loses precision on long sequences; use 'auto', "
                "'naive', 'quad', or 'fla' with 'ema' or 'bank'"
            )
        if self.family == "gn":
            self.gn.validate()
            if (self.rope.mode != "none"
                    and self.rope.placement == "feature"
                    and self.gn.read_normalization != "none"):
                raise ValueError(
                    "feature-space RoPE makes the GN feature-mass denominator "
                    "signed; use placement='input', mode='none', or "
                    "read_normalization='none'"
                )
        else:
            kernel = self.kernel
            kernel.validate()
            if (self.rope.mode != "none"
                    and self.rope.placement == "feature"
                    and kernel.read_normalization in ("key_mass", "feature_mass")):
                raise ValueError(
                    "feature-space RoPE makes the kernel normalization "
                    "denominator signed; use placement='input', mode='none', "
                    "or read_normalization='none'"
                )
            if self.depth != 1:
                raise ValueError("kernel memory currently supports depth=1 only")
            if kernel.feature_map == "projected_bspline":
                memory_features = self.memory_multiplier * (
                    self.head_dim or self.d_model // self.n_heads
                )
                if memory_features % kernel.bspline_basis_count:
                    raise ValueError(
                        "projected_bspline requires memory_multiplier * head_dim "
                        "to be divisible by bspline_basis_count"
                    )
                if kernel.score_bias:
                    raise ValueError(
                        "score_bias is an affine-score axis and is not used by "
                        "projected_bspline"
                    )
                if self.rope.placement == "feature" and self.rope.mode != "none":
                    raise ValueError(
                        "projected_bspline requires input-space RoPE; rotating the "
                        "non-negative spline features breaks their partition"
                    )
            if (kernel.value_representation == "value_anchors"
                    and kernel.read_normalization != "key_mass"):
                raise ValueError(
                    "kernel value_anchors requires read_normalization='key_mass'"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-friendly configuration manifest.

        Tuple-valued retention initializers are preserved as tuples so a
        ``from_dict(to_dict(config))`` round trip is lossless in Python. JSON
        encoders may serialize them as arrays; :meth:`from_dict` accepts both.
        """
        return deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ThetaScanConfig":
        """Build and validate a config from a nested manifest.

        A manifest may begin with ``{"preset": NAME, ...}``; remaining fields
        recursively override that preset. Only public, versioned preset names
        are accepted, avoiding arbitrary attribute lookup. Manifests must use
        the canonical field spellings emitted by :meth:`to_dict`; discarded
        pre-release spellings are not accepted.
        """
        if not isinstance(data, Mapping):
            raise TypeError("ThetaScanConfig.from_dict expects a mapping")
        payload: dict[str, Any] = deepcopy(dict(data))

        def merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
            for key, value in override.items():
                if isinstance(value, Mapping) and isinstance(base.get(key), dict):
                    base[key] = merge(dict(base[key]), value)
                else:
                    base[key] = deepcopy(value)
            return base

        preset = payload.pop("preset", None)
        if preset is not None:
            presets = {
                "gn_reference_v0_1": cls.gn_reference_v0_1,
                "gn_expanded_reference_v0_1": cls.gn_expanded_reference_v0_1,
                "kernel_expanded_reference_v0_1": cls.kernel_expanded_reference_v0_1,
            }
            if preset not in presets:
                raise ValueError(f"unknown ThetaScan preset: {preset!r}")
            d_model = payload.pop("d_model", 512)
            n_heads = payload.pop("n_heads", 8)
            payload = merge(
                presets[preset](d_model=d_model, n_heads=n_heads).to_dict(),
                payload,
            )

        nested = {
            "gn": GNConfig,
            "kernel": KernelConfig,
            "rope": RoPEConfig,
            "temporal": TemporalConfig,
            "regularization": RegularizationConfig,
            "runtime": RuntimeConfig,
        }
        for name, constructor in nested.items():
            value = payload.get(name)
            if isinstance(value, Mapping):
                item = dict(value)
                if name == "temporal":
                    for tuple_name in (
                        "retention_inits", "half_life_inits"
                    ):
                        if item.get(tuple_name) is not None:
                            item[tuple_name] = tuple(item[tuple_name])
                payload[name] = constructor(**item)
        config = cls(**payload)
        config.validate()
        return config

    def _to_core_config(self):
        """Translate the stable public hierarchy to the private implementation."""
        self.validate()
        from ._core.config import ThetaScanConfig as CoreConfig

        common: dict[str, object] = {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
            "mem_mult": self.memory_multiplier,
            "feature_expansion": self.feature_expansion,
            "expansion_key": self.expansion_key,
            "depth": self.depth,
            "backend": self.runtime.backend,
            "scan_chunk": self.runtime.scan_chunk,
            "rope": self.rope.mode,
            "rope_frac": self.rope.fraction,
            "rope_base": self.rope.base,
            "rope_placement": self.rope.placement,
            "qk_norm": True,
            "act_norm": True,
            "out_gate": self.output_gate,
            "share_kq": self.share_key_query,
            "key_value_heads": self.key_value_heads,
            "ortho_intra": self.regularization.feature_weight,
            "ortho_inter": self.regularization.head_weight,
            "value_mlp_ortho": self.regularization.value_mlp_weight,
        }

        if self.temporal.mode == "ema":
            common.update(accumulation="ema_gate", decay_gate="scalar")
        else:
            common.update(accumulation="sum", decay_gate="off")
        if self.temporal.mode == "bank":
            common.update(
                read_fade=True,
                read_fade_mode=self.temporal.bank_mode,
                fade_alpha_init=self.temporal.retention_init,
                fade_branches=self.temporal.recency_branches,
                fade_alpha_inits=self.temporal.resolved_retentions(),
                fade_blend_mode=self.temporal.blend_mode,
            )

        if self.family == "gn":
            nonlin = "relu2" if self.gn.nonlinearity == "relu2_threshold" else self.gn.nonlinearity
            common.update(
                write_rule="gn",
                nonlin=nonlin,
                learn_thresh=self.gn.nonlinearity == "relu2_threshold",
                write_iters=self.gn.jacobian_steps,
                fast_w1=True,
                read_norm=self.gn.read_normalization != "none",
                read_norm_w1=self.gn.read_normalization == "both_feature_mass",
            )
        else:
            kernel = self.kernel
            value_centers = kernel.value_anchors \
                if kernel.value_representation == "value_anchors" else 0
            value_mlp = kernel.value_mlp_multiplier \
                if kernel.value_representation == "value_mlp" else 0.0
            common.update(
                write_rule="kernel",
                nonlin=(
                    "softmax_hidden"
                    if kernel.feature_map == "softmax_partition"
                    else "relu2"
                ),
                kernel_kind=kernel.feature_map,
                kernel_score_bias=kernel.score_bias,
                kernel_relu2_threshold_mode=kernel.relu2_threshold_mode,
                kernel_sparsity=kernel.sparsity,
                kernel_sparse_blend_init=kernel.sparse_blend_init,
                kernel_relative_threshold_init=kernel.relative_threshold_init,
                kernel_threshold_temperature=kernel.threshold_temperature,
                bspline_basis_count=kernel.bspline_basis_count,
                bspline_degree=kernel.bspline_degree,
                bspline_bound=kernel.bspline_bound,
                bspline_scale=kernel.bspline_scale,
                bspline_scale_mode=kernel.bspline_scale_mode,
                softmax_gain=kernel.kernel_sharpness,
                softmax_gain_mode=kernel.kernel_sharpness_mode,
                fast_w1=False,
                write_iters=1,
                read_norm=kernel.read_normalization != "none",
                read_norm_mode=(
                    kernel.read_normalization
                    if kernel.read_normalization != "none"
                    else "key_mass"
                ),
                value_centers=value_centers,
                value_mlp_mult=value_mlp,
            )

        return CoreConfig(**common)

    # ------------------------------------------------------------------
    # Presets return normal, fully public config objects. Their benchmark
    # defaults were validated at head_dim = 64 (n_heads = d_model / 64);
    # custom dimensions remain valid.
    # ------------------------------------------------------------------

    @classmethod
    def gn_reference_v0_1(
        cls, d_model: int = 512, n_heads: int = 8
    ) -> "ThetaScanConfig":
        """Versioned one-recency reference for two-stage normalized GN."""
        return cls(
            d_model=d_model,
            n_heads=n_heads,
            memory_multiplier=3,
            share_key_query=True,
            output_gate=False,
            family="gn",
            gn=GNConfig(
                nonlinearity="relu2",
                jacobian_steps=1,
                read_normalization="both_feature_mass",
            ),
            rope=RoPEConfig(mode="partial", fraction=0.5, base=10_000.0),
            temporal=TemporalConfig(
                mode="bank",
                bank_mode="fast",
                retention_init=0.9,
                recency_branches=1,
                retention_inits=(0.9,),
                blend_mode="free",
            ),
        )

    @classmethod
    def gn_expanded_reference_v0_1(
        cls, d_model: int = 512, n_heads: int = 8
    ) -> "ThetaScanConfig":
        """Versioned expanded GN reference: 2x random feature expansion.

        Doubles the effective memory width without adding trainable memory
        parameters; the trainable core stays at the dense reference width.
        """
        return cls(
            d_model=d_model,
            n_heads=n_heads,
            memory_multiplier=6,
            feature_expansion=2,
            share_key_query=True,
            output_gate=False,
            family="gn",
            gn=GNConfig(
                nonlinearity="relu2_threshold",
                jacobian_steps=1,
                read_normalization="both_feature_mass",
            ),
            rope=RoPEConfig(mode="partial", fraction=0.5, base=10_000.0),
            temporal=TemporalConfig(
                mode="bank",
                bank_mode="fast",
                recency_branches=1,
                retention_inits=(0.9,),
                blend_mode="free",
            ),
        )

    @classmethod
    def kernel_expanded_reference_v0_1(
        cls, d_model: int = 512, n_heads: int = 8
    ) -> "ThetaScanConfig":
        """Versioned expanded kernel reference: ReLU-squared ridge, 2x expansion."""
        return cls(
            d_model=d_model,
            n_heads=n_heads,
            memory_multiplier=6,
            feature_expansion=2,
            share_key_query=True,
            output_gate=False,
            family="kernel",
            kernel=KernelConfig(
                feature_map="relu2_ridge",
                relu2_threshold_mode="learned_per_head",
                read_normalization="key_mass",
            ),
            rope=RoPEConfig(mode="partial", fraction=0.5, base=10_000.0),
            temporal=TemporalConfig(
                mode="bank",
                bank_mode="fast",
                recency_branches=2,
                half_life_inits=(8.0, 64.0),
                blend_mode="free",
            ),
        )
