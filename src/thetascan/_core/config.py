from __future__ import annotations
from dataclasses import dataclass

NONLINS = ("relu2", "silu", "swiglu", "softmax_hidden")
FADE_MODES = ("stale", "fast")
FADE_BLEND_MODES = ("free", "tanh")
ACCUMS = ("sum", "ema_gate")
WRITE_RULES = ("gn", "kernel")
ROPES = ("none", "full", "partial")
ROPE_PLACEMENTS = ("input", "feature")
DECAYS = ("off", "scalar")
BACKENDS = ("auto", "naive", "quad", "cumsum", "fla")
PARAM_DTYPES = ("fp32", "fp16", "bf16", "fp8_e4m3")
SOFTMAX_GAIN_MODES = ("fixed", "learned_per_head", "learned_per_feature")
KERNEL_KINDS = ("softmax_partition", "relu2_ridge", "projected_bspline")
KERNEL_SPARSITY_MODES = (
    "none", "relative_soft", "relative_st", "relative_st_blend"
)
KERNEL_RELU2_THRESHOLD_MODES = ("none", "learned_per_head")
READ_NORM_MODES = ("key_mass", "feature_mass")
BSPLINE_SCALE_MODES = ("fixed", "learned_per_head")


@dataclass
class ThetaScanConfig:
    d_model: int = 512
    n_heads: int = 8
    head_dim: int | None = None          # default d_model // n_heads
    mem_mult: int = 2                    # memory hidden m = mem_mult * head_dim
    depth: int = 1                       # ResNet blocks inside the memory net
    nonlin: str = "relu2"
    fast_w1: bool = True                 # False: W1/Wg slow-only (no delta/LA1) -> the
                                         # write is linear in the fast params: scalar
                                         # Gram, 1 LA per block, sigma' never used
    write_iters: int = 1                 # 1 | 2 Gauss-Newton iterations at theta0
    write_rule: str = "gn"               # gn: residual + Jacobi-preconditioned
                                         # lambda (Gauss-Newton at theta0). kernel:
                                         # lam_hat = raw v — S += a·v^T, the pure
                                         # kernel-regression deposit (no solve; pair
                                         # with read_norm for the normalized kernel
                                         # read). Requires fast_w1=False, depth=1.
    kernel_kind: str = "softmax_partition"  # Positive feature geometry.
    kernel_score_bias: bool = False       # Free per-head affine score bias [H,M].
    kernel_relu2_threshold_mode: str = "none"  # learned_per_head: tau [H],
                                                # phi=norm(ReLU(pre-tau)^2).
    kernel_sparsity: str = "none"        # Learned relative-to-max thresholding.
    kernel_sparse_blend_init: float = 0.0  # exact-forward dense/sparse mix alpha.
    kernel_relative_threshold_init: float = 0.01
    kernel_threshold_temperature: float = 0.25
    bspline_basis_count: int = 8          # Basis functions per learned projection.
    bspline_degree: int = 3               # v1: cubic only.
    bspline_bound: float = 3.0            # Standardized coordinate grid [-b,b].
    bspline_scale: float = 1.0            # Projection-coordinate scale.
    bspline_scale_mode: str = "fixed"    # fixed | learned_per_head.
    value_centers: int = 0               # >0: joint memory: values are also softmax-partition encoded —
                                         # r_t = softmax(gain·C_v·v̂_t) over M_v value
                                         # centers, S ∈ R^{M_k×M_v} accumulates the
                                         # joint key-value histogram, the read returns
                                         # a value-center mixture DECODED via C_vᵀ.
                                         # C_v [1, M_v, d] (head-shared codebook)
                                         # rides the Wg slot. v1: linear decode
                                         # (= learned codebook bottleneck on values);
                                         # mode-seeking sharpening = v1.5. Requires
                                         # write_rule='kernel' + read_norm.
    value_mlp_mult: float = 0.0          # >0: nonlinear WRITE — the deposited value
                                         # passes a head-shared residual MLP first:
                                         # v̄ = v + W_b·silu(W_a·v), hidden =
                                         # mult·head_dim, W_b zero-init (exact base
                                         # scheme at step 0). Shapes values so that
                                         # collision AVERAGING at read is benign
                                         # (MLP∘mixture ≠ mixture∘MLP — real
                                         # expressivity); scan-exact (per-token,
                                         # upstream of the engine). A/B partner of
                                         # value_centers: free code vs simplex code.
    read_fade: bool = False              # dual-timescale read-side forgetting: the SAME write streams
                                         # are scanned twice — S_slow = Σ Δ_i (plain
                                         # sum) and S_fast = EMA(Δ) (the discounted
                                         # sum with a per-head LEARNABLE alpha); the
                                         # read de-emphasizes the STALE mass:
                                         #   y = Read(S_slow)
                                         #     − η·(Read(S_slow − S_fast) − f_θ0(q))
                                         # Stale per-token weight 1 − alpha^(t−i):
                                         # zero on the current token, →1 with age.
                                         # η per-head, ZERO-init (identity at step
                                         # 0; η<0 = emphasize old — the knob is a
                                         # signed recency contrast). The f_θ0
                                         # subtraction confines the fade to memory
                                         # CONTENT (θ0's base function is never
                                         # scaled). Read(S_stale) is a true second
                                         # read pass (the state enters nonlinearly
                                         # via LA1 / the read_norm denominator, so
                                         # no output-blend shortcut); under
                                         # read_norm it is the normalized read over the
                                         # stale distribution (stale mass denom) —
                                         # this sidesteps read_norm v1's no-decay
                                         # cumsum limit without touching S_slow.
                                         # v1: accumulation='sum' only.
    read_fade_mode: str = "stale"        # stale: the formula above (suppress the
                                         # OLD tail). fast: blend toward the
                                         # RECENCY read —
                                         #   y = Read(S_slow)
                                         #     + η·(Read(S_fast) − Read(S_slow))
                                         # For the LINEAR read (W2-only, no
                                         # read_norm) the two are algebraically
                                         # identical (blend weights sum to 1, so
                                         # f_θ0 never double-counts); they differ
                                         # exactly where Read is nonlinear in the
                                         # state: under read_norm 'stale' is the
                                         # normalized read over the stale distribution,
                                         # 'fast' the normalized read over the recency-
                                         # weighted one (mass = ema_cumsum) —
                                         # temporal bandwidth asymmetry in time.
    fade_alpha_init: float = 0.9         # default one-branch retention initializer
    fade_branches: int = 1               # independently learned fast EMA views
    fade_alpha_inits: tuple[float, ...] | None = None  # explicit retention per branch
    fade_blend_mode: str = "free"        # free: eta=raw; tanh: eta=tanh(raw)
    learn_thresh: bool = False           # sigma(pre − τ) with a LEARNABLE per-unit
                                         # threshold τ [Hw1, m, 1], zero-init (=
                                         # exact base nonlin at step 0; the model
                                         # learns its own sparsity level). τ rides
                                         # the Wg slot of the theta0 tuple (free
                                         # unless swiglu — validated exclusive);
                                         # 3D shape -> Adam routing.
    feature_expansion: int = 1           # f>=2: trainable W1/W2 shrink to the base
                                         # width mem_hidden/f and fixed key-derived
                                         # sign maps expand them back to mem_hidden.
                                         # State and learned per-feature controls
                                         # stay at the effective width; trainable
                                         # slow parameters do not grow with it.
    expansion_key: str = "thetascan"     # deterministic namespace for the fixed
                                         # expansion maps (SHAKE256-derived,
                                         # non-persistent buffers)
    accumulation: str = "sum"            # sum | ema_gate
    read_norm: bool = False              # normalize the LA2 memory read by the accumulated
                                         # key mass: y_mem = Σw_i λ̂_i / (Σw_i + eps),
                                         # w_i = <h_q, h_i> — the linear-attention
                                         # denominator (softmax's convex-combination
                                         # semantics). EMA filters numerator and
                                         # denominator with the same weights. Requires
                                         # accumulation='sum' or 'ema_gate', write_iters=1.
    read_norm_mode: str = "key_mass"     # key_mass: global normalized ratio;
                                         # feature_mass: query-weighted slot means.
    read_norm_w1: bool = False           # experimental two-stage GN read. First form the
                                         # positive slow-reference feature p_q=sigma(W1 q),
                                         # retrieve g_i through the normalized p_q.p_i
                                         # kernel, add that correction to the hidden
                                         # preactivation, apply sigma again, then perform
                                         # the normalized LA2 read. This preserves the GN
                                         # write payloads but is NOT a literal read of
                                         # sum(delta W1). Requires read_norm, GN, fast W1,
                                         # relu2, one write iteration and scalar/static EMA.
    rope: str = "partial"
    rope_frac: float = 0.5
    rope_base: float = 10000.0
    rope_placement: str = "input"       # input: rotate q/k before theta0; feature:
                                         # rotate sigma(theta0 q/k) before the scans
    qk_norm: bool = True
    act_norm: bool = True                # RMSNorm on memory pre-activations
    decay_gate: str = "off"              # off | scalar (data-dependent per-head
                                         # log-decay stream)
    out_gate: bool = True
    share_kq: bool = False               # ONE shared k and one shared q projection for
                                         # ALL heads (Mamba-3 ngroups=1 mirror) plus
                                         # per-head static biases on the shared k,q
                                         # (the B_bias/C_bias analog); v and theta0
                                         # stay per-head. Zero-init biases -> heads
                                         # are symmetric at init.
    key_value_heads: int | None = None   # Transformer-GQA layout: H query heads and G
                                         # grouped key/value heads, pairwise expanded by
                                         # repeat_interleave(H // G). Mutually exclusive
                                         # with share_kq; None keeps full per-head QKV.
    param_dtype: str = "fp32"
    compute_dtype: str = "fp32"          # v1: fp32 only
    backend: str = "auto"
    softmax_gain: float = 1.0            # nonlin='softmax_hidden': h = softmax(gain·pre).
                                         # gain>1 sharpens the init distribution (near-uniform
                                         # h has ||h||²≈1/m, which can blow up the write lambda); the Jacobian
                                         # scales by the same gain (handled in engine).
                                         # nonlin='exp': reused as the ANCHOR TEMPERATURE —
                                         # scales the W1 init (block.theta0), so pre std at
                                         # init ≈ gain/sqrt(head_dim); the kernel exp(g·q̃·k)
                                         # sharpness then evolves freely with the W1 norm.
    softmax_gain_mode: str = "fixed"     # fixed: use softmax_gain as a constant;
                                         # learned_per_head: one positive gain/head;
                                         # learned_per_feature: one positive gain for
                                         # every [head, hidden-feature] logit. Learned
                                         # modes are kernel/softmax_hidden only.
    ortho_intra: float = 0.0             # weight: row-decorrelation inside each theta0 matrix
    ortho_inter: float = 0.0             # weight: cross-head theta0 decorrelation
    value_mlp_ortho: float = 0.0         # row-decorrelation for optional value MLP
    eps: float = 1e-6
    chunk_size: int = 256                # T-chunking for Gram-product memory bound

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.d_model // self.n_heads
        self.validate()

    @property
    def mem_hidden(self) -> int:
        return self.mem_mult * self.head_dim

    @property
    def resolved_fade_alphas(self) -> tuple[float, ...]:
        """One retention initializer per fast EMA branch."""
        if self.fade_alpha_inits is not None:
            return tuple(float(value) for value in self.fade_alpha_inits)
        if self.fade_branches == 2:
            return (float(self.fade_alpha_init), 0.99)
        return (float(self.fade_alpha_init),)

    def validate(self):
        def chk(v, allowed, name):
            if v not in allowed:
                raise ValueError(f"{name}={v!r} not in {allowed}")
        if self.nonlin not in NONLINS:
            raise ValueError(f"nonlin={self.nonlin!r} not in {NONLINS}")
        if self.learn_thresh and self.nonlin == "swiglu":
            raise ValueError("learn_thresh rides the Wg slot of the theta0 tuple — "
                             "swiglu occupies it; mutually exclusive")
        chk(self.accumulation, ACCUMS, "accumulation")
        chk(self.rope, ROPES, "rope")
        chk(self.rope_placement, ROPE_PLACEMENTS, "rope_placement")
        chk(self.decay_gate, DECAYS, "decay_gate")
        chk(self.backend, BACKENDS, "backend")
        chk(self.param_dtype, PARAM_DTYPES, "param_dtype")
        chk(self.softmax_gain_mode, SOFTMAX_GAIN_MODES, "softmax_gain_mode")
        chk(self.kernel_kind, KERNEL_KINDS, "kernel_kind")
        chk(self.kernel_sparsity, KERNEL_SPARSITY_MODES, "kernel_sparsity")
        chk(
            self.kernel_relu2_threshold_mode,
            KERNEL_RELU2_THRESHOLD_MODES,
            "kernel_relu2_threshold_mode",
        )
        chk(self.read_norm_mode, READ_NORM_MODES, "read_norm_mode")
        chk(self.bspline_scale_mode, BSPLINE_SCALE_MODES, "bspline_scale_mode")
        if self.compute_dtype != "fp32":
            raise ValueError("v1 supports compute_dtype='fp32' only")
        if self.write_iters not in (1, 2):
            raise ValueError("write_iters must be 1 or 2")
        chk(self.write_rule, WRITE_RULES, "write_rule")
        if self.write_rule == "kernel":
            if self.fast_w1:
                raise ValueError("write_rule='kernel' has no dW1 (the write is a pure "
                                 "LA2 deposit): set fast_w1=False")
            if self.write_iters != 1:
                raise ValueError("write_rule='kernel' requires write_iters=1")
            if self.kernel_kind == "projected_bspline":
                if self.mem_hidden % self.bspline_basis_count:
                    raise ValueError(
                        "projected_bspline requires mem_hidden divisible by "
                        "bspline_basis_count"
                    )
                if self.kernel_score_bias:
                    raise ValueError(
                        "kernel_score_bias is not defined for projected_bspline"
                    )
                if self.rope_placement == "feature" and self.rope != "none":
                    raise ValueError(
                        "projected_bspline excludes feature-space RoPE"
                    )
            if self.depth != 1:
                raise ValueError("write_rule='kernel' requires depth=1")
            if self.accumulation not in ("sum", "ema_gate"):
                raise ValueError("write_rule='kernel' supports accumulation "
                                 "sum/ema_gate")
        if self.value_mlp_mult < 0:
            raise ValueError("value_mlp_mult must be >= 0 (0 = disabled)")
        if self.value_centers:
            if self.value_centers < 2:
                raise ValueError("value_centers must be >= 2")
            if self.write_rule != "kernel" or not self.read_norm:
                raise ValueError("value_centers requires write_rule='kernel' and "
                                 "read_norm (the decode assumes a normalized mixture "
                                 "over value centers)")
            if self.learn_thresh or self.nonlin == "swiglu":
                raise ValueError("value_centers rides the Wg slot — exclusive with "
                                 "learn_thresh and swiglu")
        if self.read_fade:
            chk(self.read_fade_mode, FADE_MODES, "read_fade_mode")
            if self.accumulation != "sum":
                raise ValueError("read_fade requires accumulation='sum' (S_slow is "
                                 "the plain sum; the fast EMA is the feature's own "
                                 "second scan — ema_gate would forget twice)")
            if not (0.0 < self.fade_alpha_init < 1.0):
                raise ValueError("fade_alpha_init must be in (0, 1)")
            if self.fade_branches not in (1, 2):
                raise ValueError("fade_branches must be 1 or 2")
            chk(self.fade_blend_mode, FADE_BLEND_MODES, "fade_blend_mode")
            alphas = self.resolved_fade_alphas
            if len(alphas) != self.fade_branches:
                raise ValueError(
                    "fade_alpha_inits must contain exactly fade_branches values"
                )
            if any(not 0.0 < alpha < 1.0 for alpha in alphas):
                raise ValueError("all fade alpha initializers must be in (0, 1)")
            if self.fade_branches == 2 and alphas[0] == alphas[1]:
                raise ValueError("two fade branches require distinct initial timescales")
        if type(self.feature_expansion) is not int or self.feature_expansion < 1:
            raise ValueError("feature_expansion must be an integer >= 1")
        if self.feature_expansion > 1:
            if self.mem_hidden % self.feature_expansion:
                raise ValueError(
                    "feature_expansion must divide the effective memory width "
                    "(mem_mult * head_dim)"
                )
            if self.nonlin != "relu2":
                raise ValueError(
                    "feature_expansion requires nonlin='relu2' "
                    "(learn_thresh is supported)"
                )
            if self.write_rule == "kernel" and self.kernel_kind != "relu2_ridge":
                raise ValueError(
                    "kernel feature_expansion requires kernel_kind='relu2_ridge'"
                )
            if not self.expansion_key:
                raise ValueError("expansion_key must be a non-empty string")
        if self.depth < 1:
            raise ValueError("depth >= 1")
        if self.d_model % self.n_heads:
            raise ValueError("d_model % n_heads != 0")
        if self.key_value_heads is not None:
            if type(self.key_value_heads) is not int:
                raise TypeError("key_value_heads must be an integer or None")
            if not 1 <= self.key_value_heads <= self.n_heads:
                raise ValueError("key_value_heads must be in [1, n_heads]")
            if self.n_heads % self.key_value_heads:
                raise ValueError("n_heads must be divisible by key_value_heads")
            if self.share_kq:
                raise ValueError(
                    "key_value_heads and share_kq=True are mutually exclusive"
                )
        if self.accumulation == "ema_gate" and self.decay_gate == "off":
            raise ValueError("accumulation='ema_gate' requires decay_gate != 'off'")
        if self.decay_gate != "off" and self.accumulation != "ema_gate":
            raise ValueError("decay_gate is only used by accumulation='ema_gate'")
        if not self.fast_w1 and self.write_iters != 1:
            raise ValueError("fast_w1=False makes the write linear in the fast "
                             "params -> exact in one step; write_iters must be 1")
        if not (0.0 < self.rope_frac <= 1.0):
            raise ValueError("rope_frac in (0,1]")
        if self.ortho_intra < 0 or self.ortho_inter < 0 or self.value_mlp_ortho < 0:
            raise ValueError("ortho_* weights must be >= 0")
        if self.softmax_gain <= 0:
            raise ValueError("softmax_gain must be > 0")
        if self.softmax_gain_mode != "fixed" and (
            self.write_rule != "kernel" or self.nonlin != "softmax_hidden"
        ):
            raise ValueError(
                "learned softmax_gain_mode requires "
                "write_rule='kernel' and nonlin='softmax_hidden'"
            )
        if self.kernel_sparsity != "none" and (
            self.write_rule != "kernel"
            or self.kernel_kind != "softmax_partition"
        ):
            raise ValueError(
                "kernel relative sparsity requires kind='softmax_partition'"
            )
        if not 0.0 <= self.kernel_sparse_blend_init <= 1.0:
            raise ValueError("kernel_sparse_blend_init must be in [0, 1]")
        if (self.kernel_sparsity != "relative_st_blend"
                and self.kernel_sparse_blend_init != 0.0):
            raise ValueError(
                "kernel_sparse_blend_init is active only for "
                "kernel_sparsity='relative_st_blend'"
            )
        if self.kernel_relu2_threshold_mode != "none" and (
            self.write_rule != "kernel"
            or self.kernel_kind != "relu2_ridge"
        ):
            raise ValueError(
                "learned kernel_relu2_threshold_mode requires kernel "
                "relu2_ridge memory"
            )
        if self.kernel_relu2_threshold_mode != "none" and self.kernel_score_bias:
            raise ValueError(
                "kernel_relu2_threshold_mode and kernel_score_bias are "
                "redundant affine controls"
            )
        if not 0.0 < self.kernel_relative_threshold_init < 1.0:
            raise ValueError("kernel_relative_threshold_init must be in (0, 1)")
        if self.kernel_threshold_temperature <= 0.0:
            raise ValueError("kernel_threshold_temperature must be positive")
        if self.kernel_kind != "softmax_partition" \
                and self.softmax_gain_mode != "fixed":
            raise ValueError(
                "learned softmax_gain_mode requires "
                "kernel_kind='softmax_partition'"
            )
        if self.bspline_degree != 3 or self.bspline_basis_count <= self.bspline_degree:
            raise ValueError(
                "projected B-spline v1 requires degree=3 and basis_count > 3"
            )
        if self.bspline_bound <= 0.0 or self.bspline_scale <= 0.0:
            raise ValueError("B-spline bound and scale must be positive")
        if self.bspline_scale_mode != "fixed" \
                and self.kernel_kind != "projected_bspline":
            raise ValueError(
                "learned bspline_scale_mode requires projected_bspline features"
            )
        if self.read_norm:
            if self.rope != "none" and self.rope_placement == "feature":
                raise ValueError(
                    "feature-space RoPE makes the read_norm denominator signed; "
                    "use input-space/no RoPE or disable read_norm"
                )
            if self.accumulation not in ("sum", "ema_gate") \
                    or self.write_iters != 1:
                raise ValueError("read_norm requires accumulation='sum' or 'ema_gate', "
                                 "and write_iters=1")
            if self.write_rule == "gn" and self.nonlin != "relu2":
                raise ValueError("GN read_norm requires nonlin='relu2' so its "
                                 "feature-mass denominator is non-negative "
                                 "(learn_thresh is supported)")
        elif self.read_norm_mode != "key_mass":
            raise ValueError("read_norm_mode is active only when read_norm=True")
        if self.read_norm_w1:
            if not self.read_norm:
                raise ValueError("read_norm_w1 requires read_norm=True")
            if self.write_rule != "gn" or not self.fast_w1:
                raise ValueError("read_norm_w1 requires write_rule='gn' and fast_w1=True")
            if self.nonlin != "relu2":
                raise ValueError("read_norm_w1 requires nonlin='relu2' (learn_thresh is supported)")
