"""Positive feature maps used by normalized kernel-regression memory.

The scan backends do not need to know which map produced a feature tensor.  A
map only has to return ``[batch, heads, time, features]``; the existing scans
then accumulate its numerator and matched feature mass.
"""
from __future__ import annotations

import torch


def _head_parameter(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a ``[H]`` or ``[H,M]`` control over a head-major tensor."""
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 1 and value.shape[0] == reference.shape[1]:
        return value.view(1, value.shape[0], *([1] * (reference.ndim - 2)))
    if (
        value.ndim == 2
        and value.shape[0] == reference.shape[1]
        and value.shape[1] == reference.shape[-1]
    ):
        return value.view(
            1,
            value.shape[0],
            *([1] * (reference.ndim - 3)),
            value.shape[1],
        )
    raise ValueError(
        "kernel control must have shape [n_heads] or [n_heads, n_features], "
        f"got {tuple(value.shape)} for reference {tuple(reference.shape)}"
    )


def _gain(value: float | torch.Tensor | None, fallback: float,
          reference: torch.Tensor) -> float | torch.Tensor:
    if value is None:
        return fallback
    if not torch.is_tensor(value):
        return float(value)
    return _head_parameter(value, reference)


def relative_threshold_features(
    logits: torch.Tensor,
    rho: torch.Tensor,
    *,
    temperature: float,
    straight_through: bool,
    eps: float,
) -> torch.Tensor:
    """Relative-to-maximum sparse simplex map with a learned per-head cutoff.

    A feature is active when ``p_j > rho_h * max(p)``.  In logit space this is
    ``logit_j - max(logit) > log(rho_h)``.  The straight-through form emits
    exact zeros in the forward pass while differentiating through a sigmoid
    approximation, so an inactive feature can become active again.
    """
    if temperature <= 0.0:
        raise ValueError("relative threshold temperature must be positive")
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    rho_b = _head_parameter(rho, logits).clamp(min=eps, max=1.0 - eps)
    margin = shifted - rho_b.log()
    soft_gate = torch.sigmoid(margin / temperature)
    if straight_through:
        hard_gate = (margin >= 0.0).to(dtype=soft_gate.dtype)
        gate = hard_gate + soft_gate - soft_gate.detach()
    else:
        gate = soft_gate
    weights = shifted.exp() * gate
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)


def softmax_partition_features(
    pre: torch.Tensor,
    *,
    gain: float | torch.Tensor | None,
    fallback_gain: float,
    score_bias: torch.Tensor | None,
    sparsity: str,
    relative_threshold: torch.Tensor | None,
    sparse_blend_alpha: torch.Tensor | None = None,
    threshold_temperature: float,
    eps: float,
) -> torch.Tensor:
    """Softmax of learned linear scores, including optional affine/sparse axes."""
    scores = pre
    if score_bias is not None:
        scores = scores + _head_parameter(score_bias, scores)
    logits = scores * _gain(gain, fallback_gain, scores)
    dense = torch.softmax(logits, dim=-1)
    if sparsity == "none":
        return dense
    if relative_threshold is None:
        raise RuntimeError("relative kernel sparsity requires a threshold tensor")
    if sparsity not in ("relative_soft", "relative_st", "relative_st_blend"):
        raise ValueError(f"unknown kernel sparsity: {sparsity!r}")
    sparse = relative_threshold_features(
        logits,
        relative_threshold,
        temperature=threshold_temperature,
        straight_through=sparsity in ("relative_st", "relative_st_blend"),
        eps=eps,
    )
    if sparsity != "relative_st_blend":
        return sparse
    if sparse_blend_alpha is None:
        raise RuntimeError(
            "relative_st_blend requires a learned sparse blend tensor"
        )
    alpha = _head_parameter(sparse_blend_alpha, logits)
    # Forward is a convex mixture because alpha is bounded in [0,1].  Writing it
    # as dense + alpha*(sparse-dense) preserves the dense map exactly at alpha=0.
    return dense + alpha * (sparse - dense)


def relu2_ridge_features(
    pre: torch.Tensor,
    *,
    score_bias: torch.Tensor | None,
    relu2_threshold: torch.Tensor | None = None,
    eps: float,
) -> torch.Tensor:
    """Positive global half-space/ridge simplex with no inner-loop solve."""
    if score_bias is not None:
        pre = pre + _head_parameter(score_bias, pre)
    if relu2_threshold is not None:
        pre = pre - _head_parameter(relu2_threshold, pre)
    features = torch.relu(pre).square()
    mass = features.sum(dim=-1, keepdim=True)
    normalized = features / mass.clamp_min(eps)
    # A pathological all-negative row has no active half-space.  A uniform
    # fallback preserves a valid memory address without hiding gradients in the
    # normal region (with hundreds of ridges this branch is vanishingly rare).
    uniform = torch.full_like(features, 1.0 / features.shape[-1])
    return torch.where(mass > eps, normalized, uniform)


def open_uniform_bspline_basis(
    coordinates: torch.Tensor,
    *,
    n_basis: int,
    degree: int = 3,
    bound: float = 3.0,
) -> torch.Tensor:
    """Evaluate a clamped open-uniform B-spline basis.

    The final dimension is the projection axis.  The returned tensor appends a
    basis axis, is non-negative, sums to one, and has at most ``degree + 1``
    non-zero entries per projection.  Knot selection is piecewise constant but
    the polynomial basis has the correct coordinate gradient almost everywhere.
    """
    if degree < 0:
        raise ValueError("B-spline degree must be non-negative")
    if n_basis <= degree:
        raise ValueError("B-spline basis count must exceed its degree")
    if bound <= 0.0:
        raise ValueError("B-spline bound must be positive")

    dtype, device = coordinates.dtype, coordinates.device
    low = torch.tensor(-bound, dtype=dtype, device=device)
    high = torch.tensor(bound, dtype=dtype, device=device)
    # Open-uniform: p+1 repeated endpoint knots and G-p-1 interior knots.
    n_inner = n_basis - degree - 1
    if n_inner:
        interior = torch.linspace(
            -bound, bound, n_inner + 2, dtype=dtype, device=device
        )[1:-1]
    else:
        interior = coordinates.new_empty(0)
    knots = torch.cat(
        [low.repeat(degree + 1), interior, high.repeat(degree + 1)]
    )

    # The recursive definition uses half-open knot spans.  Move the right
    # endpoint one representable value inward so x=bound maps to the last basis.
    high_inside = torch.nextafter(high, low)
    x = coordinates.clamp(min=-bound, max=bound)
    x = torch.minimum(x, high_inside)
    basis = (
        (x.unsqueeze(-1) >= knots[:-1])
        & (x.unsqueeze(-1) < knots[1:])
    ).to(dtype)

    tiny = torch.finfo(dtype).tiny
    for order in range(1, degree + 1):
        count = knots.numel() - order - 1
        left_den = knots[order:order + count] - knots[:count]
        right_den = knots[order + 1:order + 1 + count] - knots[1:1 + count]
        left_coef = torch.where(
            left_den > 0,
            (x.unsqueeze(-1) - knots[:count]) / left_den.clamp_min(tiny),
            torch.zeros_like(left_den),
        )
        right_coef = torch.where(
            right_den > 0,
            (knots[order + 1:order + 1 + count] - x.unsqueeze(-1))
            / right_den.clamp_min(tiny),
            torch.zeros_like(right_den),
        )
        basis = left_coef * basis[..., :count] + right_coef * basis[..., 1:count + 1]

    # Roundoff near repeated endpoint knots can produce tiny negatives or a sum
    # infinitesimally different from one.  Restore the mathematical invariant.
    basis = basis.clamp_min(0.0)
    return basis / basis.sum(dim=-1, keepdim=True).clamp_min(tiny)


def projected_bspline_features(
    pre: torch.Tensor,
    *,
    n_basis: int,
    degree: int,
    bound: float,
    scale: float | torch.Tensor | None,
    fallback_scale: float,
) -> torch.Tensor:
    """Tensor-product-free projection-pursuit B-spline feature map.

    Each learned direction produces a one-dimensional partition of unity.  A
    basis cell is local along that direction but global in every orthogonal
    direction (an infinite slab in input space).  Concatenating and dividing by
    the direction count gives a global partition of unity suitable for either
    the ordinary kernel denominator or per-feature mass normalization.
    """
    coordinates = pre * _gain(scale, fallback_scale, pre)
    basis = open_uniform_bspline_basis(
        coordinates, n_basis=n_basis, degree=degree, bound=bound
    )
    n_directions = pre.shape[-1]
    return basis.flatten(-2) / float(n_directions)


def feature_map(
    pre: torch.Tensor,
    cfg,
    *,
    softmax_gain: float | torch.Tensor | None = None,
    spline_scale: float | torch.Tensor | None = None,
    score_bias: torch.Tensor | None = None,
    relu2_threshold: torch.Tensor | None = None,
    relative_threshold: torch.Tensor | None = None,
    sparse_blend_alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch the configured positive kernel feature map."""
    if cfg.kernel_kind == "softmax_partition":
        return softmax_partition_features(
            pre,
            gain=softmax_gain,
            fallback_gain=cfg.softmax_gain,
            score_bias=score_bias,
            sparsity=cfg.kernel_sparsity,
            relative_threshold=relative_threshold,
            sparse_blend_alpha=sparse_blend_alpha,
            threshold_temperature=cfg.kernel_threshold_temperature,
            eps=cfg.eps,
        )
    if cfg.kernel_kind == "relu2_ridge":
        return relu2_ridge_features(
            pre,
            score_bias=score_bias,
            relu2_threshold=relu2_threshold,
            eps=cfg.eps,
        )
    if cfg.kernel_kind == "projected_bspline":
        return projected_bspline_features(
            pre,
            n_basis=cfg.bspline_basis_count,
            degree=cfg.bspline_degree,
            bound=cfg.bspline_bound,
            scale=spline_scale,
            fallback_scale=cfg.bspline_scale,
        )
    raise ValueError(f"unknown kernel kind: {cfg.kernel_kind!r}")
