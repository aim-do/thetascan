# License: https://polyformproject.org/licenses/small-business/1.0.0
# Required Notice: Copyright 2026 Ultimamind SRL (Belgium).

"""Small bridge from the hybrid parameter-golf harness to ThetaScan's public API.

This file is copied into the prepared parameter-golf checkout.  It intentionally
contains model construction only: training policy stays in parameter-golf and
ThetaScan remains a normal installed dependency.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Final

from thetascan import (
    RoPEConfig,
    RuntimeConfig,
    ThetaScan,
    ThetaScanConfig,
)


RECIPES: Final = {
    "gn-reference-v0.1": ThetaScanConfig.gn_reference_v0_1,
    "gn-expanded-reference-v0.1": ThetaScanConfig.gn_expanded_reference_v0_1,
    "kernel-expanded-reference-v0.1": (
        ThetaScanConfig.kernel_expanded_reference_v0_1
    ),
}
ROPE_MODES: Final = ("none", "partial", "full")
BACKENDS: Final = ("auto", "naive", "quad", "cumsum", "fla")
PROJECTION_LAYOUTS: Final = ("mamba-shared", "transformer-gqa", "independent")


def build_thetascan_mixer(
    d_model: int,
    *,
    recipe: str,
    rope_mode: str,
    backend: str,
    projection_layout: str,
    layer_index: int | None = None,
) -> ThetaScan:
    """Build one benchmark mixer using only documented public configuration."""
    if recipe not in RECIPES:
        raise ValueError(f"unknown THETA_RECIPE={recipe!r}; choose one of {tuple(RECIPES)}")
    if rope_mode not in ROPE_MODES:
        raise ValueError(f"unknown THETA_ROPE={rope_mode!r}; choose one of {ROPE_MODES}")
    if backend not in BACKENDS:
        raise ValueError(f"unknown THETA_BACKEND={backend!r}; choose one of {BACKENDS}")
    if projection_layout not in PROJECTION_LAYOUTS:
        raise ValueError(
            f"unknown THETA_PROJECTION_LAYOUT={projection_layout!r}; "
            f"choose one of {PROJECTION_LAYOUTS}"
        )
    if d_model % 64:
        raise ValueError("the benchmark presets require d_model to be divisible by head_dim=64")

    config = RECIPES[recipe](d_model=d_model, n_heads=d_model // 64)
    if layer_index is not None:
        if type(layer_index) is not int or layer_index < 0:
            raise ValueError("layer_index must be a non-negative integer")
        # Expanded recipes give every swapped layer its own fixed expansion
        # maps; the suffix is inert for dense recipes.
        config.expansion_key = f"{config.expansion_key}:layer-{layer_index}"
    if projection_layout == "transformer-gqa":
        config.share_key_query = False
        config.key_value_heads = config.n_heads // 2
    elif projection_layout == "mamba-shared":
        config.share_key_query = True
        config.key_value_heads = None
    else:
        config.share_key_query = False
        config.key_value_heads = None
    # The RoPE choice is deliberately an explicit benchmark axis.  The RoPE
    # preset is the starting recommendation, not a claim that one mode wins.
    config.rope = RoPEConfig(mode=rope_mode, fraction=0.5)
    config.runtime = RuntimeConfig(backend=backend)
    config.validate()

    # The surrounding benchmark initializes every swapped mixer output at zero
    # so that the residual block starts as the identity.
    mixer = ThetaScan(config).zero_output_projection_()
    mixer._parameter_golf_config = {
        "recipe": recipe,
        "projection_layout": projection_layout,
        "resolved": asdict(config),
    }
    return mixer
