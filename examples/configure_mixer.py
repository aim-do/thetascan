"""Print and run one documented public ThetaScan recipe.

Examples:
    python examples/configure_mixer.py --recipe gn-reference-v0.1
    python examples/configure_mixer.py --recipe gn-expanded-reference-v0.1
    python examples/configure_mixer.py --recipe kernel-expanded-reference-v0.1
"""
from __future__ import annotations

import argparse

import torch

from thetascan import (
    RoPEConfig,
    ThetaScan,
    ThetaScanConfig,
)


def make_config(recipe: str) -> ThetaScanConfig:
    if recipe == "gn-reference-v0.1":
        return ThetaScanConfig.gn_reference_v0_1(d_model=64, n_heads=4)
    if recipe == "gn-expanded-reference-v0.1":
        return ThetaScanConfig.gn_expanded_reference_v0_1(
            d_model=64, n_heads=4
        )
    if recipe == "kernel-expanded-reference-v0.1":
        return ThetaScanConfig.kernel_expanded_reference_v0_1(
            d_model=64, n_heads=4
        )
    raise ValueError(recipe)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe",
        default="gn-reference-v0.1",
        choices=(
            "gn-reference-v0.1", "gn-expanded-reference-v0.1",
            "kernel-expanded-reference-v0.1",
        ),
    )
    parser.add_argument(
        "--rope",
        choices=("none", "partial", "full"),
        default=None,
        help="override the recipe's RoPE mode",
    )
    parser.add_argument(
        "--rope-fraction",
        type=float,
        default=None,
        help="override RoPE fraction; requires --rope",
    )
    parser.add_argument(
        "--rope-base",
        type=float,
        default=None,
        help="override RoPE base; requires --rope",
    )
    args = parser.parse_args()

    config = make_config(args.recipe)
    if args.rope is None and (
        args.rope_fraction is not None or args.rope_base is not None
    ):
        parser.error("--rope-fraction and --rope-base require --rope")
    if args.rope is not None:
        config.rope = RoPEConfig(
            mode=args.rope,
            fraction=(
                config.rope.fraction
                if args.rope_fraction is None
                else args.rope_fraction
            ),
            base=config.rope.base if args.rope_base is None else args.rope_base,
        )
    mixer = ThetaScan(config)
    x = torch.randn(2, 16, config.d_model)
    y = mixer(x)
    family_options = (
        config.gn if config.family == "gn" else config.kernel
    )
    print(
        f"family={config.family} d_model={config.d_model} n_heads={config.n_heads} "
        f"memory_multiplier={config.memory_multiplier} "
        f"feature_expansion={config.feature_expansion}"
    )
    print(f"family_options={family_options}")
    print(f"rope={config.rope} temporal={config.temporal}")
    print(f"recipe={args.recipe} input={tuple(x.shape)} output={tuple(y.shape)}")
    print(f"regularization_loss={mixer.regularization_loss().item():.6g}")


if __name__ == "__main__":
    main()
