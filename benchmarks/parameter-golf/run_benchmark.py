"""Run one retained parameter-golf benchmark variant from a prepared checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path


ARMS = ("attention", "thetascan", "mamba3")
RECIPES = (
    "gn-reference-v0.1",
    "gn-expanded-reference-v0.1",
    "kernel-expanded-reference-v0.1",
)
ROPE_MODES = ("none", "partial", "full")
BACKENDS = ("auto", "naive", "quad", "chunk", "cumsum", "fla")
OPTIMIZER_POLICIES = ("muon-2d", "muon-2d+theta")
PROJECTION_LAYOUTS = ("mamba-shared", "transformer-gqa", "independent")

# Prevent a caller's shell from silently changing the pinned harness defaults.
# Paths and SEED are captured explicitly before these keys are removed.
CONTROLLED_HYPERPARAMETER_ENV = (
    "ADAM_EPS", "BETA1", "BETA2", "CONTROL_TENSOR_NAME_PATTERNS",
    "CHECKPOINT_PATH", "DATA_PATH", "D_STATE", "EMBED_LR", "EXPAND",
    "EXPECTED_TRAIN_SHARDS", "GRAD_ACCUM_STEPS", "GRAD_CLIP_NORM",
    "HEAD_LR", "HEADDIM", "HYBRID_LAYERS", "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
    "ITERATIONS", "LOCAL_RANK", "LOGIT_SOFTCAP", "MAMBA3_BLOCK_PARITY",
    "MAMBA3_CHUNK", "MAMBA3_MLP_HIDDEN", "MATRIX_LR", "MAX_WALLCLOCK_SECONDS",
    "MIXER_TYPE", "MLP_HIDDEN", "MLP_MULT", "MODEL_DIM", "MUON_BACKEND_STEPS",
    "MUON_MOMENTUM", "MUON_MOMENTUM_WARMUP_START", "MUON_MOMENTUM_WARMUP_STEPS",
    "NUM_HEADS", "NUM_KV_HEADS", "NUM_LAYERS", "OPTIMIZER_POLICY",
    "QK_GAIN_INIT", "RANK", "RESUME_CHECKPOINT", "RESUME_CHECKPOINT_BYTES",
    "RESUME_CHECKPOINT_SHA256",
    "ROPE_BASE", "ROPE_FRACTION", "RUN_ID", "SCALAR_LR", "SEED",
    "THETA_BACKEND", "THETA_MLP_HIDDEN", "THETA_PROJECTION_LAYOUT",
    "SEGMENT_END_STEP", "SEGMENT_START_STEP", "THETA_RECIPE", "THETA_ROPE",
    "TIED_EMBED_INIT_STD",
    "TIED_EMBED_LR", "TIE_EMBEDDINGS", "TOKENIZER_PATH", "TRAIN_BATCH_TOKENS",
    "TRAIN_LOG_EVERY", "TRAIN_SEQ_LEN", "VAL_BATCH_SIZE", "VAL_LOSS_EVERY",
    "VAL_START_STEP", "VOCAB_SIZE", "WARMDOWN_ITERS", "WARMUP_STEPS",
    "WORLD_SIZE",
)


def _python_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="directory made by prepare_harness.py")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument(
        "--recipe",
        choices=RECIPES,
        help="required for the ThetaScan arm; invalid for external control arms",
    )
    parser.add_argument(
        "--rope",
        choices=("preset", *ROPE_MODES),
        default="preset",
        help="preserve the ThetaScan recipe's RoPE by default, or override it",
    )
    parser.add_argument("--backend", choices=BACKENDS, default="auto")
    parser.add_argument(
        "--projection-layout",
        choices=PROJECTION_LAYOUTS,
        default="mamba-shared",
        help=(
            "use one Mamba-like shared key/query group with per-head values, "
            "Transformer GQA with eight query and four pairwise key/value heads, "
            "or fully independent per-head Q/K/V projections"
        ),
    )
    parser.add_argument(
        "--optimizer-policy",
        choices=OPTIMIZER_POLICIES,
        default="muon-2d",
        help=(
            "keep Muon on ordinary 2D block matrices, or additionally route "
            "ThetaScan W1/W2 head batches through per-head Muon"
        ),
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--segment-start-step", type=int, default=0)
    parser.add_argument("--segment-end-step", type=int)
    parser.add_argument("--val-start-step", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--warmdown-iters", type=int, default=5)
    parser.add_argument("--max-wallclock-seconds", type=float, default=0.0)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=0,
        help=(
            "override the pinned harness accumulation count; zero preserves "
            "the official 8 // world_size default"
        ),
    )
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--expected-train-shards", type=int, default=0)
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint-sha256", default="")
    parser.add_argument("--resume-checkpoint-bytes", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.segment_end_step is None:
        args.segment_end_step = args.iterations
    if args.iterations < 1 or args.val_every < 1:
        parser.error("--iterations and --val-every must be positive")
    if not 0 <= args.segment_start_step < args.segment_end_step <= args.iterations:
        parser.error(
            "require 0 <= --segment-start-step < --segment-end-step <= --iterations"
        )
    if (
        args.val_start_step < 0
        or args.expected_train_shards < 0
        or args.grad_accum_steps < 0
    ):
        parser.error(
            "--val-start-step, --expected-train-shards, and "
            "--grad-accum-steps must be nonnegative"
        )
    if bool(args.resume_checkpoint) != (args.segment_start_step > 0):
        parser.error(
            "--resume-checkpoint is required exactly when --segment-start-step is positive"
        )
    if args.resume_checkpoint and (
        len(args.resume_checkpoint_sha256) != 64 or args.resume_checkpoint_bytes <= 0
    ):
        parser.error(
            "resume requires --resume-checkpoint-sha256 and positive "
            "--resume-checkpoint-bytes"
        )
    if args.resume_checkpoint_sha256 and not all(
        char in "0123456789abcdef" for char in args.resume_checkpoint_sha256
    ):
        parser.error("--resume-checkpoint-sha256 must be lowercase hexadecimal")
    if args.resume_checkpoint_sha256 and len(args.resume_checkpoint_sha256) != 64:
        parser.error("--resume-checkpoint-sha256 must contain 64 hex characters")
    if args.resume_checkpoint_bytes < 0:
        parser.error("--resume-checkpoint-bytes must be nonnegative")
    if args.optimizer_policy == "muon-2d+theta" and args.arm != "thetascan":
        parser.error(
            "--optimizer-policy muon-2d+theta requires --arm thetascan; "
            "attention and Mamba-3 use muon-2d"
        )
    if args.arm == "thetascan" and args.recipe is None:
        parser.error("--recipe is required with --arm thetascan")
    if args.arm != "thetascan" and args.recipe is not None:
        parser.error("--recipe is valid only with --arm thetascan")
    return args


def _base_environment(
    args: argparse.Namespace, seed: int, arm: str, recipe: str | None
) -> dict[str, str]:
    if arm not in ARMS:
        raise ValueError(f"unknown benchmark arm: {arm!r}")
    if arm == "thetascan" and recipe not in RECIPES:
        raise ValueError("ThetaScan requires one of the three public recipes")
    if arm != "thetascan" and recipe is not None:
        raise ValueError("external control arms do not accept a ThetaScan recipe")
    env = os.environ.copy()
    inherited_data_path = args.data_path or (
        Path(os.environ["DATA_PATH"]) if "DATA_PATH" in os.environ else None
    )
    inherited_tokenizer_path = args.tokenizer_path or (
        Path(os.environ["TOKENIZER_PATH"]) if "TOKENIZER_PATH" in os.environ else None
    )
    for key in CONTROLLED_HYPERPARAMETER_ENV:
        env.pop(key, None)
    env.update(
        SEED=str(seed),
        MODEL_DIM="512",
        NUM_HEADS="8",
        NUM_KV_HEADS="4",
        NUM_LAYERS="9",
        HYBRID_LAYERS="" if arm == "attention" else "4,5",
        MLP_HIDDEN="1024",
        ITERATIONS=str(args.iterations),
        SEGMENT_START_STEP=str(args.segment_start_step),
        SEGMENT_END_STEP=str(args.segment_end_step),
        VAL_START_STEP=str(args.val_start_step),
        VAL_LOSS_EVERY=str(args.val_every),
        WARMUP_STEPS=str(args.warmup_steps),
        WARMDOWN_ITERS=str(args.warmdown_iters),
        MAX_WALLCLOCK_SECONDS=str(args.max_wallclock_seconds),
        TRAIN_BATCH_TOKENS="524288",
        TRAIN_SEQ_LEN="1024",
        MATRIX_LR="0.04",
        OPTIMIZER_POLICY=args.optimizer_policy,
        EXPECTED_TRAIN_SHARDS=str(args.expected_train_shards),
    )
    if inherited_data_path:
        env["DATA_PATH"] = str(inherited_data_path.resolve())
    if inherited_tokenizer_path:
        env["TOKENIZER_PATH"] = str(inherited_tokenizer_path.resolve())
    if args.checkpoint_output:
        env["CHECKPOINT_PATH"] = str(args.checkpoint_output.resolve())
    grad_accum_steps = getattr(args, "grad_accum_steps", None)
    if grad_accum_steps:
        env["GRAD_ACCUM_STEPS"] = str(grad_accum_steps)
    if args.resume_checkpoint:
        env["RESUME_CHECKPOINT"] = str(args.resume_checkpoint.resolve())
        env["RESUME_CHECKPOINT_SHA256"] = args.resume_checkpoint_sha256
        env["RESUME_CHECKPOINT_BYTES"] = str(args.resume_checkpoint_bytes)
    if arm == "attention":
        env.update(MIXER_TYPE="attn", MAMBA3_BLOCK_PARITY="0")
    elif arm == "thetascan":
        theta_mlp_hidden = {
            "mamba-shared": "1023",
            "transformer-gqa": "832",
            "independent": "576",
        }[args.projection_layout]
        if recipe == "gn-expanded-reference-v0.1":
            # The expanded GN threshold vector lives at the doubled effective
            # width (3,072 extra parameters per block); shrinking the swapped
            # FFN by 3 channels gives the same amount back:
            # 2 blocks * 2 matrices * 512 * 3 = 6,144.
            theta_mlp_hidden = str(int(theta_mlp_hidden) - 3)
        env.update(
            MIXER_TYPE="thetascan",
            MAMBA3_BLOCK_PARITY="0",
            THETA_MLP_HIDDEN=theta_mlp_hidden,
            THETA_PROJECTION_LAYOUT=args.projection_layout,
            THETA_RECIPE=str(recipe),
            THETA_BACKEND=args.backend,
        )
    else:
        # Parameter matching changes only the two surrounding FFNs.  The
        # official Mamba-3 module and its full mixer dimensions are untouched.
        env.update(
            MIXER_TYPE="mamba3",
            MAMBA3_BLOCK_PARITY="1",
            MAMBA3_MLP_HIDDEN="938",
            D_STATE="64",
            HEADDIM="64",
            EXPAND="1",
            ROPE_FRACTION="0.5",
            MAMBA3_CHUNK="64",
        )
    return env


def _run_spec(
    arm: str, rope: str, recipe: str | None
) -> tuple[str, str | None, str]:
    """Resolve one of the five retained benchmark variants."""
    if arm == "thetascan":
        if recipe not in RECIPES:
            raise ValueError("ThetaScan requires one of the three public recipes")
        return arm, recipe, "partial" if rope == "preset" else rope
    if recipe is not None:
        raise ValueError("external control arms do not accept a ThetaScan recipe")
    return arm, None, "n/a"


IMPORTANT_ENV = (
    "SEED",
    "DATA_PATH",
    "TOKENIZER_PATH",
    "MIXER_TYPE",
    "HYBRID_LAYERS",
    "MODEL_DIM",
    "NUM_LAYERS",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "MLP_HIDDEN",
    "MAMBA3_BLOCK_PARITY",
    "MAMBA3_MLP_HIDDEN",
    "THETA_MLP_HIDDEN",
    "D_STATE",
    "HEADDIM",
    "EXPAND",
    "ROPE_FRACTION",
    "MAMBA3_CHUNK",
    "THETA_RECIPE",
    "THETA_ROPE",
    "THETA_BACKEND",
    "THETA_PROJECTION_LAYOUT",
    "ITERATIONS",
    "SEGMENT_START_STEP",
    "SEGMENT_END_STEP",
    "VAL_START_STEP",
    "VAL_LOSS_EVERY",
    "TRAIN_BATCH_TOKENS",
    "TRAIN_SEQ_LEN",
    "MATRIX_LR",
    "OPTIMIZER_POLICY",
    "GRAD_ACCUM_STEPS",
    "EXPECTED_TRAIN_SHARDS",
    "CHECKPOINT_PATH",
    "RESUME_CHECKPOINT",
)


def _verify_prepared_checkout(checkout: Path) -> dict:
    entrypoint = checkout / "ours" / "train_gpt_hybrid.py"
    manifest_path = checkout / "ours" / "ADAPTER-MANIFEST.json"
    if not entrypoint.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"{checkout} is not a prepared checkout; run prepare_harness.py first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["generated_sha256"].items():
        path = checkout / "ours" / name
        if not path.is_file():
            raise FileNotFoundError(f"prepared adapter file is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"prepared adapter hash mismatch for {name}: "
                f"expected {expected}, got {actual}"
            )

    try:
        import thetascan
    except ImportError as exc:
        raise RuntimeError(
            "ThetaScan is not installed in this environment; run "
            "`python -m pip install -e .` from the ThetaScan source checkout"
        ) from exc
    expected_source = manifest["thetascan_source"]
    actual_version = thetascan.__version__
    package_root = Path(thetascan.__file__).resolve().parent
    actual_source_hash = _python_tree_sha256(package_root)
    if actual_version != expected_source["version"]:
        raise RuntimeError(
            f"ThetaScan version mismatch: prepared with {expected_source['version']}, "
            f"running {actual_version}"
        )
    if actual_source_hash != expected_source["source_sha256"]:
        raise RuntimeError(
            "ThetaScan source mismatch: rerun prepare_harness.py with the installed "
            "ThetaScan source that will execute the benchmark"
        )
    return manifest


def _command(checkout: Path, nproc: int) -> list[str]:
    entrypoint = checkout / "ours" / "train_gpt_hybrid.py"
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(entrypoint),
    ]


def main() -> None:
    args = _parse_args()
    checkout = args.checkout.resolve()
    manifest = _verify_prepared_checkout(checkout)
    command = _command(checkout, args.nproc_per_node)
    # A run gets a fresh seed by default; an explicitly supplied SEED remains a
    # supported expert override.  No seed or result is committed to this repo.
    seed = int(os.environ.get("SEED", secrets.randbelow(2**31 - 1)))
    arm, recipe, rope_mode = _run_spec(args.arm, args.rope, args.recipe)
    print(f"adapter_manifest={checkout / 'ours' / 'ADAPTER-MANIFEST.json'}")
    print(f"thetascan_source={manifest['thetascan_source']}")

    env = _base_environment(args, seed, arm, recipe)
    if arm == "thetascan":
        env["THETA_ROPE"] = rope_mode
    print(
        f"run arm={arm} recipe={recipe or 'n/a'} theta_rope={rope_mode} "
        f"backend={args.backend} projection_layout={args.projection_layout} "
        f"optimizer_policy={args.optimizer_policy} "
        f"seed={seed}"
    )
    if args.dry_run:
        print("command:", subprocess.list2cmdline(command))
        print(
            "environment:",
            " ".join(f"{key}={env[key]}" for key in IMPORTANT_ENV if key in env),
        )
        return
    subprocess.run(command, cwd=checkout, env=env, check=True)


if __name__ == "__main__":
    main()
