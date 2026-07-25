"""Same-process A/B of the scan backends for the promoted presets.

Backend choice is an execution decision, so it must be measured on one device in
one process: pod-to-pod and machine-to-machine variance is larger than the
effect being measured.  Every backend is therefore timed in the same run, and
the whole backend list is replayed in reverse so that warm-up drift and clock
ramping cannot be attributed to the backend that happened to go first.

    python benchmarks/scan_backends.py                       # defaults
    python benchmarks/scan_backends.py --seq-len 2048 --dtype bf16
    python benchmarks/scan_backends.py --backends quad chunk:64 chunk:128 fla

``chunk:N`` selects the chunk backend with ``scan_chunk=N``.  On CUDA the script
reports peak allocated memory; on CPU, where there is no allocator high-water
mark, it reports the bytes autograd retains for backward, which is the quantity
the chunked backend is designed to reduce.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

from thetascan import ThetaScan, ThetaScanConfig

PRESETS = (
    "gn_reference_v0_1",
    "gn_expanded_reference_v0_1",
    "kernel_expanded_reference_v0_1",
)
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _parse_backend(token: str) -> tuple[str, int | None]:
    if ":" in token:
        name, _, size = token.partition(":")
        return name, int(size)
    return token, None


class RetainedBytes:
    """Bytes autograd keeps alive for backward, as a device-free proxy."""

    def __init__(self) -> None:
        self.total = 0

    def __enter__(self) -> "RetainedBytes":
        def pack(tensor: torch.Tensor) -> torch.Tensor:
            self.total += tensor.numel() * tensor.element_size()
            return tensor

        self._hooks = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._hooks.__exit__(*exc)


def _build(preset: str, backend: str, chunk: int | None, args) -> ThetaScan:
    runtime: dict[str, object] = {"backend": backend}
    if chunk is not None:
        runtime["scan_chunk"] = chunk
    config = ThetaScanConfig.from_dict({
        "preset": preset,
        "d_model": args.model_dim,
        "n_heads": args.heads,
        "runtime": runtime,
    })
    mixer = ThetaScan(config).to(args.device)
    if args.dtype != "fp32":
        mixer = mixer.to(DTYPES[args.dtype])
    return mixer


def _step(mixer: ThetaScan, x: torch.Tensor) -> None:
    out = mixer(x)
    out.float().square().mean().backward()
    mixer.zero_grad(set_to_none=True)


def _measure(preset: str, token: str, args) -> dict[str, float] | None:
    backend, chunk = _parse_backend(token)
    device = torch.device(args.device)
    try:
        mixer = _build(preset, backend, chunk, args)
    except Exception as exc:                       # unsupported combination
        print(f"  skip {token}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    x = torch.randn(
        args.batch, args.seq_len, args.model_dim,
        device=device, dtype=DTYPES[args.dtype],
    )

    compile_seconds = 0.0
    if args.compile:
        # Match the benchmark harness policy: a ThetaScan arm compiles without
        # fullgraph, so a graph break is tolerated rather than fatal.  Reset
        # dynamo per backend so one backend's guards and recompile budget cannot
        # influence the next measurement in the same process.
        torch._dynamo.reset()
        mixer = torch.compile(mixer, dynamic=False, fullgraph=False)

    retained = None
    try:
        # A backend can fail inside forward or backward rather than at
        # construction: FLA raises from its gated backward on some
        # Hopper/Triton builds.  One backend failing must not lose the
        # measurements for the rest of the list.
        compile_start = time.perf_counter()
        for _ in range(args.warmup):
            _step(mixer, x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if args.compile:
            # Warm-up wall time is dominated by compilation, which is paid once
            # per training run and must not be mixed into the per-step median.
            compile_seconds = time.perf_counter() - compile_start
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif not args.compile:
            # Saved-tensor hooks see the eager graph.  Under a compiled graph the
            # retained set is inductor's, not this one, so do not report a number
            # that would invite a false comparison.
            with RetainedBytes() as counter:
                mixer(x)
            retained = counter.total

        samples: list[float] = []
        for _ in range(args.reps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _step(mixer, x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1e3)
    except torch.cuda.OutOfMemoryError:
        print(f"  skip {token}: out of memory", file=sys.stderr)
        return None
    except RuntimeError as exc:
        print(f"  skip {token}: {exc}".splitlines()[0], file=sys.stderr)
        return None
    finally:
        peak = (
            torch.cuda.max_memory_allocated() / 2 ** 20
            if device.type == "cuda" else 0.0
        )
        del mixer, x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "ms": statistics.median(samples),
        "peak_mib": peak,
        "retained_mib": (retained or 0) / 2 ** 20,
        "compile_s": compile_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="+", default=list(PRESETS))
    parser.add_argument(
        "--backends", nargs="+",
        default=["quad", "chunk:64", "chunk:128", "chunk:256"],
    )
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--compile", action="store_true",
        help="wrap each mixer in torch.compile with the benchmark harness "
             "policy (dynamic=False, fullgraph=False). Compilation time is "
             "reported separately and excluded from the per-step median.",
    )
    args = parser.parse_args()
    if args.compile and args.warmup < 3:
        # The first call compiles and the next few may recompile on guards.
        args.warmup = 3

    if args.device.startswith("cuda"):
        print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"torch {torch.__version__} | dtype {args.dtype} | "
        f"batch {args.batch} | T {args.seq_len} | "
        f"d_model {args.model_dim} | heads {args.heads} | reps {args.reps} | "
        f"{'torch.compile' if args.compile else 'eager'}"
    )
    memory_label = "peak MiB" if args.device.startswith("cuda") else "retained MiB"

    for preset in args.presets:
        print(f"\n### {preset}")
        # Replay the list in reverse so ordering cannot masquerade as an effect.
        order = list(args.backends) + list(reversed(args.backends))
        results: dict[str, list[dict[str, float]]] = {}
        for token in order:
            measured = _measure(preset, token, args)
            if measured is not None:
                results.setdefault(token, []).append(measured)

        if not results:
            print("no backend completed")
            continue
        table = {
            token: {
                "ms": statistics.mean(run["ms"] for run in runs),
                "mem": max(
                    run["peak_mib"] if args.device.startswith("cuda")
                    else run["retained_mib"] for run in runs
                ),
                "compile_s": max(run["compile_s"] for run in runs),
            }
            for token, runs in results.items()
        }
        baseline = table.get("quad") or next(iter(table.values()))
        extra = " compile s |" if args.compile else ""
        print(
            f"| backend | fwd+bwd ms | speedup | {memory_label} | "
            f"memory ratio |{extra}"
        )
        print("|---|---:|---:|---:|---:|" + ("---:|" if args.compile else ""))
        for token, row in table.items():
            speed = baseline["ms"] / row["ms"] if row["ms"] else float("nan")
            ratio = baseline["mem"] / row["mem"] if row["mem"] else float("nan")
            tail = f" {row['compile_s']:.0f} |" if args.compile else ""
            print(
                f"| {token} | {row['ms']:.2f} | {speed:.2f}x | "
                f"{row['mem']:.1f} | {ratio:.2f}x |{tail}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
