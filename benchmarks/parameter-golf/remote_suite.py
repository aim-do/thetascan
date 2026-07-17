"""Execute one versioned parameter-golf suite inside an already prepared pod.

This module contains no RunPod credentials or API calls.  It runs the public
adapter sequentially, writes only sanitized logs/JSON to ``export_dir``, and
deletes model artifacts after every arm.  ``run_runpod.py`` is responsible for
the pod lifecycle and for placing the pinned checkout and source snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_CAP_BYTES = 16_000_000
ROUNDTRIP_EXACT_PATTERN = (
    r"^final_int8_zlib_roundtrip_exact val_loss:[0-9.eE+-]+ "
    r"val_bpb:[0-9.eE+-]+$"
)
ARTIFACT_SIZE_PATTERN = r"^Total submission size int8\+zlib: (\d+) bytes$"
CHECKPOINT_FILENAME = "research-checkpoint.pt"
CHECKPOINT_METADATA_FILENAME = "research-checkpoint.json"
DEFAULT_GRAD_ACCUM_STEPS = 8
THETASCAN_RECIPES = frozenset(
    {
        "gn-reference-v0.1",
        "gn-expanded-reference-v0.1",
        "kernel-expanded-reference-v0.1",
    }
)
CHECKPOINT_PATTERN = (
    r"^checkpoint_saved:path=.* step:(\d+) target:(\d+) bytes:(\d+) "
    r"sha256:([0-9a-f]{64})$"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_resume_checkpoint(
    checkpoint: Path,
    expected_sha256: str,
    expected_bytes: int,
) -> Path:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"resume checkpoint not found: {checkpoint}")
    if expected_bytes <= 0:
        raise RuntimeError("resume checkpoint byte contract must be positive")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError("resume checkpoint SHA-256 contract is invalid")
    actual_bytes = checkpoint.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            "resume checkpoint size mismatch: "
            f"expected={expected_bytes}, actual={actual_bytes}"
        )
    actual_sha256 = _sha256(checkpoint)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "resume checkpoint SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    return checkpoint


def _parse_metrics(text: str) -> dict[str, Any]:
    size_matches = re.findall(ARTIFACT_SIZE_PATTERN, text, flags=re.MULTILINE)
    artifact_bytes = int(size_matches[-1]) if size_matches else None
    metrics: dict[str, Any] = {
        "artifact_bytes": artifact_bytes,
        "artifact_cap_bytes": ARTIFACT_CAP_BYTES,
        "artifact_within_cap": (
            artifact_bytes is not None and artifact_bytes <= ARTIFACT_CAP_BYTES
        ),
        "final_int8_roundtrip_exact": bool(
            re.search(ROUNDTRIP_EXACT_PATTERN, text, flags=re.MULTILINE)
        ),
    }
    scalar_patterns = {
        "model_params": r"^model_params:(\d+)$",
        "final_int8_val_bpb": (
            r"^final_int8_zlib_roundtrip_exact val_loss:[0-9.eE+-]+ "
            r"val_bpb:([0-9.eE+-]+)$"
        ),
        "peak_memory_mib": r"^peak_memory_mib:([0-9.eE+-]+)$",
    }
    for key, pattern in scalar_patterns.items():
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if matches:
            value = matches[-1]
            metrics[key] = int(value) if key == "model_params" else float(value)

    checkpoints = []
    for match in re.finditer(
        r"^step:(\d+)/(\d+) val_loss:([0-9.eE+-]+) val_bpb:([0-9.eE+-]+) "
        r"train_time:([0-9.eE+-]+)ms step_avg:([0-9.eE+-]+)ms$",
        text,
        flags=re.MULTILINE,
    ):
        checkpoints.append(
            {
                "step": int(match.group(1)),
                "iterations": int(match.group(2)),
                "val_loss": float(match.group(3)),
                "val_bpb": float(match.group(4)),
                "train_time_ms": float(match.group(5)),
                "step_avg_ms": float(match.group(6)),
            }
        )
    metrics["checkpoints"] = checkpoints

    for key, prefix in (
        ("muon_2d", "optimizer_muon_2d"),
        ("muon_theta", "optimizer_muon_theta"),
        ("adam_theta_controls", "optimizer_adam_theta_controls"),
        ("adam_block_and_skip", "optimizer_adam_block_and_skip"),
    ):
        match = re.search(
            rf"^{prefix}:tensor_count=(\d+) parameter_count=(\d+)",
            text,
            flags=re.MULTILINE,
        )
        if match:
            metrics[key] = {
                "tensor_count": int(match.group(1)),
                "parameter_count": int(match.group(2)),
            }
    parity = re.search(r"^mamba3_block_parity:(.+)$", text, flags=re.MULTILINE)
    if parity:
        metrics["mamba3_block_parity"] = parity.group(1)
    return metrics


def _artifact_contract_failures(
    metrics: dict[str, Any], *, allow_oversize: bool = False
) -> list[str]:
    failures: list[str] = []
    if not metrics.get("final_int8_roundtrip_exact"):
        failures.append("missing exact int8+zlib roundtrip validation")
    artifact_bytes = metrics.get("artifact_bytes")
    if artifact_bytes is None:
        failures.append("missing Total submission size int8+zlib byte count")
    elif artifact_bytes > ARTIFACT_CAP_BYTES and not allow_oversize:
        failures.append(
            f"int8+zlib artifact exceeds {ARTIFACT_CAP_BYTES} byte cap: "
            f"observed={artifact_bytes}"
        )
    return failures


def _resolved_grad_accum_steps(protocol: dict[str, Any]) -> int:
    configured = protocol.get("grad_accum_steps", 0)
    if not isinstance(configured, int) or isinstance(configured, bool):
        raise ValueError("protocol grad_accum_steps must be an integer")
    if configured < 0:
        raise ValueError("protocol grad_accum_steps must be nonnegative")
    # The remote suite always launches one process. The pinned harness default is
    # 8 // world_size, so normalize zero/omitted to the actual runtime value.
    return configured or DEFAULT_GRAD_ACCUM_STEPS


def _command(
    benchmark_runner: Path,
    checkout: Path,
    run: dict[str, Any],
    protocol: dict[str, Any],
    expected_train_shards: int,
    checkpoint_output: Path | None,
    resume_checkpoint: Path | None,
    resume_checkpoint_sha256: str,
    resume_checkpoint_bytes: int,
) -> list[str]:
    grad_accum_steps = _resolved_grad_accum_steps(protocol)
    command = [
        sys.executable,
        str(benchmark_runner),
        str(checkout),
        "--arm",
        run["arm"],
        "--rope",
        run["rope"],
        "--backend",
        run["backend"],
        "--projection-layout",
        run.get("projection_layout", "mamba-shared"),
        "--optimizer-policy",
        run["optimizer_policy"],
        "--iterations",
        str(protocol["iterations"]),
        "--segment-start-step",
        str(protocol.get("segment_start_step", 0)),
        "--segment-end-step",
        str(protocol.get("segment_end_step", protocol["iterations"])),
        "--val-start-step",
        str(protocol.get("val_start_step", 0)),
        "--val-every",
        str(protocol["val_every"]),
        "--warmup-steps",
        str(protocol["warmup_steps"]),
        "--warmdown-iters",
        str(protocol["warmdown_iters"]),
        "--max-wallclock-seconds",
        str(protocol.get("max_wallclock_seconds", 0.0)),
        "--nproc-per-node",
        "1",
        "--expected-train-shards",
        str(expected_train_shards),
        "--grad-accum-steps",
        str(grad_accum_steps),
    ]
    if run["arm"] == "thetascan":
        recipe = run.get("recipe")
        if recipe not in THETASCAN_RECIPES:
            raise ValueError(
                "ThetaScan run requires one of the three public recipes; "
                f"got {recipe!r}"
            )
        command.extend(("--recipe", recipe))
    elif run.get("recipe") is not None:
        raise ValueError("external control arms must not define a ThetaScan recipe")
    if checkpoint_output is not None:
        command.extend(("--checkpoint-output", str(checkpoint_output)))
    if resume_checkpoint is not None:
        command.extend(
            (
                "--resume-checkpoint",
                str(resume_checkpoint),
                "--resume-checkpoint-sha256",
                resume_checkpoint_sha256,
                "--resume-checkpoint-bytes",
                str(resume_checkpoint_bytes),
            )
        )
    return command


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()


def _run_one(
    *,
    stage: str,
    run: dict[str, Any],
    protocol: dict[str, Any],
    benchmark_runner: Path,
    checkout: Path,
    export_dir: Path,
    seed: int,
    status: dict[str, Any],
    expected_train_shards: int,
    resume_checkpoint: Path | None,
    resume_checkpoint_sha256: str,
    resume_checkpoint_bytes: int,
) -> dict[str, Any]:
    run_id = run["id"]
    log_name = f"{stage}-{run_id}.log"
    log_path = export_dir / log_name
    stage_resume = resume_checkpoint
    checkpoint_required = bool(protocol.get("checkpoint_at_end", False))
    checkpoint_output = (
        checkout / CHECKPOINT_FILENAME if checkpoint_required else None
    )
    if checkpoint_output is not None and (
        stage_resume is None
        or checkpoint_output.resolve() != stage_resume.resolve()
    ):
        checkpoint_output.unlink(missing_ok=True)
    command = _command(
        benchmark_runner,
        checkout,
        run,
        protocol,
        expected_train_shards,
        checkpoint_output,
        stage_resume,
        resume_checkpoint_sha256 if stage_resume is not None else "",
        resume_checkpoint_bytes if stage_resume is not None else 0,
    )
    env = os.environ.copy()
    env["SEED"] = str(seed)
    env["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    deadline = started + float(protocol["timeout_seconds"])
    status.update(stage=stage, run=run_id, state="running", started_at=_utc_now())
    _write_json(export_dir / "status.json", status)

    return_code: int | None = None
    error: str | None = None
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write("command=" + subprocess.list2cmdline(command) + "\n")
        log.write(f"shared_runtime_seed={seed}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=checkout,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            status.update(elapsed_seconds=round(elapsed, 1), updated_at=_utc_now())
            _write_json(export_dir / "status.json", status)
            if time.monotonic() >= deadline:
                error = f"run timeout after {protocol['timeout_seconds']} seconds"
                _terminate_process(process)
                break
            time.sleep(10)
        return_code = process.poll()

    text = log_path.read_text(encoding="utf-8", errors="replace")
    required_patterns = list(run.get("required_log_regex", []))
    theta_tensors = run.get("expected_theta_headwise_muon_tensors")
    theta_params = run.get("expected_theta_headwise_muon_params")
    if theta_tensors is not None and theta_params is not None:
        required_patterns.append(
            rf"optimizer_muon_theta:tensor_count={theta_tensors} "
            rf"parameter_count={theta_params}"
        )
    missing = [pattern for pattern in required_patterns if not re.search(pattern, text)]
    metrics = _parse_metrics(text)
    allow_oversize = protocol.get("research_allow_oversize_artifact", False)
    if type(allow_oversize) is not bool:
        missing.append("research_allow_oversize_artifact must be boolean")
        allow_oversize = False
    if allow_oversize and run.get("recipe") not in {
        "gn-expanded-reference-v0.1",
        "kernel-expanded-reference-v0.1",
    }:
        missing.append(
            "research_allow_oversize_artifact is reserved for the two "
            "retained expanded research recipes"
        )
        allow_oversize = False
    missing.extend(
        _artifact_contract_failures(metrics, allow_oversize=allow_oversize)
    )
    configured_cap = run.get("artifact_cap_bytes")
    if configured_cap != ARTIFACT_CAP_BYTES:
        missing.append(
            f"artifact_cap_bytes must be exactly {ARTIFACT_CAP_BYTES}; "
            f"configured={configured_cap!r}"
        )
    expected_params = run.get("expected_model_params")
    if expected_params is not None and metrics.get("model_params") != expected_params:
        missing.append(
            f"expected_model_params={expected_params}; "
            f"observed={metrics.get('model_params')!r}"
        )
    segment_end_step = int(protocol.get("segment_end_step", protocol["iterations"]))
    observed_validation_steps = [item["step"] for item in metrics["checkpoints"]]
    if not observed_validation_steps or observed_validation_steps[-1] != segment_end_step:
        missing.append(
            f"segment must finish with validation at step {segment_end_step}; "
            f"observed={observed_validation_steps}"
        )
    forbidden_early = [
        step
        for step in observed_validation_steps
        if step < int(protocol.get("val_start_step", 0)) and step != segment_end_step
    ]
    if forbidden_early:
        missing.append(
            f"validation occurred before val_start_step: {forbidden_early}"
        )
    # The compressed model is a benchmark artifact, not a source-repository result.
    artifact_candidates = (
        checkout / "final_model.int8.ptz",
        checkout / "ours" / "final_model.int8.ptz",
    )
    artifact_present = any(candidate.is_file() for candidate in artifact_candidates)
    metrics["artifact_file_present_before_delete"] = artifact_present
    if not artifact_present:
        missing.append("final_model.int8.ptz was not created")
    for candidate in artifact_candidates:
        candidate.unlink(missing_ok=True)

    checkpoint_metadata: dict[str, Any] | None = None
    checkpoint_matches = re.findall(CHECKPOINT_PATTERN, text, flags=re.MULTILINE)
    if checkpoint_required:
        if checkpoint_output is None or not checkpoint_output.is_file():
            missing.append(f"{CHECKPOINT_FILENAME} was not created")
        elif not checkpoint_matches:
            missing.append("missing checkpoint_saved metadata in log")
        else:
            saved_step, saved_target, logged_bytes, logged_sha256 = checkpoint_matches[-1]
            actual_bytes = checkpoint_output.stat().st_size
            actual_sha256 = _sha256(checkpoint_output)
            checkpoint_metadata = {
                "schema_version": 1,
                "filename": CHECKPOINT_FILENAME,
                "step": int(saved_step),
                "target_iterations": int(saved_target),
                "grad_accum_steps": _resolved_grad_accum_steps(protocol),
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "run_id": run_id,
                "runtime_seed": seed,
            }
            if int(saved_step) != segment_end_step:
                missing.append(
                    f"checkpoint step must be {segment_end_step}, got {saved_step}"
                )
            if int(saved_target) != int(protocol["iterations"]):
                missing.append(
                    f"checkpoint target must be {protocol['iterations']}, got {saved_target}"
                )
            if int(logged_bytes) != actual_bytes or logged_sha256 != actual_sha256:
                missing.append("checkpoint log hash/size does not match checkpoint file")
            exported_checkpoint = export_dir / CHECKPOINT_FILENAME
            exported_checkpoint.unlink(missing_ok=True)
            shutil.move(str(checkpoint_output), str(exported_checkpoint))
            _write_json(export_dir / CHECKPOINT_METADATA_FILENAME, checkpoint_metadata)
            metrics["research_checkpoint"] = checkpoint_metadata

    passed = return_code == 0 and not missing and error is None
    elapsed = time.monotonic() - started

    row = {
        "id": run_id,
        "stage": stage,
        "status": "passed" if passed else "failed",
        "arm": run["arm"],
        "recipe": run.get("recipe"),
        "rope": run["rope"],
        "backend": run["backend"],
        "projection_layout": run.get("projection_layout", "mamba-shared"),
        "optimizer_policy": run["optimizer_policy"],
        "return_code": return_code,
        "elapsed_seconds": round(elapsed, 3),
        "assertions": {"passed": not missing, "missing_patterns": missing},
        "metrics": metrics,
        "log": log_name,
        "error": error,
    }
    status.update(
        state=row["status"], elapsed_seconds=round(elapsed, 1), updated_at=_utc_now()
    )
    _write_json(export_dir / "status.json", status)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--benchmark-runner", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--git-revision")
    parser.add_argument("--git-dirty", choices=("true", "false", "unknown"))
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint-sha256", default="")
    parser.add_argument("--resume-checkpoint-bytes", type=int, default=0)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    export_dir = args.export_dir.resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    checkout = args.checkout.resolve()
    manifest_path = checkout / "ours" / "ADAPTER-MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    (export_dir / "suite-config.json").write_bytes(args.config.read_bytes())
    (export_dir / "adapter-manifest.json").write_bytes(manifest_path.read_bytes())

    started_at = _utc_now()
    result: dict[str, Any] = {
        "schema_version": 1,
        "suite_id": config["suite_id"],
        "status": "failed",
        "runtime_seed": args.seed,
        "started_at": started_at,
        "finished_at": started_at,
        "source_snapshot": {
            "payload_sha256": args.source_sha256,
            "git_revision": args.git_revision,
            "git_dirty": (
                None if args.git_dirty == "unknown" else args.git_dirty == "true"
            ),
        },
        "adapter_manifest_sha256": _sha256(manifest_path),
        "protocols": config["protocols"],
        "runs": [],
        "error": None,
    }
    status: dict[str, Any] = {
        "suite_id": config["suite_id"],
        "state": "starting",
        "updated_at": _utc_now(),
    }
    _write_json(export_dir / "status.json", status)
    _write_json(export_dir / "suite-result.json", result)

    active_resume = None
    active_resume_sha256 = args.resume_checkpoint_sha256
    active_resume_bytes = args.resume_checkpoint_bytes
    try:
        smoke_protocol = config["protocols"]["smoke"]
        production_protocol = config["protocols"]["production"]
        if int(production_protocol.get("segment_start_step", 0)) > 0:
            smoke_grad_accum_steps = _resolved_grad_accum_steps(smoke_protocol)
            production_grad_accum_steps = _resolved_grad_accum_steps(
                production_protocol
            )
            if smoke_grad_accum_steps != production_grad_accum_steps:
                raise RuntimeError(
                    "checkpoint continuation requires identical effective "
                    "grad_accum_steps across smoke and production: "
                    f"smoke={smoke_grad_accum_steps}, "
                    f"production={production_grad_accum_steps}"
                )
        if args.resume_checkpoint is not None:
            if int(config["protocols"]["smoke"].get("segment_start_step", 0)) <= 0:
                raise RuntimeError(
                    "external resume requires a positive smoke segment_start_step"
                )
            if not config["protocols"]["smoke"].get("checkpoint_at_end", False):
                raise RuntimeError(
                    "external resume requires smoke checkpoint_at_end=true so "
                    "production cannot reuse the original external checkpoint"
                )
            active_resume = _verify_resume_checkpoint(
                args.resume_checkpoint,
                active_resume_sha256,
                active_resume_bytes,
            )
        elif active_resume_sha256 or active_resume_bytes:
            raise RuntimeError(
                "resume checkpoint hash/size was supplied without a checkpoint path"
            )
        for stage in ("smoke", "production"):
            protocol = config["protocols"][stage]
            stage_start = int(protocol.get("segment_start_step", 0))
            if stage_start > 0 and active_resume is None:
                raise RuntimeError(
                    f"{stage} segment starting at {stage_start} requires a checkpoint"
                )
            stage_resume = active_resume if stage_start > 0 else None
            for run in config["runs"]:
                row = _run_one(
                    stage=stage,
                    run=run,
                    protocol=protocol,
                    benchmark_runner=args.benchmark_runner.resolve(),
                    checkout=checkout,
                    export_dir=export_dir,
                    seed=args.seed,
                    status=status,
                    expected_train_shards=int(config["data"]["train_shards"]),
                    resume_checkpoint=stage_resume,
                    resume_checkpoint_sha256=(
                        active_resume_sha256 if stage_resume is not None else ""
                    ),
                    resume_checkpoint_bytes=(
                        active_resume_bytes if stage_resume is not None else 0
                    ),
                )
                result["runs"].append(row)
                result["finished_at"] = _utc_now()
                _write_json(export_dir / "suite-result.json", result)
                if row["status"] != "passed":
                    raise RuntimeError(f"{stage}/{run['id']} failed")
                if protocol.get("checkpoint_at_end", False):
                    metadata_path = export_dir / CHECKPOINT_METADATA_FILENAME
                    checkpoint_path = export_dir / CHECKPOINT_FILENAME
                    if not metadata_path.is_file() or not checkpoint_path.is_file():
                        raise RuntimeError(f"{stage} did not export its checkpoint")
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    expected_step = int(
                        protocol.get("segment_end_step", protocol["iterations"])
                    )
                    if int(metadata.get("step", -1)) != expected_step:
                        raise RuntimeError(
                            f"{stage} checkpoint step mismatch: "
                            f"checkpoint={metadata.get('step')}, expected={expected_step}"
                        )
                    expected_grad_accum_steps = _resolved_grad_accum_steps(protocol)
                    if metadata.get("grad_accum_steps") != expected_grad_accum_steps:
                        raise RuntimeError(
                            f"{stage} checkpoint grad_accum_steps mismatch: "
                            f"checkpoint={metadata.get('grad_accum_steps')}, "
                            f"expected={expected_grad_accum_steps}"
                        )
                    active_resume_sha256 = str(metadata.get("sha256", ""))
                    active_resume_bytes = int(metadata.get("bytes", 0))
                    active_resume = _verify_resume_checkpoint(
                        checkpoint_path,
                        active_resume_sha256,
                        active_resume_bytes,
                    )
        result["status"] = "passed"
        status.update(state="passed", updated_at=_utc_now())
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        status.update(state="failed", error=result["error"], updated_at=_utc_now())
    finally:
        result["finished_at"] = _utc_now()
        _write_json(export_dir / "suite-result.json", result)
        _write_json(export_dir / "status.json", status)
        (export_dir / "done.flag").write_text(
            f"{result['status']} {result['finished_at']}\n", encoding="utf-8"
        )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
