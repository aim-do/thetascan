"""Safely run one public Parameter Golf arm on one RunPod H100 SXM.

The launcher deliberately accepts exactly one run per configuration.  The pod
performs that arm's smoke test first and ``remote_suite.py`` starts production
only when the smoke assertions pass.

Safety is three-layered:

* attached launches terminate the pod from a local ``finally`` block;
* pod creation declares RunPod ``terminateAfter`` before any SSH bootstrap;
* after bootstrap, an independent pod-side watchdog and normal-completion path
  also delete the pod.

The source snapshot and startup script are transferred over account-key SSH.
When RunPod's default private-key file is unreadable, the launcher also checks
the project's conventional ``~/.ssh/thetascan_runpod_ed25519`` key before it
considers the legacy control-plane fallback. Only
``/workspace/thetascan-export`` is exposed through the RunPod HTTP proxy.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import http.client
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 benchmark tooling only.
    import tomli as tomllib


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
GRAPHQL_URL = "https://api.runpod.io/graphql"
# The historical public template y5cejece4j now returns 404 from RunPod's
# template API.  Address its underlying Parameter Golf image by the historical
# source tag instead of depending on that vanished control-plane object. RunPod
# accepts an image name here, not an OCI digest; the separately recorded index
# digest is expected provenance metadata, not platform-attested runtime state.
TEMPLATE_ID: str | None = None
IMAGE_NAME = "runpod/parameter-golf:5e377cdd76814bd8d13488af05a237795957be13"
IMAGE_INDEX_DIGEST = "sha256:74af8ca6ea79dde333038cd824055fc1fbac06b9d3b69360222d3353c5482d1f"
GPU_TYPE_ID = "NVIDIA H100 80GB HBM3"
ARTIFACT_CAP_BYTES = 16_000_000
DEFAULT_GRAD_ACCUM_STEPS = 8
THETASCAN_RECIPES = frozenset(
    {
        "gn-reference-v0.1",
        "gn-expanded-reference-v0.1",
        "kernel-expanded-reference-v0.1",
    }
)
DEFAULT_RATE_USD_PER_HOUR = {"SECURE": 2.99, "COMMUNITY": 2.69}
# RunPod's Cloudflare proxy rejects some non-browser Python user agents with
# HTTP 403 even though the same public endpoint is reachable with curl.
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
PAYLOAD_CHUNK_SIZE = 3500
SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
MAX_CONSECUTIVE_CONTROL_PLANE_ERRORS = 5
REMOTE_RESUME_CHECKPOINT = "/tmp/thetascan-resume-checkpoint.pt"
SSH_CHECKPOINT_UPLOAD_TIMEOUT_SECONDS = 900
# RunPod's GraphQL edge currently returns HTTP 500 once a request grows past
# roughly 100 KiB. Keep the legacy environment transport fail-closed below
# that observed boundary; normal SSH creation requests are tiny.
MAX_CONTROL_PLANE_BOOTSTRAP_REQUEST_BYTES = 100_000
DEFAULT_HARD_TIMEOUT_SECONDS = 3900
DEFAULT_RESULT_GRACE_SECONDS = 300
DEFAULT_MAX_BUDGET_USD = 3.25


class LauncherError(RuntimeError):
    """A fail-closed launcher error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LauncherError(f"expected a JSON object in {path}")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_grad_accum_steps(protocol: dict[str, Any]) -> int:
    configured = protocol.get("grad_accum_steps", 0)
    if not isinstance(configured, int) or isinstance(configured, bool):
        raise LauncherError("protocol grad_accum_steps must be an integer")
    if configured < 0:
        raise LauncherError("protocol grad_accum_steps must be nonnegative")
    # Public RunPod suites are fixed to one process, where the pinned harness
    # default 8 // world_size resolves to eight accumulation micro-steps.
    return configured or DEFAULT_GRAD_ACCUM_STEPS


def _validate_resume_checkpoint(
    checkpoint_path: Path,
    *,
    config: dict[str, Any],
    run: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Validate an exported checkpoint before any billable API operation."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    metadata_path = checkpoint_path.parent / "research-checkpoint.json"
    if not checkpoint_path.is_file():
        raise LauncherError(f"resume checkpoint not found: {checkpoint_path}")
    if not metadata_path.is_file():
        raise LauncherError(
            "resume checkpoint requires companion research-checkpoint.json: "
            f"{metadata_path}"
        )
    try:
        metadata = _read_json(metadata_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LauncherError(
            f"could not read resume checkpoint metadata {metadata_path}: {exc}"
        ) from exc
    if metadata.get("schema_version") != 1:
        raise LauncherError("resume checkpoint metadata schema_version must be 1")
    expected_bytes = metadata.get("bytes")
    expected_sha256 = metadata.get("sha256")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise LauncherError("resume checkpoint metadata bytes must be a positive int")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise LauncherError("resume checkpoint metadata sha256 must be lowercase SHA-256")
    try:
        actual_bytes = checkpoint_path.stat().st_size
        actual_sha256 = _file_sha256(checkpoint_path)
    except OSError as exc:
        raise LauncherError(
            f"could not read resume checkpoint {checkpoint_path}: {exc}"
        ) from exc
    if actual_bytes != expected_bytes:
        raise LauncherError(
            "resume checkpoint size mismatch: "
            f"metadata={expected_bytes}, actual={actual_bytes}"
        )
    if actual_sha256 != expected_sha256:
        raise LauncherError(
            "resume checkpoint SHA-256 mismatch: "
            f"metadata={expected_sha256}, actual={actual_sha256}"
        )

    smoke = config["protocols"]["smoke"]
    smoke_start = int(smoke.get("segment_start_step", 0))
    target_iterations = int(smoke["iterations"])
    if smoke_start <= 0:
        raise LauncherError(
            "--resume-checkpoint requires protocols.smoke.segment_start_step > 0"
        )
    contracts = {
        "step": smoke_start,
        "target_iterations": target_iterations,
        "grad_accum_steps": _resolved_grad_accum_steps(smoke),
        "runtime_seed": seed,
        "run_id": run["id"],
    }
    for key, expected in contracts.items():
        if metadata.get(key) != expected:
            raise LauncherError(
                f"resume checkpoint {key} does not match launch contract: "
                f"metadata={metadata.get(key)!r}, expected={expected!r}"
            )
    if not smoke.get("checkpoint_at_end", False):
        raise LauncherError(
            "an externally resumed smoke segment must set checkpoint_at_end=true "
            "so production resumes the verified smoke output"
        )
    return {
        "path": checkpoint_path,
        "metadata_path": metadata_path.resolve(),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "step": smoke_start,
        "target_iterations": target_iterations,
        "grad_accum_steps": _resolved_grad_accum_steps(smoke),
        "runtime_seed": seed,
        "run_id": run["id"],
    }


def _validate_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read_json(path)
    if config.get("schema_version") != 1:
        raise LauncherError("suite config schema_version must be 1")
    suite_id = config.get("suite_id")
    if not isinstance(suite_id, str) or not SAFE_RUN_ID.fullmatch(suite_id):
        raise LauncherError(f"unsafe suite_id {suite_id!r}")
    runs = config.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        raise LauncherError(
            "one config must contain exactly one run; one config is deployed per pod"
        )
    run = runs[0]
    if not isinstance(run, dict):
        raise LauncherError("runs[0] must be an object")
    run_id = run.get("id")
    if not isinstance(run_id, str) or not SAFE_RUN_ID.fullmatch(run_id):
        raise LauncherError(f"unsafe run id {run_id!r}")
    arm = run.get("arm")
    if arm not in {"attention", "mamba3", "thetascan"}:
        raise LauncherError(f"unsupported arm {arm!r}")
    recipe = run.get("recipe")
    if arm == "thetascan":
        if recipe not in THETASCAN_RECIPES:
            raise LauncherError(
                f"{run_id}: ThetaScan recipe must be one of "
                f"{sorted(THETASCAN_RECIPES)!r}; got {recipe!r}"
            )
    elif recipe is not None:
        raise LauncherError(
            f"{run_id}: external arm {arm!r} must not define a ThetaScan recipe"
        )
    expected_params = run.get("expected_model_params")
    if not isinstance(expected_params, int) or expected_params <= 0:
        raise LauncherError(f"{run_id}: expected_model_params must be a positive int")
    if run.get("artifact_cap_bytes") != ARTIFACT_CAP_BYTES:
        raise LauncherError(
            f"{run_id}: artifact_cap_bytes must be exactly {ARTIFACT_CAP_BYTES}"
        )

    hardware = config.get("hardware")
    if not isinstance(hardware, dict):
        raise LauncherError("hardware must be an object")
    if hardware.get("gpu_count") != 1 or hardware.get("gpu_type") != GPU_TYPE_ID:
        raise LauncherError(
            f"hardware must be exactly 1 x {GPU_TYPE_ID}; got {hardware!r}"
        )
    data = config.get("data")
    if not isinstance(data, dict):
        raise LauncherError("data must be an object")
    if data.get("variant") != "sp1024":
        raise LauncherError("data.variant must be exactly 'sp1024'")
    train_shards = data.get("train_shards")
    if not isinstance(train_shards, int) or not 1 <= train_shards <= 100:
        raise LauncherError("data.train_shards must be an integer between 1 and 100")
    launcher = config.get("launcher", {})
    if not isinstance(launcher, dict):
        raise LauncherError("launcher must be an object when present")
    unknown_launcher_keys = set(launcher) - {
        "hard_timeout_seconds",
        "result_grace_seconds",
        "max_budget_usd",
    }
    if unknown_launcher_keys:
        raise LauncherError(
            "unsupported launcher field(s): "
            + ", ".join(sorted(unknown_launcher_keys))
        )
    _validate_launch_guards(
        launcher.get("hard_timeout_seconds", DEFAULT_HARD_TIMEOUT_SECONDS),
        launcher.get("result_grace_seconds", DEFAULT_RESULT_GRACE_SECONDS),
        launcher.get("max_budget_usd", DEFAULT_MAX_BUDGET_USD),
        source="launcher config",
    )
    protocols = config.get("protocols")
    if not isinstance(protocols, dict):
        raise LauncherError("protocols must be an object")
    for stage in ("smoke", "production"):
        protocol = protocols.get(stage)
        if not isinstance(protocol, dict):
            raise LauncherError(f"missing protocols.{stage}")
        for key in ("iterations", "val_every", "timeout_seconds"):
            value = protocol.get(key)
            if not isinstance(value, (int, float)) or value <= 0:
                raise LauncherError(f"protocols.{stage}.{key} must be positive")
        max_wallclock = protocol.get("max_wallclock_seconds", 0)
        if not isinstance(max_wallclock, (int, float)) or max_wallclock < 0:
            raise LauncherError(
                f"protocols.{stage}.max_wallclock_seconds must be nonnegative"
            )
        segment_start = protocol.get("segment_start_step", 0)
        segment_end = protocol.get("segment_end_step", protocol["iterations"])
        val_start = protocol.get("val_start_step", 0)
        if not all(isinstance(value, int) for value in (segment_start, segment_end, val_start)):
            raise LauncherError(f"protocols.{stage} segment/validation steps must be ints")
        if not 0 <= segment_start < segment_end <= protocol["iterations"]:
            raise LauncherError(
                f"protocols.{stage} requires 0 <= segment_start < segment_end <= iterations"
            )
        if val_start < 0:
            raise LauncherError(f"protocols.{stage}.val_start_step must be nonnegative")
        if not isinstance(protocol.get("checkpoint_at_end", False), bool):
            raise LauncherError(f"protocols.{stage}.checkpoint_at_end must be boolean")
        allow_oversize = protocol.get("research_allow_oversize_artifact", False)
        if type(allow_oversize) is not bool:
            raise LauncherError(
                f"protocols.{stage}.research_allow_oversize_artifact must be boolean"
            )
        if allow_oversize and recipe not in {
            "gn-expanded-reference-v0.1",
            "kernel-expanded-reference-v0.1",
        }:
            raise LauncherError(
                f"protocols.{stage}.research_allow_oversize_artifact is reserved "
                "for the two retained expanded research recipes"
            )
        _resolved_grad_accum_steps(protocol)
        if protocol["max_wallclock_seconds"] > protocol["timeout_seconds"]:
            raise LauncherError(
                f"protocols.{stage}.max_wallclock_seconds exceeds its process timeout"
            )
    smoke = protocols["smoke"]
    production = protocols["production"]
    production_start = int(production.get("segment_start_step", 0))
    if production_start > 0:
        smoke_end = int(smoke.get("segment_end_step", smoke["iterations"]))
        if (
            not smoke.get("checkpoint_at_end", False)
            or smoke_end != production_start
            or smoke["iterations"] != production["iterations"]
            or smoke.get("warmdown_iters") != production.get("warmdown_iters")
            or _resolved_grad_accum_steps(smoke)
            != _resolved_grad_accum_steps(production)
        ):
            raise LauncherError(
                "a nonzero production start must continue the smoke checkpoint "
                "with the same target, warmdown schedule, and effective "
                "grad_accum_steps"
            )
    return config, run


def _validate_launch_guards(
    hard_timeout_seconds: object,
    result_grace_seconds: object,
    max_budget_usd: object,
    *,
    source: str,
) -> tuple[int, int, float]:
    if type(hard_timeout_seconds) is not int or not 300 <= hard_timeout_seconds <= 14_400:
        raise LauncherError(
            f"{source} hard_timeout_seconds must be an integer between 300 and 14400"
        )
    if type(result_grace_seconds) is not int or not 0 <= result_grace_seconds <= 3600:
        raise LauncherError(
            f"{source} result_grace_seconds must be an integer between 0 and 3600"
        )
    if result_grace_seconds >= hard_timeout_seconds:
        raise LauncherError(
            f"{source} result grace must be shorter than the absolute hard timeout"
        )
    if (
        isinstance(max_budget_usd, bool)
        or not isinstance(max_budget_usd, (int, float))
        or not math.isfinite(float(max_budget_usd))
        or not 0 < float(max_budget_usd) <= 100
    ):
        raise LauncherError(
            f"{source} max_budget_usd must be a finite number in (0, 100]"
        )
    return hard_timeout_seconds, result_grace_seconds, float(max_budget_usd)


def _resolve_launch_guards(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[int, int, float]:
    """Resolve CLI overrides over per-config guards over conservative defaults."""
    launcher = config.get("launcher", {})
    hard_timeout_seconds = (
        args.hard_timeout_seconds
        if args.hard_timeout_seconds is not None
        else launcher.get("hard_timeout_seconds", DEFAULT_HARD_TIMEOUT_SECONDS)
    )
    result_grace_seconds = (
        args.result_grace_seconds
        if args.result_grace_seconds is not None
        else launcher.get("result_grace_seconds", DEFAULT_RESULT_GRACE_SECONDS)
    )
    max_budget_usd = (
        args.max_budget_usd
        if args.max_budget_usd is not None
        else launcher.get("max_budget_usd", DEFAULT_MAX_BUDGET_USD)
    )
    return _validate_launch_guards(
        hard_timeout_seconds,
        result_grace_seconds,
        max_budget_usd,
        source="effective launch guard",
    )


def _git_identity() -> tuple[str | None, str]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        )
        return revision, "true" if dirty else "false"
    except (OSError, subprocess.CalledProcessError):
        return None, "unknown"


def _payload_files() -> list[Path]:
    package = PROJECT_ROOT / "src" / "thetascan"
    files = [
        path
        for path in package.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    ]
    for name in (
        "prepare_harness.py",
        "run_benchmark.py",
        "remote_suite.py",
        "thetascan_benchmark_adapter.py",
        "adapter_manifest.json",
    ):
        files.append(HERE / name)
    for name in (
        "LICENSE",
        "NOTICE",
        "PATENTS.md",
        "LICENSING.md",
        "THIRD_PARTY_NOTICES.md",
    ):
        files.append(PROJECT_ROOT / name)
    files.extend(path for path in (PROJECT_ROOT / "licenses").rglob("*") if path.is_file())
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise LauncherError("payload inputs are missing: " + ", ".join(missing))
    return sorted(set(files), key=lambda item: item.relative_to(PROJECT_ROOT).as_posix())


def _add_tar_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def _build_payload(config_path: Path) -> bytes:
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w") as archive:
        for path in _payload_files():
            arcname = path.relative_to(PROJECT_ROOT).as_posix()
            _add_tar_bytes(archive, arcname, path.read_bytes())
        _add_tar_bytes(archive, "suite-config.json", config_path.read_bytes())
    return gzip.compress(raw_tar.getvalue(), compresslevel=9, mtime=0)


def _mamba_revision() -> str:
    manifest = _read_json(HERE / "adapter_manifest.json")
    reference = manifest.get("mamba3_reference")
    if not isinstance(reference, dict):
        raise LauncherError("adapter_manifest.json lacks mamba3_reference")
    revision = reference.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise LauncherError(f"invalid pinned Mamba revision {revision!r}")
    return revision


def _startup_script(
    *,
    run: dict[str, Any],
    data: dict[str, Any],
    seed: int,
    source_sha256: str,
    git_revision: str | None,
    git_dirty: str,
    hard_timeout_seconds: int,
    result_grace_seconds: int,
    resume_checkpoint_sha256: str | None = None,
    resume_checkpoint_bytes: int = 0,
) -> str:
    mamba_stage = ""
    if run["arm"] == "mamba3":
        revision = _mamba_revision()
        mamba_stage = f"""
write_status mamba_setup
git clone --quiet --filter=blob:none https://github.com/state-spaces/mamba.git "$MAMBA_ROOT"
git -C "$MAMBA_ROOT" checkout --quiet {revision}
: > "$MAMBA_ROOT/mamba_ssm/__init__.py"
: > "$MAMBA_ROOT/mamba_ssm/modules/__init__.py"
PYTHONPATH="$MAMBA_ROOT" python3 -c 'from mamba_ssm.modules.mamba3 import Mamba3; print("mamba3_probe:ok")'
"""
    git_args = ""
    if git_revision:
        git_args += f" --git-revision {git_revision}"
    git_args += f" --git-dirty {git_dirty}"
    resume_args = ""
    if resume_checkpoint_sha256 is not None:
        resume_args = (
            f" \\\n  --resume-checkpoint {REMOTE_RESUME_CHECKPOINT}"
            f" \\\n  --resume-checkpoint-sha256 {resume_checkpoint_sha256}"
            f" \\\n  --resume-checkpoint-bytes {resume_checkpoint_bytes}"
        )

    # The HTTP server is rooted only at EXPORT_ROOT.  The payload and generated
    # checkout can therefore never be requested through the public proxy.
    train_shards = int(data["train_shards"])
    variant = str(data["variant"])
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
SOURCE_ROOT=/workspace/thetascan-source
CHECKOUT=/workspace/parameter-golf-checkout
MAMBA_ROOT=/workspace/mamba-source
EXPORT_ROOT=/workspace/thetascan-export
RESULT_GRACE_SECONDS={result_grace_seconds}
HARD_TIMEOUT_SECONDS={hard_timeout_seconds}
# The Xet CAS path has repeatedly stalled at an exact 200,001,024-byte partial
# blob on otherwise healthy RunPod hosts. Force ordinary Hugging Face HTTP
# downloads before the library's first import, with bounded network waits.
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
mkdir -p "$SOURCE_ROOT" "$EXPORT_ROOT"
write_status() {{
  local state="$1"
  printf '{{"state":"%s","updated_at":"%s"}}\n' "$state" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$EXPORT_ROOT/status.json.tmp"
  mv "$EXPORT_ROOT/status.json.tmp" "$EXPORT_ROOT/status.json"
}}

HTTP_PID=""
WATCHDOG_PID=""
self_delete() {{
  # Both values are injected by RunPod. Python keeps the pod-scoped bearer
  # token out of the process command line and out of all exported logs.
  python3 - <<'PY'
import os, time, urllib.error, urllib.request
pod_id = os.environ.get("RUNPOD_POD_ID", "")
api_key = os.environ.get("RUNPOD_API_KEY", "")
if not pod_id or not api_key:
    raise SystemExit(2)
url = f"https://rest.runpod.io/v1/pods/{{pod_id}}"
delay = 0
while True:
    if delay:
        time.sleep(delay)
    request = urllib.request.Request(
        url,
        headers={{
            "Authorization": f"Bearer {{api_key}}",
            "User-Agent": "ThetaScan-Pod-Watchdog/1.0",
        }},
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status in (200, 202, 204):
                raise SystemExit(0)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise SystemExit(0)
    except urllib.error.URLError:
        pass
    delay = min(30, 2 if delay == 0 else delay * 2)
PY
}}

absolute_watchdog() {{
  sleep "$HARD_TIMEOUT_SECONDS"
  self_delete
}}
absolute_watchdog &
WATCHDOG_PID=$!

finish() {{
  local rc=$?
  trap - EXIT TERM INT
  set +e
  if [[ ! -s "$EXPORT_ROOT/done.flag" ]]; then
    write_status setup_failed
    printf 'failed %s rc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" > "$EXPORT_ROOT/done.flag"
  fi
  sleep "$RESULT_GRACE_SECONDS"
  # Normal completion/failure permanently deletes the Pod and its volume. If
  # the immediate retries fail, preserve and wait for the independent watchdog.
  if ! self_delete; then
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
  if [[ -n "$HTTP_PID" ]]; then
    kill "$HTTP_PID" 2>/dev/null || true
    wait "$HTTP_PID" 2>/dev/null || true
  fi
  exit "$rc"
}}
trap finish EXIT
trap 'exit 124' TERM INT

write_status booting
python3 -m http.server 8000 --bind 0.0.0.0 --directory "$EXPORT_ROOT" > /tmp/thetascan-http.log 2>&1 &
HTTP_PID=$!
exec > >(tee "$EXPORT_ROOT/pod.log") 2>&1

write_status extracting_payload
python3 - <<'PY'
import base64, io, os, tarfile
payload_path = "/tmp/thetascan-payload.tar.gz"
if os.path.isfile(payload_path):
    with tarfile.open(payload_path, mode="r:gz") as archive:
        archive.extractall("/workspace/thetascan-source", filter="data")
else:
    chunks = [
        value for key, value in sorted(os.environ.items())
        if key.startswith("TS_PAYLOAD_")
    ]
    if not chunks:
        raise RuntimeError("missing SSH payload file and TS_PAYLOAD_* fallback")
    payload = base64.b64decode("".join(chunks), validate=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall("/workspace/thetascan-source", filter="data")
PY
unset $(compgen -v | grep '^TS_PAYLOAD_' || true)

export PYTHONPATH="$SOURCE_ROOT/src${{PYTHONPATH:+:$PYTHONPATH}}"
write_status preparing_harness
python3 "$SOURCE_ROOT/benchmarks/parameter-golf/prepare_harness.py" "$CHECKOUT"

write_status verifying_python_dependencies
python3 - <<'PY'
import importlib, importlib.metadata

required = {{
    "huggingface-hub": "huggingface_hub",
    "sentencepiece": "sentencepiece",
    "numpy": "numpy",
    "einops": "einops",
}}
for distribution, module in required.items():
    importlib.import_module(module)
    print(f"runtime_dependency:{{distribution}}={{importlib.metadata.version(distribution)}}")
PY

write_status recording_runtime
python3 - <<'PY'
import importlib.metadata, json, platform, subprocess, sys
from pathlib import Path
import torch
try:
    import triton
    triton_version = triton.__version__
except Exception as exc:
    triton_version = f"unavailable:{{type(exc).__name__}}"
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    text=True, capture_output=True, check=False,
).stdout.strip().splitlines()
value = {{
    "template_id": {TEMPLATE_ID!r},
    "image_name": "{IMAGE_NAME}",
    "image_index_digest": "{IMAGE_INDEX_DIGEST}",
    "required_gpu_type": "{GPU_TYPE_ID}",
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "triton": triton_version,
    "python_dependencies": {{
        name: importlib.metadata.version(name)
        for name in ("huggingface-hub", "sentencepiece", "numpy", "einops")
    }},
    "gpus": gpu,
}}
Path("/workspace/thetascan-export/runtime.json").write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)
if not value["cuda_available"] or value["cuda_device_count"] != 1:
    raise RuntimeError(f"expected exactly one CUDA GPU, got {{value}}")
if not gpu or not all("H100 80GB HBM3" in item for item in gpu):
    raise RuntimeError(f"wrong GPU: {{gpu}}")
PY

write_status downloading_data
python3 "$CHECKOUT/ours/download_data.py" --variant {variant} --train-shards {train_shards}
{mamba_stage}
export PYTHONPATH="$SOURCE_ROOT/src${{MAMBA_ROOT:+:$MAMBA_ROOT}}${{PYTHONPATH:+:$PYTHONPATH}}"
write_status starting_suite
set +e
python3 "$SOURCE_ROOT/benchmarks/parameter-golf/remote_suite.py" \
  --config "$SOURCE_ROOT/suite-config.json" \
  --checkout "$CHECKOUT" \
  --benchmark-runner "$SOURCE_ROOT/benchmarks/parameter-golf/run_benchmark.py" \
  --export-dir "$EXPORT_ROOT" \
  --seed {seed} \
  --source-sha256 {source_sha256}{git_args}{resume_args}
SUITE_RC=$?
set -e
exit "$SUITE_RC"
"""


def _pod_environment(payload: bytes) -> list[dict[str, str]]:
    encoded = base64.b64encode(payload).decode("ascii")
    result = [
        {"key": f"TS_PAYLOAD_{index:03d}", "value": encoded[offset : offset + PAYLOAD_CHUNK_SIZE]}
        for index, offset in enumerate(range(0, len(encoded), PAYLOAD_CHUNK_SIZE))
    ]
    if len(result) > 45:
        raise LauncherError(
            f"payload needs {len(result)} environment variables; RunPod allows 50"
        )
    return result


def _docker_command(startup_script: str) -> str:
    encoded = base64.b64encode(
        gzip.compress(startup_script.encode("utf-8"), compresslevel=9, mtime=0)
    ).decode("ascii")
    return (
        f"bash -lc 'printf %s {encoded} | base64 -d | gunzip > /tmp/thetascan-run.sh "
        "&& chmod 700 /tmp/thetascan-run.sh && exec bash /tmp/thetascan-run.sh'"
    )


def _load_api_key() -> str:
    config_path = Path.home() / ".runpod" / "config.toml"
    if config_path.is_file():
        value = tomllib.loads(config_path.read_text(encoding="utf-8")).get("apikey")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = os.environ.get("RUNPOD_API_KEY", "").strip()
    if value:
        return value
    raise LauncherError(
        f"RunPod auth not found in {config_path}; run runpodctl doctor first"
    )


def _gql_request_body(query: str, variables: dict[str, Any] | None = None) -> bytes:
    return json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")


def _gql(api_key: str, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        GRAPHQL_URL,
        data=_gql_request_body(query, variables),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise LauncherError(
            f"RunPod GraphQL request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LauncherError(f"RunPod GraphQL request failed: {exc}") from exc
    if value.get("errors"):
        raise LauncherError("RunPod GraphQL error: " + json.dumps(value["errors"]))
    data = value.get("data")
    if not isinstance(data, dict):
        raise LauncherError("RunPod GraphQL response has no data object")
    return data


def _account_and_rate(api_key: str, cloud_type: str) -> tuple[float, float]:
    data = _gql(
        api_key,
        "query { myself { clientBalance } gpuTypes { id securePrice communityPrice } }",
    )
    balance = float(data["myself"]["clientBalance"])
    field = "securePrice" if cloud_type == "SECURE" else "communityPrice"
    rate = next(
        (float(item[field]) for item in data["gpuTypes"] if item["id"] == GPU_TYPE_ID),
        DEFAULT_RATE_USD_PER_HOUR[cloud_type],
    )
    return balance, rate


def _create_pod(
    api_key: str,
    *,
    name: str,
    cloud_type: str,
    terminate_after: str,
    bootstrap_transport: str = "ssh",
    docker_command: str | None = None,
    environment: list[dict[str, str]] | None = None,
) -> str:
    mutation = """
    mutation podFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) { id desiredStatus }
    }
    """
    pod_input = {
        "name": name,
        "gpuCount": 1,
        "gpuTypeId": GPU_TYPE_ID,
        "imageName": IMAGE_NAME,
        "cloudType": cloud_type,
        "containerDiskInGb": 50,
        "volumeInGb": 100,
        "volumeMountPath": "/workspace",
        "ports": "22/tcp,8000/http",
        "startSsh": True,
        "terminateAfter": terminate_after,
    }
    if bootstrap_transport == "environment":
        if docker_command is None or environment is None:
            raise LauncherError(
                "environment bootstrap requires both docker_command and environment"
            )
        pod_input["dockerArgs"] = docker_command
        pod_input["env"] = environment
    elif bootstrap_transport != "ssh":
        raise LauncherError(f"unsupported bootstrap transport {bootstrap_transport!r}")
    variables = {"input": pod_input}
    request_bytes = len(_gql_request_body(mutation, variables))
    if (
        bootstrap_transport == "environment"
        and request_bytes > MAX_CONTROL_PLANE_BOOTSTRAP_REQUEST_BYTES
    ):
        raise LauncherError(
            "legacy environment bootstrap request is too large for RunPod GraphQL "
            f"({request_bytes} > {MAX_CONTROL_PLANE_BOOTSTRAP_REQUEST_BYTES} bytes); "
            "use auto or ssh bootstrap"
        )
    result = _gql(api_key, mutation, variables).get(
        "podFindAndDeployOnDemand"
    )
    if not isinstance(result, dict) or not result.get("id"):
        raise LauncherError(f"RunPod did not create a pod: {result!r}")
    return str(result["id"])


def _ssh_key_candidates() -> tuple[Path, ...]:
    override = os.environ.get("RUNPOD_SSH_KEY", "").strip()
    if override:
        return (Path(override).expanduser(),)
    return (
        Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go",
        Path.home() / ".ssh" / "thetascan_runpod_ed25519",
    )


def _ssh_key_path() -> Path:
    failures: list[str] = []
    for path in _ssh_key_candidates():
        if not path.is_file():
            failures.append(f"not found: {path}")
            continue
        try:
            with path.open("rb") as stream:
                if not stream.read(1):
                    failures.append(f"empty: {path}")
                    continue
        except OSError as exc:
            failures.append(f"unreadable: {path}: {exc}")
            continue
        return path.resolve()
    detail = "; ".join(failures) or "no SSH key candidates"
    raise LauncherError(
        "no readable RunPod SSH private key; set RUNPOD_SSH_KEY or run "
        f"`runpodctl doctor` ({detail})"
    )


def _ssh_bootstrap_preflight() -> tuple[bool, str]:
    """Return whether account-key SSH can be used before creating a pod."""
    if not shutil.which("ssh"):
        return False, "OpenSSH client `ssh` was not found"
    try:
        key_path = _ssh_key_path()
    except LauncherError as exc:
        return False, str(exc)
    return True, f"readable RunPod SSH key at {key_path}"


def _select_bootstrap_transport(requested: str) -> tuple[str, str]:
    if requested == "environment":
        return "environment", "explicit environment transport"
    if requested not in {"auto", "ssh"}:
        raise LauncherError(f"unsupported bootstrap transport {requested!r}")
    ssh_available, detail = _ssh_bootstrap_preflight()
    if ssh_available:
        return "ssh", detail
    if requested == "ssh":
        raise LauncherError(f"SSH bootstrap preflight failed: {detail}")
    return "environment", f"automatic legacy fallback because {detail}"


def _bootstrap_via_ssh(
    api_key: str,
    pod_id: str,
    *,
    payload: bytes,
    startup_script: str,
    resume_checkpoint: Path | None = None,
    run_dir: Path,
    timeout_seconds: int,
) -> None:
    ssh_program = shutil.which("ssh")
    if not ssh_program:
        raise LauncherError("OpenSSH client `ssh` is required for safe pod bootstrap")
    key_path = _ssh_key_path()
    known_hosts = run_dir / "ssh-known-hosts"
    deadline = time.monotonic() + timeout_seconds
    ssh_base: list[str] | None = None
    last_error = "pod has no SSH endpoint yet"
    while time.monotonic() < deadline:
        try:
            info = _pod_info(api_key, pod_id)
        except (
            LauncherError,
            TimeoutError,
            urllib.error.URLError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            # The control plane occasionally stalls while the pod itself is
            # booting normally. The outer bootstrap deadline remains the
            # fail-closed bound, so a single API timeout must not kill the pod.
            last_error = f"transient RunPod API error: {exc}"
            time.sleep(5)
            continue
        runtime = (info or {}).get("runtime") or {}
        ports = runtime.get("ports") or []
        endpoint = next(
            (
                item
                for item in ports
                if int(item.get("privatePort") or 0) == 22
                and item.get("ip")
                and item.get("publicPort")
            ),
            None,
        )
        if endpoint is not None:
            ssh_base = [
                ssh_program,
                "-i", str(key_path),
                "-p", str(endpoint["publicPort"]),
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no",
                "-o", f"UserKnownHostsFile={known_hosts}",
                f"root@{endpoint['ip']}",
            ]
            try:
                probe = subprocess.run(
                    [*ssh_base, "true"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=20,
                    check=False,
                )
                if probe.returncode == 0:
                    break
                last_error = probe.stderr.decode("utf-8", errors="replace")[-500:]
            except (OSError, subprocess.TimeoutExpired) as exc:
                last_error = str(exc)
        time.sleep(5)
    else:
        raise LauncherError(
            f"pod SSH did not become ready within {timeout_seconds}s: {last_error}"
        )
    if ssh_base is None:
        raise LauncherError("internal error: SSH endpoint disappeared")

    transfers = (
        (payload, "cat > /tmp/thetascan-payload.tar.gz"),
        (startup_script.encode("utf-8"), "cat > /tmp/thetascan-run.sh"),
    )
    for data, remote_command in transfers:
        completed = subprocess.run(
            [*ssh_base, remote_command],
            input=data,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
            raise LauncherError(f"SSH payload transfer failed: {detail}")
    if resume_checkpoint is not None:
        try:
            with resume_checkpoint.open("rb") as handle:
                completed = subprocess.run(
                    [*ssh_base, f"cat > {REMOTE_RESUME_CHECKPOINT}"],
                    stdin=handle,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=SSH_CHECKPOINT_UPLOAD_TIMEOUT_SECONDS,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise LauncherError(
                "SSH resume checkpoint upload exceeded "
                f"{SSH_CHECKPOINT_UPLOAD_TIMEOUT_SECONDS} seconds"
            ) from exc
        except OSError as exc:
            raise LauncherError(f"could not upload resume checkpoint: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
            raise LauncherError(f"SSH resume checkpoint upload failed: {detail}")
    completed = subprocess.run(
        [
            *ssh_base,
            "chmod 700 /tmp/thetascan-run.sh && "
            "setsid -f bash /tmp/thetascan-run.sh "
            ">/tmp/thetascan-bootstrap.log 2>&1 </dev/null",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise LauncherError(f"SSH bootstrap start failed: {detail}")


def _pod_info(api_key: str, pod_id: str) -> dict[str, Any] | None:
    query = """
    query pod($input: PodFilter!) {
      pod(input: $input) {
        id name desiredStatus machine { gpuTypeId }
        runtime { uptimeInSeconds ports { privatePort publicPort ip type } }
      }
    }
    """
    return _gql(api_key, query, {"input": {"podId": pod_id}}).get("pod")


def _terminate_pod(api_key: str, pod_id: str) -> bool:
    mutation = """
    mutation podTerminate($input: PodTerminateInput!) { podTerminate(input: $input) }
    """
    delays = (0, 2, 5, 10, 20, 30)
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            _gql(api_key, mutation, {"input": {"podId": pod_id}})
        except Exception as exc:
            print(f"terminate retry for {pod_id}: {exc}", file=sys.stderr, flush=True)
        try:
            pods = _gql(api_key, "query { myself { pods { id desiredStatus } } }")[
                "myself"
            ]["pods"]
            matching = [item for item in pods if item["id"] == pod_id]
            if not matching or all(
                item.get("desiredStatus") == "TERMINATED" for item in matching
            ):
                return True
        except Exception as exc:
            print(
                f"termination verification retry for {pod_id}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    return False


def _fetch(url: str, timeout: int = 20) -> bytes | None:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except (OSError, urllib.error.URLError, TimeoutError):
        return None


def _expected_exports(run_id: str, *, checkpoint: bool) -> list[str]:
    result = [
        "done.flag",
        "status.json",
        "suite-result.json",
        "suite-config.json",
        "adapter-manifest.json",
        "runtime.json",
        "pod.log",
        f"smoke-{run_id}.log",
        f"production-{run_id}.log",
    ]
    if checkpoint:
        result.extend(("research-checkpoint.json", "research-checkpoint.pt"))
    return result


def _fetch_to_path(url: str, path: Path, timeout: int = 120) -> int | None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        size = temporary.stat().st_size
        if size <= 0:
            temporary.unlink(missing_ok=True)
            return None
        temporary.replace(path)
        return size
    except (OSError, urllib.error.URLError, TimeoutError, http.client.IncompleteRead):
        temporary.unlink(missing_ok=True)
        return None


def _download_exports(
    state: dict[str, Any], run_dir: Path, *, require_complete: bool = False
) -> dict[str, int]:
    proxy_url = str(state["proxy_url"])
    sizes: dict[str, int] = {
        name: (run_dir / name).stat().st_size
        for name in state["expected_exports"]
        if (run_dir / name).is_file() and (run_dir / name).stat().st_size > 0
    }
    pending = set(state["expected_exports"]).difference(sizes)
    for delay in (0, 2, 5, 10, 20):
        if delay:
            time.sleep(delay)
        names = sorted(pending)
        if not names:
            break
        for name in names:
            timeout = 300 if name == "research-checkpoint.pt" else 30
            size = _fetch_to_path(
                f"{proxy_url}/{name}", run_dir / name, timeout=timeout
            )
            if size is not None:
                sizes[name] = size
        pending.difference_update(sizes)
        if not pending:
            break
    _write_json(run_dir / "download-manifest.json", sizes)
    checkpoint_meta_path = run_dir / "research-checkpoint.json"
    checkpoint_path = run_dir / "research-checkpoint.pt"
    if checkpoint_meta_path.is_file() and checkpoint_path.is_file():
        checkpoint_meta = _read_json(checkpoint_meta_path)
        actual_size = checkpoint_path.stat().st_size
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if (
            checkpoint_meta.get("bytes") != actual_size
            or checkpoint_meta.get("sha256") != actual_sha256
        ):
            checkpoint_path.unlink(missing_ok=True)
            sizes.pop("research-checkpoint.pt", None)
            raise LauncherError("downloaded research checkpoint failed hash/size verification")
    if require_complete:
        run_id = str(state["run_id"])
        required = {
            "suite-result.json",
            f"smoke-{run_id}.log",
            f"production-{run_id}.log",
        }
        if "research-checkpoint.pt" in state["expected_exports"]:
            required.update(("research-checkpoint.json", "research-checkpoint.pt"))
        missing = sorted(required.difference(sizes))
        if missing:
            raise LauncherError(
                "done.flag was visible but required exports stayed unavailable after "
                f"five retries: {missing}"
            )
    return sizes


def _monitor_and_collect(
    api_key: str,
    state: dict[str, Any],
    state_path: Path,
    *,
    poll_seconds: int,
    boot_timeout_seconds: int,
) -> bool:
    pod_id = str(state["pod_id"])
    run_dir = state_path.parent
    proxy_url = str(state["proxy_url"])
    started = time.monotonic()
    remaining_hard_seconds = max(
        float(state.get("hard_deadline_epoch", 0.0)) - time.time(), 0.0
    )
    hard_deadline = started + (
        remaining_hard_seconds
        if state.get("hard_deadline_epoch")
        else int(state["hard_timeout_seconds_from_boot"])
    ) + 120
    boot_deadline = started + boot_timeout_seconds
    proxy_seen = False
    last_status = ""
    control_plane_errors = 0

    while time.monotonic() < hard_deadline:
        try:
            info = _pod_info(api_key, pod_id)
        except (
            LauncherError,
            TimeoutError,
            urllib.error.URLError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            control_plane_errors += 1
            if control_plane_errors == 1:
                print(
                    f"[{pod_id}] transient RunPod API error; monitoring continues: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            if control_plane_errors >= MAX_CONSECUTIVE_CONTROL_PLANE_ERRORS:
                raise LauncherError(
                    "RunPod control plane failed "
                    f"{control_plane_errors} consecutive monitor polls: {exc}"
                ) from exc
            info = False
        else:
            if control_plane_errors:
                print(f"[{pod_id}] RunPod API recovered", flush=True)
            control_plane_errors = 0

        if isinstance(info, dict):
            machine = info.get("machine") or {}
            actual_gpu = machine.get("gpuTypeId")
            if actual_gpu and actual_gpu != GPU_TYPE_ID:
                raise LauncherError(f"wrong GPU provisioned: {actual_gpu!r}")

        status_data = _fetch(f"{proxy_url}/status.json", timeout=10)
        if status_data:
            proxy_seen = True
            try:
                status = json.loads(status_data.decode("utf-8")).get("state", "unknown")
            except (UnicodeDecodeError, json.JSONDecodeError):
                status = "unparseable"
            if status != last_status:
                print(f"[{pod_id}] {status}", flush=True)
                last_status = status
        done = _fetch(f"{proxy_url}/done.flag", timeout=10)
        if done:
            done_text = done.decode("utf-8", errors="replace").strip()
            print(f"[{pod_id}] {done_text}")
            sizes = _download_exports(
                state, run_dir, require_complete=not done_text.startswith("failed ")
            )
            state.update(
                collected_at=_utc_now(),
                collected_files=sizes,
                lifecycle_state="collected",
            )
            _write_json(state_path, state)
            result_path = run_dir / "suite-result.json"
            if not result_path.is_file():
                return False
            return _read_json(result_path).get("status") == "passed"

        if info is None:
            break
        if info is False:
            time.sleep(poll_seconds)
            continue
        desired = info.get("desiredStatus")
        if desired not in {"RUNNING", "CREATED"}:
            break
        if not proxy_seen and time.monotonic() >= boot_deadline:
            raise LauncherError(
                f"pod proxy was not ready within {boot_timeout_seconds} seconds"
            )
        time.sleep(poll_seconds)

    _download_exports(state, run_dir)
    raise LauncherError(f"pod {pod_id} stopped or timed out before done.flag was collected")


def _state_dir(config: dict[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return HERE / "results" / "_local" / f"{config['suite_id']}-{stamp}"


def _launch(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config, run = _validate_config(config_path)
    smoke_start = int(
        config["protocols"]["smoke"].get("segment_start_step", 0)
    )
    if smoke_start > 0 and args.resume_checkpoint is None:
        raise LauncherError(
            f"continuation smoke segment starts at step {smoke_start}; "
            "--resume-checkpoint is required before a pod can be created"
        )
    if smoke_start == 0 and args.resume_checkpoint is not None:
        raise LauncherError(
            "--resume-checkpoint is invalid when the smoke segment starts at step 0"
        )
    hard_timeout_seconds, result_grace_seconds, max_budget_usd = (
        _resolve_launch_guards(args, config)
    )
    protocol_timeouts = sum(
        float(config["protocols"][stage]["timeout_seconds"])
        for stage in ("smoke", "production")
    )
    reserved_seconds = protocol_timeouts + result_grace_seconds
    if reserved_seconds > hard_timeout_seconds:
        raise LauncherError(
            "smoke timeout + production timeout + result grace exceeds the "
            "boot-absolute pod timeout"
        )
    revision, dirty = _git_identity()
    payload = _build_payload(config_path)
    source_sha = _sha256(payload)
    seed = (
        args.seed
        if args.seed is not None
        else int.from_bytes(os.urandom(8), "big") % (2**31 - 1)
    )
    resume: dict[str, Any] | None = None
    if args.resume_checkpoint is not None:
        if args.seed is None:
            raise LauncherError(
                "--resume-checkpoint requires an explicit --seed matching its metadata"
            )
        resume = _validate_resume_checkpoint(
            args.resume_checkpoint,
            config=config,
            run=run,
            seed=seed,
        )
    startup = _startup_script(
        run=run,
        data=config["data"],
        seed=seed,
        source_sha256=source_sha,
        git_revision=revision,
        git_dirty=dirty,
        hard_timeout_seconds=hard_timeout_seconds,
        result_grace_seconds=result_grace_seconds,
        resume_checkpoint_sha256=(resume["sha256"] if resume else None),
        resume_checkpoint_bytes=(resume["bytes"] if resume else 0),
    )
    legacy_environment_chunks = _pod_environment(payload)
    bootstrap_transport, bootstrap_transport_detail = _select_bootstrap_transport(
        args.bootstrap_transport
    )
    if resume is not None and bootstrap_transport != "ssh":
        raise LauncherError("--resume-checkpoint requires SSH bootstrap transport")
    legacy_docker_command = (
        _docker_command(startup) if bootstrap_transport == "environment" else None
    )
    preview = {
        "suite_id": config["suite_id"],
        "run_id": run["id"],
        "arm": run["arm"],
        "expected_model_params": run["expected_model_params"],
        "train_shards": config["data"]["train_shards"],
        "gpu": f"1 x {GPU_TYPE_ID}",
        "template_id": TEMPLATE_ID,
        "image_name": IMAGE_NAME,
        "image_index_digest": IMAGE_INDEX_DIGEST,
        "image_digest_attested": False,
        "cloud_type": args.cloud_type,
        "hard_timeout_seconds_from_boot": hard_timeout_seconds,
        "result_grace_seconds": result_grace_seconds,
        "setup_headroom_seconds": hard_timeout_seconds - reserved_seconds,
        "max_budget_usd": max_budget_usd,
        "payload_sha256": source_sha,
        "payload_bytes_gzip": len(payload),
        "bootstrap_transport_requested": args.bootstrap_transport,
        "bootstrap_transport": bootstrap_transport,
        "bootstrap_transport_detail": bootstrap_transport_detail,
        "legacy_payload_env_chunks": len(legacy_environment_chunks),
        "runtime_seed": seed,
        "resume_checkpoint": (
            {
                "path": str(resume["path"]),
                "metadata_path": str(resume["metadata_path"]),
                "bytes": resume["bytes"],
                "sha256": resume["sha256"],
                "step": resume["step"],
                "target_iterations": resume["target_iterations"],
                "grad_accum_steps": resume["grad_accum_steps"],
            }
            if resume
            else None
        ),
        "git_revision": revision,
        "git_dirty": None if dirty == "unknown" else dirty == "true",
    }
    print(json.dumps(preview, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    if not args.yes:
        raise LauncherError("refusing to create a billable pod without --yes")

    api_key = _load_api_key()
    balance, rate = _account_and_rate(api_key, args.cloud_type)
    maximum_cost = rate * hard_timeout_seconds / 3600.0
    print(
        f"balance=${balance:.2f}; rate=${rate:.2f}/h; per-pod hard cap~${maximum_cost:.2f}",
        flush=True,
    )
    if maximum_cost > max_budget_usd:
        raise LauncherError(
            f"hard cap ${maximum_cost:.2f} exceeds the effective max budget "
            f"${max_budget_usd:.2f}"
        )
    if balance < maximum_cost:
        raise LauncherError(
            f"balance ${balance:.2f} is below this pod's hard-cap cost ${maximum_cost:.2f}"
        )

    run_dir = _state_dir(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "requested-config.json").write_bytes(config_path.read_bytes())
    pod_id: str | None = None
    state_path = run_dir / "launch.json"
    hard_deadline_epoch = time.time() + hard_timeout_seconds
    terminate_after = datetime.fromtimestamp(
        hard_deadline_epoch, tz=timezone.utc
    ).isoformat().replace("+00:00", "Z")
    try:
        pod_environment: list[dict[str, str]] | None = None
        if bootstrap_transport == "environment":
            pod_environment = legacy_environment_chunks
        pod_id = _create_pod(
            api_key,
            name=args.name or f"theta-{run['id']}-{int(time.time())}",
            cloud_type=args.cloud_type,
            terminate_after=terminate_after,
            bootstrap_transport=bootstrap_transport,
            docker_command=legacy_docker_command,
            environment=pod_environment,
        )
        state = {
            **preview,
            "pod_id": pod_id,
            "proxy_url": f"https://{pod_id}-8000.proxy.runpod.net",
            "created_at": _utc_now(),
            "hard_deadline_epoch": hard_deadline_epoch,
            "platform_terminate_after": terminate_after,
            "max_budget_usd": max_budget_usd,
            "expected_exports": _expected_exports(
                run["id"],
                checkpoint=bool(config["protocols"]["production"].get("checkpoint_at_end")),
            ),
            "lifecycle_state": "bootstrapping",
        }
        _write_json(state_path, state)
        print(f"pod={pod_id}; state={state_path}", flush=True)
        if bootstrap_transport == "ssh":
            _bootstrap_via_ssh(
                api_key,
                pod_id,
                payload=payload,
                startup_script=startup,
                resume_checkpoint=(resume["path"] if resume else None),
                run_dir=run_dir,
                timeout_seconds=args.boot_timeout_seconds,
            )
        state["bootstrap_completed_at"] = _utc_now()
        state["lifecycle_state"] = "detached" if args.detach else "monitoring"
        _write_json(state_path, state)
        print(f"pod={pod_id}; bootstrap={bootstrap_transport}-started", flush=True)
        if args.detach:
            print(
                f"collect with: {sys.executable} {Path(__file__).resolve()} collect --state {state_path}",
                flush=True,
            )
            # Detached mode has both RunPod terminateAfter and the pod watchdog.
            pod_id = None
            return 0
        passed = _monitor_and_collect(
            api_key,
            state,
            state_path,
            poll_seconds=args.poll_seconds,
            boot_timeout_seconds=args.boot_timeout_seconds,
        )
        return 0 if passed else 2
    finally:
        if pod_id is not None:
            print(f"terminating pod {pod_id}", flush=True)
            if not _terminate_pod(api_key, pod_id):
                print(
                    f"CRITICAL: termination could not be verified for {pod_id}; "
                    "the pod-side absolute watchdog remains active",
                    file=sys.stderr,
                )


def _collect(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    state = _read_json(state_path)
    api_key = _load_api_key()
    pod_id = str(state["pod_id"])
    try:
        passed = _monitor_and_collect(
            api_key,
            state,
            state_path,
            poll_seconds=args.poll_seconds,
            boot_timeout_seconds=args.boot_timeout_seconds,
        )
        return 0 if passed else 2
    finally:
        print(f"terminating pod {pod_id}", flush=True)
        if not _terminate_pod(api_key, pod_id):
            print(
                f"CRITICAL: termination could not be verified for {pod_id}; "
                "the pod-side absolute watchdog remains active",
                file=sys.stderr,
            )


def _terminate(args: argparse.Namespace) -> int:
    state = _read_json(args.state.resolve())
    pod_id = str(state["pod_id"])
    if _terminate_pod(_load_api_key(), pod_id):
        print(f"terminated {pod_id}")
        return 0
    raise LauncherError(f"could not verify termination of {pod_id}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch", help="create and optionally monitor one pod")
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--name")
    launch.add_argument("--cloud-type", choices=("SECURE", "COMMUNITY"), default="SECURE")
    launch.add_argument(
        "--hard-timeout-seconds",
        type=int,
        help="override the config's boot-absolute safety timeout",
    )
    launch.add_argument(
        "--result-grace-seconds",
        type=int,
        help="override the config's post-training result-export grace period",
    )
    launch.add_argument("--boot-timeout-seconds", type=int, default=600)
    launch.add_argument("--poll-seconds", type=int, default=10)
    launch.add_argument(
        "--max-budget-usd",
        type=float,
        help="override the config's per-pod hard budget cap",
    )
    launch.add_argument(
        "--seed",
        type=int,
        help="shared runtime seed; pass the same fresh value to every comparison pod",
    )
    launch.add_argument(
        "--resume-checkpoint",
        type=Path,
        help=(
            "continue from an exported research-checkpoint.pt; its companion "
            "research-checkpoint.json must match the smoke start, target, run, "
            "seed, and effective grad_accum_steps"
        ),
    )
    launch.add_argument("--detach", action="store_true")
    launch.add_argument(
        "--bootstrap-transport",
        choices=("auto", "ssh", "environment"),
        default="auto",
        help=(
            "bootstrap path; auto tries the account and project SSH keys "
            "before the bounded legacy environment fallback"
        ),
    )
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--yes", action="store_true", help="authorize billable pod creation")
    launch.set_defaults(handler=_launch)

    collect = subparsers.add_parser("collect", help="monitor a detached pod, fetch, terminate")
    collect.add_argument("--state", type=Path, required=True)
    collect.add_argument("--boot-timeout-seconds", type=int, default=600)
    collect.add_argument("--poll-seconds", type=int, default=10)
    collect.set_defaults(handler=_collect)

    terminate = subparsers.add_parser("terminate", help="terminate the pod in a state file")
    terminate.add_argument("--state", type=Path, required=True)
    terminate.set_defaults(handler=_terminate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "launch":
        if args.seed is not None and not 0 <= args.seed <= 2**31 - 2:
            raise LauncherError("--seed must be between 0 and 2^31-2")
    return int(args.handler(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LauncherError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
