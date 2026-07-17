"""Prepare a clean, pinned parameter-golf checkout with the ThetaScan overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
MANIFEST_PATH = HERE / "adapter_manifest.json"
BUNDLED_OFFICIAL_ROOT = HERE / "_official_snapshot"


def _run(*args: str, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(
        list(args),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def _clone_official(repository: str, destination: Path) -> None:
    """Clone the pinned public harness, tolerating transient GitHub throttling.

    RunPod can place several fresh pods behind the same public egress address.
    GitHub occasionally answers simultaneous anonymous clones with HTTP 403;
    leaving a partial checkout behind would also make the next attempt fail.
    Keep retries bounded and always re-verify the pinned revision/blob below.
    """
    delays = (5, 15, 30)
    for attempt in range(1, len(delays) + 2):
        if destination.exists():
            shutil.rmtree(destination)
        try:
            _run(
                "git",
                "-c", "core.autocrlf=false",
                "-c", "core.longpaths=true",
                "-c", "http.userAgent=ThetaScan-Parameter-Golf/0.1",
                "clone", "--quiet",
                repository, str(destination),
            )
            return
        except subprocess.CalledProcessError:
            if attempt > len(delays):
                raise
            delay = delays[attempt - 1]
            print(
                f"official_clone_retry:attempt={attempt} sleep_seconds={delay}",
                flush=True,
            )
            time.sleep(delay)


def _materialize_official(official: dict[str, object], destination: Path) -> None:
    required = (
        str(official["path"]),
        str(official["data_downloader_path"]),
    )
    if all((BUNDLED_OFFICIAL_ROOT / relative).is_file() for relative in required):
        destination.mkdir(parents=True, exist_ok=True)
        for source in BUNDLED_OFFICIAL_ROOT.rglob("*"):
            if not source.is_file():
                continue
            target = destination / source.relative_to(BUNDLED_OFFICIAL_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        print("official_source:bundled_verified_blobs", flush=True)
        return
    _clone_official(str(official["repository"]), destination)
    _run(
        "git", "checkout", "--quiet", "--detach", str(official["revision"]),
        cwd=destination,
    )


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"adapter expected one {label}, found {count}")
    return source.replace(old, new, 1)


def _replace_region(source: str, start: str, end: str, replacement: str, label: str) -> str:
    start_at = source.find(start)
    if start_at < 0:
        raise RuntimeError(f"adapter could not find start of {label}")
    end_at = source.find(end, start_at)
    if end_at < 0:
        raise RuntimeError(f"adapter could not find end of {label}")
    return source[:start_at] + replacement + source[end_at:]


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data,
        usedforsecurity=False,
    ).hexdigest()


def _python_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def _thetascan_source_identity() -> dict[str, object]:
    version = "unknown"
    init_path = PROJECT_ROOT / "src" / "thetascan" / "__init__.py"
    if init_path.is_file():
        for line in init_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("__version__ = "):
                version = line.split("=", 1)[1].strip().strip("\"'")
                break
    identity: dict[str, object] = {"version": version}
    package_root = PROJECT_ROOT / "src" / "thetascan"
    identity["source_sha256"] = (
        _python_tree_sha256(package_root) if package_root.is_dir() else None
    )
    try:
        identity["git_revision"] = _run(
            "git", "rev-parse", "HEAD", cwd=PROJECT_ROOT, capture=True
        ).strip()
        identity["git_dirty"] = bool(
            _run("git", "status", "--porcelain", cwd=PROJECT_ROOT, capture=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        identity["git_revision"] = None
        identity["git_dirty"] = None
    return identity


def adapt_downloader(
    source: str, *, repository: str, revision: str, remote_root_prefix: str
) -> str:
    """Pin the official downloader to one Hugging Face dataset revision."""
    source = source.replace("\r\n", "\n")
    source = _replace_once(
        source,
        'ROOT = Path(__file__).resolve().parent',
        'ROOT = Path(__file__).resolve().parents[1] / "data"',
        "downloader output root",
    )
    source = _replace_once(
        source,
        'REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")',
        f'REPO_ID = "{repository}"',
        "dataset repository",
    )
    source = _replace_once(
        source,
        'REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")',
        f'REMOTE_ROOT_PREFIX = "{remote_root_prefix}"',
        "dataset remote root",
    )
    source = _replace_once(
        source,
        '            repo_type="dataset",\n',
        f'            repo_type="dataset",\n            revision="{revision}",\n',
        "dataset revision",
    )
    return (
        "# SPDX-License-Identifier: MIT\n"
        "# Derived from the pinned OpenAI parameter-golf source; see ../LICENSE.\n\n"
        + source
    )


def adapt_harness(source: str) -> str:
    """Add a small hybrid mixer boundary to pinned official train_gpt.py."""
    source = source.replace("\r\n", "\n")
    source = _replace_once(
        source,
        "import glob\n",
        "import glob\nimport hashlib\nimport json\n",
        "adapter imports",
    )
    source = _replace_once(
        source,
        '    seed = int(os.environ.get("SEED", 1337))',
        '    seed = int(os.environ["SEED"])  # generated by run_benchmark.py unless supplied',
        "seed default",
    )
    source = _replace_once(
        source,
        '    mlp_mult = int(os.environ.get("MLP_MULT", 2))',
        '    mlp_hidden = int(os.environ.get("MLP_HIDDEN", 1024))\n'
        '    mamba3_mlp_hidden = int(os.environ.get("MAMBA3_MLP_HIDDEN", 938))\n'
        '    theta_mlp_hidden = int(os.environ.get("THETA_MLP_HIDDEN", 1023))',
        "absolute FFN widths",
    )
    source = _replace_once(
        source,
        '    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))',
        '''    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Clean layer-swap axis added by the ThetaScan benchmark adapter.
    mixer_type = os.environ.get("MIXER_TYPE", "attn")
    hybrid_layers = os.environ.get("HYBRID_LAYERS", "")
    mamba3_block_parity = bool(int(os.environ.get("MAMBA3_BLOCK_PARITY", "0")))
    d_state = int(os.environ.get("D_STATE", 64))
    headdim = int(os.environ.get("HEADDIM", 64))
    expand = float(os.environ.get("EXPAND", 1.0))
    rope_fraction = float(os.environ.get("ROPE_FRACTION", 0.5))
    mamba3_chunk = int(os.environ.get("MAMBA3_CHUNK", 64))
    theta_recipe = os.environ.get("THETA_RECIPE", "gn-reference-v0.1")
    theta_rope = os.environ.get("THETA_ROPE", "partial")
    theta_backend = os.environ.get("THETA_BACKEND", "auto")
    theta_projection_layout = os.environ.get(
        "THETA_PROJECTION_LAYOUT", "mamba-shared"
    )
    optimizer_policy = os.environ.get("OPTIMIZER_POLICY", "muon-2d")''',
        "hybrid mixer options",
    )
    source = _replace_once(
        source,
        '    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))\n'
        '    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))',
        '''    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    # ``iterations`` is the global LR-schedule target. A segment can stop earlier
    # without pretending that its last step is the end of training.
    segment_start_step = int(os.environ.get("SEGMENT_START_STEP", 0))
    segment_end_step = int(os.environ.get("SEGMENT_END_STEP", iterations))
    val_start_step = int(os.environ.get("VAL_START_STEP", 0))
    expected_train_shards = int(os.environ.get("EXPECTED_TRAIN_SHARDS", 0))
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "")
    resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", "")
    resume_checkpoint_sha256 = os.environ.get("RESUME_CHECKPOINT_SHA256", "")
    resume_checkpoint_bytes = int(os.environ.get("RESUME_CHECKPOINT_BYTES", 0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))''',
        "segment and checkpoint hyperparameters",
    )

    source = _replace_region(
        source,
        "class TokenStream:\n",
        "# -----------------------------\n# TRANSFORMER MODULES",
        '''def shard_metadata(file: Path) -> dict[str, int | str]:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    return {
        "name": file.name,
        "tokens": num_tokens,
        "bytes": expected_size,
        "sha256": sha256_file(file),
    }


class TokenStream:
    # Reads shards sequentially. A segment is rejected before training when it
    # would wrap, so a resumable research run never silently repeats data.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.metadata = [shard_metadata(file) for file in self.files]
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    @property
    def total_tokens(self) -> int:
        return sum(int(item["tokens"]) for item in self.metadata)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "shard_prefix": self.metadata,
            "file_idx": self.file_idx,
            "pos": self.pos,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("schema_version") != 1:
            raise RuntimeError("unsupported TokenStream checkpoint schema")
        saved_prefix = state.get("shard_prefix")
        if not isinstance(saved_prefix, list) or not saved_prefix:
            raise RuntimeError("checkpoint has no training-shard prefix")
        if saved_prefix != self.metadata[: len(saved_prefix)]:
            raise RuntimeError(
                "training shard prefix changed; resume requires the original shards "
                "in the same order, with optional new shards appended"
            )
        file_idx = int(state["file_idx"])
        pos = int(state["pos"])
        if not 0 <= file_idx < len(saved_prefix):
            raise RuntimeError(f"invalid checkpoint shard index {file_idx}")
        self.file_idx = file_idx
        self.tokens = load_data_shard(self.files[file_idx])
        if not 0 <= pos <= self.tokens.numel():
            raise RuntimeError(f"invalid checkpoint shard position {pos}")
        self.pos = pos

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "rank": self.rank,
            "world_size": self.world_size,
            "stream": self.stream.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("schema_version") != 1:
            raise RuntimeError("unsupported DistributedTokenLoader checkpoint schema")
        if int(state["rank"]) != self.rank or int(state["world_size"]) != self.world_size:
            raise RuntimeError("rank/world-size changed across checkpoint resume")
        stream_state = state.get("stream")
        if not isinstance(stream_state, dict):
            raise RuntimeError("checkpoint has no TokenStream state")
        self.stream.load_state_dict(stream_state)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, object]:
    py_state = random.getstate()
    np_state = np.random.get_state()
    return {
        "python": [py_state[0], list(py_state[1]), py_state[2]],
        "numpy": [
            np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]
        ],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, object]) -> None:
    py_state = state["python"]
    np_state = state["numpy"]
    random.setstate((int(py_state[0]), tuple(py_state[1]), py_state[2]))
    np.random.set_state(
        (
            str(np_state[0]),
            np.asarray(np_state[1], dtype=np.uint32),
            int(np_state[2]),
            int(np_state[3]),
            float(np_state[4]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state["torch_cuda"]])


def checkpoint_contract(
    args: Hyperparameters,
    code_sha256: str,
    base_model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    world_size: int,
    grad_accum_steps: int,
) -> dict[str, object]:
    manifest = json.loads(
        Path(__file__).with_name("ADAPTER-MANIFEST.json").read_text(encoding="utf-8")
    )
    shape_rows = [
        [name, list(tensor.shape), str(tensor.dtype)]
        for name, tensor in base_model.state_dict().items()
    ]
    shape_sha256 = hashlib.sha256(
        json.dumps(shape_rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "code_sha256": code_sha256,
        "thetascan_source": manifest["thetascan_source"],
        "dataset_reference": manifest["dataset_reference"],
        "runtime": {
            # Torch 2.9 exposes ``__version__`` as a TorchVersion object. Keep
            # checkpoint metadata inside the weights_only safe-type allowlist.
            "torch": str(torch.__version__),
            "torch_cuda": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
        },
        "target_iterations": args.iterations,
        "warmdown_iters": args.warmdown_iters,
        "train_batch_tokens": args.train_batch_tokens,
        "train_seq_len": args.train_seq_len,
        "seed": args.seed,
        "world_size": world_size,
        "grad_accum_steps": grad_accum_steps,
        "model_shape_sha256": shape_sha256,
        "model_params": sum(p.numel() for p in base_model.parameters()),
        "architecture": {
            "mixer_type": args.mixer_type,
            "hybrid_layers": args.hybrid_layers,
            "model_dim": args.model_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "mlp_hidden": args.mlp_hidden,
            "mamba3_mlp_hidden": args.mamba3_mlp_hidden,
            "theta_mlp_hidden": args.theta_mlp_hidden,
            "d_state": args.d_state,
            "headdim": args.headdim,
            "expand": args.expand,
            "rope_fraction": args.rope_fraction,
            "mamba3_chunk": args.mamba3_chunk,
            "theta_recipe": args.theta_recipe,
            "theta_rope": args.theta_rope,
            "theta_backend": args.theta_backend,
            "theta_projection_layout": args.theta_projection_layout,
        },
        "optimizer": {
            "classes": [type(opt).__name__ for opt in optimizers],
            "policy": args.optimizer_policy,
            "matrix_lr": args.matrix_lr,
            "embed_lr": args.embed_lr,
            "head_lr": args.head_lr,
            "tied_embed_lr": args.tied_embed_lr,
            "scalar_lr": args.scalar_lr,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "adam_eps": args.adam_eps,
            "muon_momentum": args.muon_momentum,
            "muon_backend_steps": args.muon_backend_steps,
            "muon_momentum_warmup_start": args.muon_momentum_warmup_start,
            "muon_momentum_warmup_steps": args.muon_momentum_warmup_steps,
        },
    }


def save_research_checkpoint(
    path: Path,
    *,
    contract: dict[str, object],
    step: int,
    last_validation_step: int,
    cumulative_training_time_ms: float,
    base_model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    train_loader: DistributedTokenLoader,
) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "contract": contract,
        "step": step,
        "last_validation_step": last_validation_step,
        "cumulative_training_time_ms": cumulative_training_time_ms,
        "model": base_model.state_dict(),
        "optimizers": [opt.state_dict() for opt in optimizers],
        "dataloader": train_loader.state_dict(),
        "rng": capture_rng_state(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path.stat().st_size, sha256_file(path)


def load_research_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    contract: dict[str, object],
    expected_step: int,
    base_model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    train_loader: DistributedTokenLoader,
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if expected_bytes and actual_bytes != expected_bytes:
        raise RuntimeError(
            f"resume checkpoint size mismatch: expected {expected_bytes}, got {actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            "resume checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("unsupported research checkpoint schema")
    if payload.get("contract") != contract:
        raise RuntimeError("resume checkpoint contract does not match this run")
    step = int(payload.get("step", -1))
    if step != expected_step:
        raise RuntimeError(
            f"resume checkpoint step mismatch: expected {expected_step}, got {step}"
        )
    optimizer_states = payload.get("optimizers")
    if not isinstance(optimizer_states, list) or len(optimizer_states) != len(optimizers):
        raise RuntimeError("resume checkpoint optimizer list does not match this run")
    base_model.load_state_dict(payload["model"], strict=True)
    for optimizer, state in zip(optimizers, optimizer_states, strict=True):
        optimizer.load_state_dict(state)
    train_loader.load_state_dict(payload["dataloader"])
    restore_rng_state(payload["rng"])
    return {
        "step": step,
        "last_validation_step": int(payload.get("last_validation_step", -1)),
        "cumulative_training_time_ms": float(
            payload.get("cumulative_training_time_ms", 0.0)
        ),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
    }


# -----------------------------
# TRANSFORMER MODULES''',
        "resumable token stream and checkpoint helpers",
    )

    source = _replace_region(
        source,
        "def zeropower_via_newtonschulz5(",
        "\n\nclass Muon",
        '''def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7
) -> Tensor:
    # Orthogonalize either one 2D update or a 3D batch independently over
    # its final two dimensions. ThetaScan stores one W1/W2 matrix per head,
    # so a 3D tensor is deliberately equivalent to stacking the 2D result
    # for every head; heads are never flattened or mixed together.
    if G.ndim not in (2, 3):
        raise ValueError(f"Muon expects a 2D matrix or 3D matrix batch, got {G.ndim}D")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm(dim=(-2, -1), keepdim=True) + eps
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.transpose(-2, -1) if transposed else X
''',
        "batched Muon Newton-Schulz kernel",
    )
    source = _replace_once(
        source,
        "                    g *= max(1, g.size(0) / g.size(1)) ** 0.5",
        "                    g *= max(1, g.size(-2) / g.size(-1)) ** 0.5",
        "Muon matrix scaling",
    )

    source = _replace_region(
        source,
        "class MLP(nn.Module):\n",
        "\n\nclass Block(nn.Module):",
        '''class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


MIXER_TYPES = ("attn", "thetascan", "mamba3")
THETA_RECIPES = (
    "gn-reference-v0.1",
    "gn-expanded-reference-v0.1",
    "kernel-expanded-reference-v0.1",
)
OPTIMIZER_POLICIES = ("muon-2d", "muon-2d+theta")
THETA_PROJECTION_LAYOUTS = ("mamba-shared", "transformer-gqa", "independent")


def parse_hybrid_layers(spec: str, num_layers: int) -> set[int]:
    if not spec.strip():
        return set()
    layers = {int(token) for token in spec.split(",") if token.strip()}
    bad = sorted(layer for layer in layers if not 0 <= layer < num_layers)
    if bad:
        raise ValueError(f"HYBRID_LAYERS contains out-of-range layers: {bad}")
    return layers


def build_mixer(
    mixer_type: str, dim: int, args: Hyperparameters, layer_index: int
) -> nn.Module:
    if mixer_type == "thetascan":
        from thetascan_benchmark_adapter import build_thetascan_mixer
        return build_thetascan_mixer(
            dim,
            recipe=args.theta_recipe,
            rope_mode=args.theta_rope,
            backend=args.theta_backend,
            projection_layout=args.theta_projection_layout,
            layer_index=layer_index,
        )
    if mixer_type == "mamba3":
        try:
            from mamba_ssm.modules import mamba3 as mamba3_module
        except Exception as exc:
            raise RuntimeError(
                "The strict Mamba-3 arm requires official "
                "mamba_ssm.modules.mamba3.Mamba3; no fallback is used"
            ) from exc
        adapter_manifest = json.loads(
            Path(__file__).with_name("ADAPTER-MANIFEST.json").read_text(encoding="utf-8")
        )
        expected_source_hash = adapter_manifest["mamba3_reference"]["source_sha256"]
        module_path = Path(mamba3_module.__file__).resolve()
        actual_source_hash = hashlib.sha256(
            module_path.read_bytes().replace(b"\\r\\n", b"\\n")
        ).hexdigest()
        if actual_source_hash != expected_source_hash:
            raise RuntimeError(
                "Mamba-3 source mismatch: install the exact revision from "
                "ADAPTER-MANIFEST.json; "
                f"expected {expected_source_hash}, got {actual_source_hash} at {module_path}"
            )
        Mamba3 = mamba3_module.Mamba3
        mixer = Mamba3(
            d_model=dim,
            d_state=args.d_state,
            expand=args.expand,
            headdim=args.headdim,
            ngroups=1,
            rope_fraction=args.rope_fraction,
            is_outproj_norm=True,
            is_mimo=False,
            chunk_size=args.mamba3_chunk,
        )
        mixer.out_proj._zero_init = True
        return mixer
    raise ValueError(f"cannot build non-attention mixer {mixer_type!r}")
''',
        "MLP and mixer boundary",
    )
    source = _replace_region(
        source,
        "class Block(nn.Module):\n",
        "\n\nclass GPT(nn.Module):",
        '''class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_hidden: int,
        rope_base: float,
        qk_gain_init: float,
        mixer_type: str,
        args: Hyperparameters,
        layer_index: int,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.mixer_type = mixer_type
        self.attn = (
            CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
            if mixer_type == "attn"
            else build_mixer(mixer_type, dim, args, layer_index)
        )
        self.mlp = MLP(dim, mlp_hidden)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x
''',
        "hybrid Block",
    )
    source = _replace_once(
        source,
        "        mlp_mult: int,\n        tie_embeddings: bool,",
        "        mlp_hidden: int,\n        mamba3_mlp_hidden: int,\n        theta_mlp_hidden: int,\n        tie_embeddings: bool,",
        "GPT FFN arguments",
    )
    source = _replace_once(
        source,
        "        logit_softcap: float,\n        rope_base: float,\n        qk_gain_init: float,\n    ):",
        "        logit_softcap: float,\n        rope_base: float,\n        qk_gain_init: float,\n        mixer_type: str,\n        hybrid_layers: set[int],\n        args: Hyperparameters,\n    ):",
        "GPT hybrid arguments",
    )
    source = _replace_once(
        source,
        "                    mlp_mult,\n                    rope_base,\n                    qk_gain_init,",
        '''                    (
                        mamba3_mlp_hidden
                        if args.mamba3_block_parity
                        and mixer_type == "mamba3"
                        and i in hybrid_layers
                        else theta_mlp_hidden
                        if mixer_type == "thetascan" and i in hybrid_layers
                        else mlp_hidden
                    ),
                    rope_base,
                    qk_gain_init,
                    mixer_type=(mixer_type if i in hybrid_layers else "attn"),
                    args=args,
                    layer_index=i,''',
        "per-layer mixer and FFN",
    )
    source = _replace_once(
        source,
        "    args = Hyperparameters()\n    zeropower_via_newtonschulz5",
        '''    args = Hyperparameters()
    if not 0 <= args.segment_start_step < args.segment_end_step <= args.iterations:
        raise ValueError(
            "require 0 <= SEGMENT_START_STEP < SEGMENT_END_STEP <= ITERATIONS; "
            f"got {args.segment_start_step}, {args.segment_end_step}, {args.iterations}"
        )
    if args.val_start_step < 0:
        raise ValueError(f"VAL_START_STEP must be nonnegative, got {args.val_start_step}")
    if args.max_wallclock_seconds < 0:
        raise ValueError("MAX_WALLCLOCK_SECONDS must be nonnegative")
    if bool(args.resume_checkpoint) != (args.segment_start_step > 0):
        raise ValueError(
            "RESUME_CHECKPOINT is required exactly when SEGMENT_START_STEP is positive"
        )
    if args.mixer_type not in MIXER_TYPES:
        raise ValueError(f"MIXER_TYPE must be one of {MIXER_TYPES}, got {args.mixer_type!r}")
    if args.theta_recipe not in THETA_RECIPES:
        raise ValueError(
            f"THETA_RECIPE must be one of {THETA_RECIPES}, "
            f"got {args.theta_recipe!r}"
        )
    if args.optimizer_policy not in OPTIMIZER_POLICIES:
        raise ValueError(
            f"OPTIMIZER_POLICY must be one of {OPTIMIZER_POLICIES}, "
            f"got {args.optimizer_policy!r}"
        )
    if args.optimizer_policy == "muon-2d+theta" and args.mixer_type != "thetascan":
        raise ValueError(
            "OPTIMIZER_POLICY='muon-2d+theta' requires MIXER_TYPE='thetascan'; "
            "use 'muon-2d' for attention and Mamba-3 arms"
        )
    if args.theta_projection_layout not in THETA_PROJECTION_LAYOUTS:
        raise ValueError(
            f"THETA_PROJECTION_LAYOUT must be one of {THETA_PROJECTION_LAYOUTS}, "
            f"got {args.theta_projection_layout!r}"
        )
    hybrid_set = parse_hybrid_layers(args.hybrid_layers, args.num_layers)
    if args.mixer_type == "attn" and hybrid_set:
        raise ValueError("attention baseline requires an empty HYBRID_LAYERS")
    if args.mixer_type != "attn" and not hybrid_set:
        raise ValueError("a non-attention arm requires at least one HYBRID_LAYERS entry")
    if args.mixer_type == "thetascan":
        expected_theta_hidden = {
            "mamba-shared": 1023,
            "transformer-gqa": 832,
            "independent": 576,
        }[args.theta_projection_layout]
        if args.theta_recipe == "gn-expanded-reference-v0.1":
            # The expanded GN threshold vector lives at the doubled effective
            # width (+3,072 parameters per block); the swapped FFN gives the
            # same amount back to stay parameter-matched.
            expected_theta_hidden -= 3
        theta_shape = (
            args.model_dim,
            args.num_heads,
            args.num_kv_heads,
            args.mlp_hidden,
            args.theta_mlp_hidden,
            tuple(sorted(hybrid_set)),
        )
        expected_theta_shape = (512, 8, 4, 1024, expected_theta_hidden, (4, 5))
        if theta_shape != expected_theta_shape:
            raise ValueError(
                "ThetaScan parity requires model/heads/kv_heads/full_ffn/theta_ffn/"
                f"layers={expected_theta_shape}, got {theta_shape}"
            )
    if args.mamba3_block_parity:
        if args.mixer_type != "mamba3" or len(hybrid_set) != 2:
            raise ValueError("Mamba-3 block parity requires exactly two Mamba-3 layers")
        shape = (
            args.model_dim, args.mlp_hidden, args.mamba3_mlp_hidden,
            args.d_state, args.headdim, args.expand,
            args.rope_fraction, args.mamba3_chunk,
        )
        if shape != (512, 1024, 938, 64, 64, 1.0, 0.5, 64):
            raise ValueError(
                "block parity requires d_model/full_ffn/mamba_ffn/d_state/headdim/"
                "expand/rope_fraction/chunk = 512/1024/938/64/64/1/0.5/64, "
                f"got {shape}"
            )
    zeropower_via_newtonschulz5''',
        "hybrid validation",
    )
    source = _replace_once(
        source,
        "        mlp_mult=args.mlp_mult,\n        tie_embeddings=args.tie_embeddings,",
        "        mlp_hidden=args.mlp_hidden,\n        mamba3_mlp_hidden=args.mamba3_mlp_hidden,\n        theta_mlp_hidden=args.theta_mlp_hidden,\n        tie_embeddings=args.tie_embeddings,",
        "GPT construction FFN widths",
    )
    source = _replace_once(
        source,
        "        qk_gain_init=args.qk_gain_init,\n    ).to(device).bfloat16()",
        "        qk_gain_init=args.qk_gain_init,\n        mixer_type=args.mixer_type,\n        hybrid_layers=hybrid_set,\n        args=args,\n    ).to(device).bfloat16()",
        "GPT construction hybrid arguments",
    )
    source = _replace_once(
        source,
        "        if isinstance(module, CastedLinear):\n            module.float()",
        '''        if isinstance(module, CastedLinear) or type(module).__name__ in (
            "ThetaScan", "Mamba3"
        ):
            module.float()''',
        "mixer precision policy",
    )
    source = _replace_once(
        source,
        "    grad_accum_steps = 8 // world_size",
        '''    default_grad_accum_steps = 8 // world_size
    grad_accum_steps = int(os.environ.get(
        "GRAD_ACCUM_STEPS", default_grad_accum_steps
    ))
    if grad_accum_steps < 1:
        raise ValueError(
            f"GRAD_ACCUM_STEPS must be positive, got {grad_accum_steps}"
        )''',
        "explicit gradient-accumulation override",
    )
    source = _replace_once(
        source,
        "    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)",
        "    compiled_model = torch.compile(\n        base_model, dynamic=False, fullgraph=not hybrid_set\n    )",
        "hybrid compile policy",
    )
    source = _replace_region(
        source,
        "    # Optimizer split:\n",
        "    n_params = ",
        '''    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - ordinary 2D matrices in transformer blocks use MATRIX_LR via Muon
    # - with muon-2d+theta, ThetaScan memory weights W1/W2 use batched
    #   per-head Muon
    # - all remaining vectors, scalars, and ThetaScan controls use Adam
    block_named_params = list(base_model.blocks.named_parameters())
    block_param_by_name = dict(block_named_params)
    theta_prefixes = tuple(
        f"{layer}.attn._core." for layer in sorted(hybrid_set)
    ) if args.mixer_type == "thetascan" else ()
    theta_weight_named_params = sorted(
        (
            (name, p)
            for name, p in block_named_params
            if theta_prefixes
            and name.startswith(theta_prefixes)
            and (
                ".attn._core.W1." in name
                or ".attn._core.W2." in name
            )
        ),
        key=lambda item: item[0],
    )
    theta_weight_slots = ["W1", "W2"]
    expected_theta_weight_names = sorted(
        f"{layer}.attn._core.{weight}.0"
        for layer in sorted(hybrid_set)
        for weight in theta_weight_slots
    ) if args.mixer_type == "thetascan" else []
    actual_theta_weight_names = [name for name, _ in theta_weight_named_params]
    if actual_theta_weight_names != expected_theta_weight_names:
        raise RuntimeError(
            "ThetaScan W1/W2 optimizer targets do not match the fail-closed "
            f"benchmark contract: expected={expected_theta_weight_names}, "
            f"actual={actual_theta_weight_names}"
        )
    bad_theta_shapes = {
        name: tuple(p.shape)
        for name, p in theta_weight_named_params
        if p.ndim != 3 or p.shape[0] != args.num_heads
    }
    if bad_theta_shapes:
        raise RuntimeError(
            "ThetaScan Muon targets must be [heads, rows, columns] with "
            f"heads={args.num_heads}; bad={bad_theta_shapes}"
        )

    matrix_named_params = [
        (name, p)
        for name, p in block_named_params
        if p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        # Temporal-mode-bank scalar controls can be 2D tensors. Keep those
        # matrix-shaped controls on Adam; ordinary
        # core projections (proj_k/q/v/out.weight) remain Muon matrices.
        and name.rsplit(".", 1)[-1] not in {
            "fade_alpha", "fade_eta"
        }
    ]
    routed_theta_names = set(
        expected_theta_weight_names
        if args.optimizer_policy == "muon-2d+theta"
        else ()
    )
    theta_muon_named_params = [
        (name, block_param_by_name[name]) for name in expected_theta_weight_names
        if name in routed_theta_names
    ]
    muon_names = {
        *(name for name, _ in matrix_named_params),
        *(name for name, _ in theta_muon_named_params),
    }
    scalar_named_params = [
        (name, p) for name, p in block_named_params if name not in muon_names
    ]

    routed_names = [
        *(name for name, _ in matrix_named_params),
        *(name for name, _ in theta_muon_named_params),
        *(name for name, _ in scalar_named_params),
    ]
    all_block_names = [name for name, _ in block_named_params]
    if len(routed_names) != len(set(routed_names)) or sorted(routed_names) != sorted(all_block_names):
        raise RuntimeError("optimizer routing must assign every block parameter exactly once")
    actual_routed_theta_names = [name for name, _ in theta_muon_named_params]
    expected_routed_theta_names = (
        expected_theta_weight_names
        if args.optimizer_policy == "muon-2d+theta"
        else []
    )
    if actual_routed_theta_names != expected_routed_theta_names:
        raise RuntimeError(
            "ThetaScan Muon routing mismatch: "
            f"expected={expected_routed_theta_names}, actual={actual_routed_theta_names}"
        )

    matrix_name_set = {name for name, _ in matrix_named_params}
    theta_weight_name_set = set(expected_theta_weight_names)
    theta_control_named_params = sorted(
        (
            (name, p)
            for name, p in block_named_params
            if theta_prefixes
            and name.startswith(theta_prefixes)
            and name not in matrix_name_set
            and name not in theta_weight_name_set
        ),
        key=lambda item: item[0],
    )
    scalar_name_set = {name for name, _ in scalar_named_params}
    non_adam_theta_controls = [
        name for name, _ in theta_control_named_params if name not in scalar_name_set
    ]
    if non_adam_theta_controls:
        raise RuntimeError(
            "ThetaScan controls must remain on Adam; incorrectly routed: "
            f"{non_adam_theta_controls}"
        )
    if args.optimizer_policy == "muon-2d":
        non_adam_theta_weights = [
            name for name in expected_theta_weight_names if name not in scalar_name_set
        ]
        if non_adam_theta_weights:
            raise RuntimeError(
                "muon-2d must leave 3D ThetaScan W1/W2 on Adam; "
                f"incorrectly routed: {non_adam_theta_weights}"
            )

    matrix_params = [p for _, p in matrix_named_params]
    theta_muon_params = [p for _, p in theta_muon_named_params]
    scalar_params = [p for _, p in scalar_named_params]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        [*matrix_params, *theta_muon_params],
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

''',
        "optimizer routing",
    )
    source = _replace_once(
        source,
        "    n_params = sum(p.numel() for p in base_model.parameters())\n"
        '    log0(f"model_params:{n_params}")',
        '''    n_params = sum(p.numel() for p in base_model.parameters())
    if args.mixer_type == "thetascan":
        has_dual_recency = args.theta_recipe == "kernel-expanded-reference-v0.1"
        has_kernel_relu2_threshold = (
            args.theta_recipe == "kernel-expanded-reference-v0.1"
        )
        has_learned_threshold = (
            args.theta_recipe == "gn-expanded-reference-v0.1"
        )
        # Expanded recipes double the effective memory width (mem_mult 6 at
        # expansion 2) while the trainable W1/W2 cores stay at the dense
        # reference width; only width-sized controls scale with it.
        recipe_memory_multiplier = (
            6
            if args.theta_recipe in (
                "gn-expanded-reference-v0.1",
                "kernel-expanded-reference-v0.1",
            )
            else 3
        )
        expected_weight_slots = ["W1", "W2"]
        expected_weight_names = sorted(
            f"{layer}.attn._core.{weight}.0"
            for layer in (4, 5)
            for weight in expected_weight_slots
        )
        expected_control_names = sorted(
            [
                f"{layer}.attn._core.{name}"
                for layer in (4, 5)
                for name in ("fade_alpha", "fade_eta")
            ]
            + (
                [f"{layer}.attn._core.kq_bias" for layer in (4, 5)]
                if args.theta_projection_layout == "mamba-shared"
                else []
            )
            + (
                [
                    f"{layer}.attn._core.kernel_relu2_threshold"
                    for layer in (4, 5)
                ]
                if has_kernel_relu2_threshold
                else []
            )
            + (
                [f"{layer}.attn._core.Wg.0" for layer in (4, 5)]
                if has_learned_threshold
                else []
            )
        )
        route_theta = args.optimizer_policy == "muon-2d+theta"
        profile = json.loads(
            Path(__file__).with_name("ADAPTER-MANIFEST.json").read_text(
                encoding="utf-8"
            )
        )["expected_theta_profiles"][args.theta_projection_layout]
        # Account for the selected recipe's parameter-matching FFN width.
        ffn_parameter_delta = (
            2 * len(hybrid_set) * args.model_dim
            * (args.theta_mlp_hidden - profile["swapped_ffn_hidden"])
        )
        control_parameter_delta = (
            (4 * args.num_heads if has_dual_recency else 0)
            + (2 * args.num_heads if has_kernel_relu2_threshold else 0)
            + (
                2 * args.num_heads * recipe_memory_multiplier * args.headdim
                if has_learned_threshold
                else 0
            )
        )
        control_tensor_delta = (
            (2 if has_learned_threshold else 0)
            + (2 if has_kernel_relu2_threshold else 0)
        )
        theta_candidate_parameter_count = profile["theta_candidate_parameter_count"]
        expected_contract = {
            "model_params": (
                profile["model_params"]
                + control_parameter_delta
                + ffn_parameter_delta
            ),
            "muon_2d_tensor_count": profile["muon_2d_tensor_count"],
            "muon_2d_parameter_count": (
                profile["muon_2d_parameter_count"] + ffn_parameter_delta
            ),
            "theta_candidate_names": expected_weight_names,
            "theta_candidate_parameter_count": theta_candidate_parameter_count,
            "theta_muon_names": expected_weight_names if route_theta else [],
            "theta_muon_parameter_count": (
                theta_candidate_parameter_count if route_theta else 0
            ),
            "theta_control_names": expected_control_names,
            "theta_control_parameter_count": (
                profile["theta_control_parameter_count"] + control_parameter_delta
            ),
            "adam_block_and_skip_tensor_count": profile[
                "adam_block_and_skip_tensor_count_with_theta_muon"
                if route_theta
                else "adam_block_and_skip_tensor_count_without_theta_muon"
            ] + control_tensor_delta,
            "adam_block_and_skip_parameter_count": profile[
                "adam_block_and_skip_parameter_count_with_theta_muon"
                if route_theta
                else "adam_block_and_skip_parameter_count_without_theta_muon"
            ] + control_parameter_delta,
        }
        actual_contract = {
            "model_params": n_params,
            "muon_2d_tensor_count": len(matrix_named_params),
            "muon_2d_parameter_count": sum(p.numel() for _, p in matrix_named_params),
            "theta_candidate_names": actual_theta_weight_names,
            "theta_candidate_parameter_count": sum(
                p.numel() for _, p in theta_weight_named_params
            ),
            "theta_muon_names": actual_routed_theta_names,
            "theta_muon_parameter_count": sum(
                p.numel() for _, p in theta_muon_named_params
            ),
            "theta_control_names": [name for name, _ in theta_control_named_params],
            "theta_control_parameter_count": sum(
                p.numel() for _, p in theta_control_named_params
            ),
            "adam_block_and_skip_tensor_count": len(scalar_params),
            "adam_block_and_skip_parameter_count": sum(p.numel() for p in scalar_params),
        }
        contract_mismatches = {
            key: {"expected": expected_contract[key], "actual": actual_contract[key]}
            for key in expected_contract
            if actual_contract[key] != expected_contract[key]
        }
        if contract_mismatches:
            raise RuntimeError(
                f"public ThetaScan benchmark contract mismatch: {contract_mismatches}"
            )
    log0(f"model_params:{n_params}")''',
        "ThetaScan benchmark optimizer contract",
    )
    source = _replace_once(
        source,
        '    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")',
        '''    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(f"optimizer_policy:{args.optimizer_policy}")
    log0(f"theta_projection_layout:{args.theta_projection_layout}")
    log0(
        f"ffn_hidden:full={args.mlp_hidden} mamba3={args.mamba3_mlp_hidden} "
        f"thetascan={args.theta_mlp_hidden}"
    )
    log0(
        "optimizer_muon_2d:"
        f"tensor_count={len(matrix_named_params)} "
        f"parameter_count={sum(p.numel() for _, p in matrix_named_params)}"
    )
    log0(
        "optimizer_muon_theta:"
        f"tensor_count={len(theta_muon_named_params)} "
        f"parameter_count={sum(p.numel() for _, p in theta_muon_named_params)} "
        f"names={json.dumps([name for name, _ in theta_muon_named_params])}"
    )
    log0(
        "optimizer_adam_theta_controls:"
        f"tensor_count={len(theta_control_named_params)} "
        f"parameter_count={sum(p.numel() for _, p in theta_control_named_params)} "
        f"names={json.dumps([name for name, _ in theta_control_named_params])}"
    )
    log0(
        "optimizer_adam_block_and_skip:"
        f"tensor_count={len(scalar_params)} "
        f"parameter_count={sum(p.numel() for p in scalar_params)}"
    )
    log0(f"hybrid:mixer_type={args.mixer_type} layers={sorted(hybrid_set)}")
    if hybrid_set:
        implementations = sorted({type(base_model.blocks[i].attn).__name__ for i in hybrid_set})
        log0(f"mixer_impl:{','.join(implementations)}")
        reference_layer = next(i for i in range(args.num_layers) if i not in hybrid_set)
        mixer_sub = sum(p.numel() for p in base_model.blocks[min(hybrid_set)].attn.parameters())
        attention_sub = sum(p.numel() for p in base_model.blocks[reference_layer].attn.parameters())
        log0(
            f"hybrid:mixer_sublayer_params:{mixer_sub} "
            f"attn_sublayer_params:{attention_sub}"
        )
        if args.mixer_type == "thetascan":
            resolved = base_model.blocks[min(hybrid_set)].attn._parameter_golf_config
            log0(f"thetascan_config:{resolved}")
        if args.mamba3_block_parity:
            full_ffn = sum(p.numel() for p in base_model.blocks[reference_layer].mlp.parameters())
            actual_ffn = sum(
                p.numel()
                for layer in hybrid_set
                for p in base_model.blocks[layer].mlp.parameters()
            )
            widths = {base_model.blocks[layer].mlp.hidden for layer in hybrid_set}
            if widths != {938}:
                raise RuntimeError(f"unexpected Mamba-3 FFN widths: {widths}")
            mixer_delta = len(hybrid_set) * (mixer_sub - attention_sub)
            ffn_delta = actual_ffn - len(hybrid_set) * full_ffn
            expected_ffn_delta = len(hybrid_set) * 2 * args.model_dim * (
                args.mamba3_mlp_hidden - args.mlp_hidden
            )
            if ffn_delta != expected_ffn_delta:
                raise RuntimeError(
                    f"FFN delta mismatch: actual={ffn_delta}, expected={expected_ffn_delta}"
                )
            net_delta = mixer_delta + ffn_delta
            baseline_params = n_params - net_delta
            model_delta_pct = 100.0 * net_delta / baseline_params
            expected = json.loads(
                Path(__file__).with_name("ADAPTER-MANIFEST.json").read_text(
                    encoding="utf-8"
                )
            )["expected_mamba3_parity_delta"]
            actual = {
                "mixer_two_blocks": mixer_delta,
                "ffn_two_blocks": ffn_delta,
                "net": net_delta,
                "pinned_attention_model_params": baseline_params,
                "pinned_mamba3_model_params": n_params,
            }
            mismatches = {
                key: (actual[key], expected[key])
                for key in actual
                if actual[key] != expected[key]
            }
            if mismatches:
                raise RuntimeError(
                    f"Mamba-3 block parity mismatch: {mismatches}"
                )
            log0(
                "mamba3_block_parity:"
                f"layers={sorted(hybrid_set)} full_ffn_hidden={args.mlp_hidden} "
                f"mamba_ffn_hidden={args.mamba3_mlp_hidden} "
                f"mixer_shape=d_state:{args.d_state},headdim:{args.headdim},expand:{args.expand:g} "
                f"mixer_delta:{mixer_delta:+d} ffn_delta:{ffn_delta:+d} "
                f"net_delta:{net_delta:+d} model_delta_pct:{model_delta_pct:+.4f}%"
            )''',
        "hybrid parameter report",
    )
    source = _replace_once(
        source,
        '''    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)''',
        '''    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if args.expected_train_shards and actual_train_files != args.expected_train_shards:
        raise RuntimeError(
            f"expected {args.expected_train_shards} train shards, found {actual_train_files}"
        )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)''',
        "expected training shard count",
    )
    source = _replace_region(
        source,
        "    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)\n",
        "    log0(\n        f\"peak memory allocated:",
        '''    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    # Wallclock is only an emergency stop. Learning rate is a pure function of
    # global step and the predeclared target, so resuming cannot reset it.
    max_wallclock_ms = (
        1000.0 * args.max_wallclock_seconds
        if args.max_wallclock_seconds > 0
        else None
    )

    def lr_mul(step: int) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        warmdown_start = max(args.iterations - args.warmdown_iters, 0)
        if warmdown_start <= step < args.iterations:
            return max(
                (args.iterations - step) / max(args.warmdown_iters, 1), 0.0
            )
        return 1.0

    # Warmup primes compiled forward/backward/optimizer paths, then restores the
    # true initialization. On resume it happens before loading the checkpoint.
    if args.warmup_steps > 0:
        initial_model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in base_model.state_dict().items()
        }
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len, grad_accum_steps
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if (
                args.warmup_steps <= 20
                or (warmup_step + 1) % 10 == 0
                or warmup_step + 1 == args.warmup_steps
            ):
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    stream_tokens_per_step = (
        args.train_batch_tokens + world_size * grad_accum_steps
    )
    required_stream_tokens = args.segment_end_step * stream_tokens_per_step
    available_stream_tokens = train_loader.stream.total_tokens
    if available_stream_tokens < required_stream_tokens:
        raise RuntimeError(
            "training shards would wrap before SEGMENT_END_STEP: "
            f"available={available_stream_tokens}, required={required_stream_tokens}, "
            f"segment_end={args.segment_end_step}"
        )
    log0(
        "train_stream_capacity:"
        f"available={available_stream_tokens} required={required_stream_tokens} "
        f"tokens_per_step={stream_tokens_per_step} no_wrap=true"
    )

    if (args.checkpoint_path or args.resume_checkpoint) and world_size != 1:
        raise RuntimeError("research checkpoint/resume currently requires WORLD_SIZE=1")
    contract = checkpoint_contract(
        args,
        hashlib.sha256(code.encode("utf-8")).hexdigest(),
        base_model,
        optimizers,
        world_size,
        grad_accum_steps,
    )

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    step = 0
    last_validation_step = -1
    cumulative_training_time_ms = 0.0
    if args.resume_checkpoint:
        restored = load_research_checkpoint(
            Path(args.resume_checkpoint),
            expected_sha256=args.resume_checkpoint_sha256,
            expected_bytes=args.resume_checkpoint_bytes,
            contract=contract,
            expected_step=args.segment_start_step,
            base_model=base_model,
            optimizers=optimizers,
            train_loader=train_loader,
        )
        step = int(restored["step"])
        last_validation_step = int(restored["last_validation_step"])
        cumulative_training_time_ms = float(restored["cumulative_training_time_ms"])
        log0(
            f"resume_loaded:path={args.resume_checkpoint} step:{step} "
            f"target:{args.iterations} bytes:{restored['bytes']} "
            f"sha256:{restored['sha256']}"
        )
    elif args.segment_start_step != 0:
        raise RuntimeError("nonzero segment start requires a loaded checkpoint")

    segment_training_time_ms = 0.0
    wallclock_stop_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    while True:
        reached_segment_end = step >= args.segment_end_step
        reached_wallclock = (
            wallclock_stop_step is not None and step >= wallclock_stop_step
        )
        last_step = reached_segment_end or reached_wallclock
        scheduled_validation = (
            args.val_loss_every > 0
            and step >= args.val_start_step
            and step % args.val_loss_every == 0
        )
        should_validate = (
            (last_step or scheduled_validation) and step != last_validation_step
        )
        if should_validate:
            torch.cuda.synchronize()
            elapsed = 1000.0 * (time.perf_counter() - t0)
            segment_training_time_ms += elapsed
            cumulative_training_time_ms += elapsed
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            last_validation_step = step
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} "
                f"val_bpb:{val_bpb:.4f} "
                f"train_time:{cumulative_training_time_ms:.0f}ms "
                f"step_avg:{cumulative_training_time_ms / max(step, 1):.2f}ms"
            )
            if args.mixer_type == "thetascan":
                bank_timescales = {}
                for layer in sorted(hybrid_set):
                    core = base_model.blocks[layer].attn._core
                    alpha = torch.sigmoid(core.fade_alpha.detach().float())
                    half_life = torch.log(torch.full_like(alpha, 0.5)) / torch.log(alpha)
                    eta = core.fade_blends().detach().float()
                    bank_timescales[str(layer)] = {
                        "alpha": alpha.cpu().tolist(),
                        "half_life": half_life.cpu().tolist(),
                        "eta": eta.cpu().tolist(),
                    }
                log0(
                    "bank_timescales_per_head:"
                    + json.dumps(
                        bank_timescales, sort_keys=True, separators=(",", ":")
                    )
                )
            if (
                args.mixer_type == "thetascan"
                and args.theta_recipe == "kernel-expanded-reference-v0.1"
            ):
                thresholds = {
                    str(layer): base_model.blocks[layer].attn._core
                    .kernel_relu2_threshold.detach().float().cpu().tolist()
                    for layer in sorted(hybrid_set)
                }
                log0(
                    "kernel_relu2_threshold_per_head:"
                    + json.dumps(thresholds, sort_keys=True, separators=(",", ":"))
                )
            if (
                args.mixer_type == "thetascan"
                and args.theta_recipe == "gn-expanded-reference-v0.1"
            ):
                threshold_stats = {}
                for layer in sorted(hybrid_set):
                    core = base_model.blocks[layer].attn._core
                    primary = core.Wg[0].detach().float()
                    threshold_stats[str(layer)] = {
                        "mean": primary.mean().item(),
                        "std": primary.std().item(),
                        "min": primary.min().item(),
                        "max": primary.max().item(),
                    }
                log0(
                    "gn_threshold_stats:"
                    + json.dumps(
                        threshold_stats, sort_keys=True, separators=(",", ":")
                    )
                )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if reached_wallclock and not reached_segment_end:
                log0(
                    "stopping_early:wallclock_safety "
                    f"segment_train_time:{segment_training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations} requested_end:{args.segment_end_step}"
                )
            else:
                log0(
                    f"segment_complete:start={args.segment_start_step} end={step} "
                    f"target={args.iterations}"
                )
            break

        scale = lr_mul(step)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = (
            min(step / args.muon_momentum_warmup_steps, 1.0)
            if args.muon_momentum_warmup_steps > 0
            else 1.0
        )
        muon_momentum = (
            (1 - frac) * args.muon_momentum_warmup_start
            + frac * args.muon_momentum
        )
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        elapsed = 1000.0 * (time.perf_counter() - t0)
        approx_segment_time_ms = segment_training_time_ms + elapsed
        approx_cumulative_time_ms = cumulative_training_time_ms + elapsed
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"lr_mul:{scale:.8f} train_time:{approx_cumulative_time_ms:.0f}ms "
                f"step_avg:{approx_cumulative_time_ms / step:.2f}ms"
            )

        reached_cap = (
            max_wallclock_ms is not None
            and approx_segment_time_ms >= max_wallclock_ms
        )
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if wallclock_stop_step is None and reached_cap:
            wallclock_stop_step = step

    if args.checkpoint_path:
        if not master_process:
            raise RuntimeError("only rank zero may save the research checkpoint")
        checkpoint_file = Path(args.checkpoint_path).resolve()
        checkpoint_bytes, checkpoint_sha256 = save_research_checkpoint(
            checkpoint_file,
            contract=contract,
            step=step,
            last_validation_step=last_validation_step,
            cumulative_training_time_ms=cumulative_training_time_ms,
            base_model=base_model,
            optimizers=optimizers,
            train_loader=train_loader,
        )
        log0(
            f"checkpoint_saved:path={checkpoint_file} step:{step} "
            f"target:{args.iterations} bytes:{checkpoint_bytes} "
            f"sha256:{checkpoint_sha256}"
        )
''',
        "global-step training segments and lossless checkpoint",
    )
    return (
        "# SPDX-License-Identifier: MIT\n"
        "# Derived from the pinned OpenAI parameter-golf source; see ../LICENSE.\n\n"
        + source
    )


def prepare(destination: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    official = manifest["official_parameter_golf"]
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"destination is not empty: {destination}; choose a new directory"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _materialize_official(official, destination)

    base_path = destination / official["path"]
    base_bytes = base_path.read_bytes()
    actual_blob = _git_blob_sha1(base_bytes)
    if actual_blob != official["blob_sha1"]:
        raise RuntimeError(
            f"official base blob mismatch: expected {official['blob_sha1']}, got {actual_blob}"
        )
    adapted = adapt_harness(base_bytes.decode("utf-8"))

    downloader_path = destination / official["data_downloader_path"]
    downloader_bytes = downloader_path.read_bytes()
    actual_downloader_blob = _git_blob_sha1(downloader_bytes)
    if actual_downloader_blob != official["data_downloader_blob_sha1"]:
        raise RuntimeError(
            "official data downloader blob mismatch: expected "
            f"{official['data_downloader_blob_sha1']}, got {actual_downloader_blob}"
        )
    dataset_reference = manifest["dataset_reference"]
    adapted_downloader = adapt_downloader(
        downloader_bytes.decode("utf-8"),
        repository=dataset_reference["repository"],
        revision=dataset_reference["revision"],
        remote_root_prefix=dataset_reference["remote_root_prefix"],
    )

    ours = destination / "ours"
    ours.mkdir(exist_ok=True)
    harness_path = ours / "train_gpt_hybrid.py"
    harness_path.write_text(adapted, encoding="utf-8", newline="\n")
    downloader_output = ours / "download_data.py"
    downloader_output.write_text(
        adapted_downloader, encoding="utf-8", newline="\n"
    )
    shutil.copy2(HERE / "thetascan_benchmark_adapter.py", ours)
    theta_license_dir = ours / "thetascan-license"
    theta_license_dir.mkdir()
    for source_name in (
        "LICENSE",
        "NOTICE",
        "PATENTS.md",
        "LICENSING.md",
        "THIRD_PARTY_NOTICES.md",
    ):
        shutil.copy2(PROJECT_ROOT / source_name, theta_license_dir / source_name)
    shutil.copytree(PROJECT_ROOT / "licenses", theta_license_dir / "licenses")

    generated = dict(manifest)
    generated["thetascan_source"] = _thetascan_source_identity()
    generated["generated_sha256"] = {
        "train_gpt_hybrid.py": hashlib.sha256(adapted.encode("utf-8")).hexdigest(),
        "download_data.py": hashlib.sha256(
            adapted_downloader.encode("utf-8")
        ).hexdigest(),
        "thetascan_benchmark_adapter.py": hashlib.sha256(
            (HERE / "thetascan_benchmark_adapter.py").read_bytes()
        ).hexdigest(),
    }
    (ours / "ADAPTER-MANIFEST.json").write_text(
        json.dumps(generated, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"prepared: {destination}")
    print(f"entrypoint: {harness_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new or empty checkout directory")
    args = parser.parse_args()
    prepare(args.destination.resolve())


if __name__ == "__main__":
    main()
