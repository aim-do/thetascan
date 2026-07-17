from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "parameter-golf"


def _load_runner() -> types.ModuleType:
    path = BENCHMARK / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("thetascan_parameter_golf_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_adapter() -> types.ModuleType:
    path = BENCHMARK / "thetascan_benchmark_adapter.py"
    spec = importlib.util.spec_from_file_location("thetascan_parameter_golf_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_remote_suite() -> types.ModuleType:
    path = BENCHMARK / "remote_suite.py"
    spec = importlib.util.spec_from_file_location("thetascan_parameter_golf_remote", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load remote suite from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParameterGolfProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner()
        cls.adapter = _load_adapter()
        cls.remote = _load_remote_suite()
        cls.manifest = json.loads(
            (BENCHMARK / "adapter_manifest.json").read_text(encoding="utf-8")
        )

    def _args(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            data_path=None,
            tokenizer_path=None,
            iterations=20,
            segment_start_step=0,
            segment_end_step=20,
            val_start_step=0,
            val_every=20,
            warmup_steps=5,
            warmdown_iters=5,
            max_wallclock_seconds=0.0,
            optimizer_policy="muon-2d",
            projection_layout="mamba-shared",
            recipe="gn-reference-v0.1",
            backend="quad",
            expected_train_shards=16,
            checkpoint_output=None,
            resume_checkpoint=None,
            resume_checkpoint_sha256="",
            resume_checkpoint_bytes=0,
        )

    def test_runner_uses_official_16mb_ffn_profile(self) -> None:
        attention = self.runner._base_environment(
            self._args(), 7, "attention", None
        )
        mamba = self.runner._base_environment(self._args(), 7, "mamba3", None)
        theta = self.runner._base_environment(
            self._args(), 7, "thetascan", "gn-reference-v0.1"
        )

        self.assertEqual(attention["MODEL_DIM"], "512")
        self.assertEqual(attention["MLP_HIDDEN"], "1024")
        self.assertEqual(theta["MLP_HIDDEN"], "1024")
        self.assertEqual(theta["THETA_MLP_HIDDEN"], "1023")
        self.assertEqual(theta["THETA_PROJECTION_LAYOUT"], "mamba-shared")
        self.assertEqual(theta["THETA_RECIPE"], "gn-reference-v0.1")
        self.assertNotIn("THETA_RECIPE", attention)
        self.assertNotIn("THETA_RECIPE", mamba)
        self.assertEqual(mamba["MLP_HIDDEN"], "1024")
        self.assertEqual(mamba["MAMBA3_MLP_HIDDEN"], "938")
        self.assertEqual(self.manifest["default_full_ffn_hidden"], 1024)
        self.assertEqual(self.manifest["mamba3_parity_ffn_hidden"], 938)

    def test_mamba_parity_arithmetic_and_exact_counts(self) -> None:
        parity = self.manifest["expected_mamba3_parity_delta"]
        full = self.manifest["default_full_ffn_hidden"]
        mamba = self.manifest["mamba3_parity_ffn_hidden"]
        expected_ffn_delta = 2 * 2 * 512 * (mamba - full)
        expected_net = parity["mixer_two_blocks"] + expected_ffn_delta

        self.assertEqual(expected_ffn_delta, -176_128)
        self.assertEqual(parity["ffn_two_blocks"], expected_ffn_delta)
        self.assertEqual(parity["net"], expected_net)
        self.assertEqual(parity["pinned_attention_model_params"], 17_059_912)
        self.assertEqual(
            parity["pinned_mamba3_model_params"],
            parity["pinned_attention_model_params"] + expected_net,
        )

    def test_runner_resolves_exactly_the_five_retained_variants(self) -> None:
        for arm in ("attention", "mamba3"):
            with self.subTest(arm=arm):
                self.assertEqual(
                    self.runner._run_spec(arm, "preset", None),
                    (arm, None, "n/a"),
                )
                with self.assertRaisesRegex(ValueError, "do not accept"):
                    self.runner._run_spec(
                        arm, "preset", "gn-reference-v0.1"
                    )
        for recipe in self.runner.RECIPES:
            with self.subTest(recipe=recipe):
                self.assertEqual(
                    self.runner._run_spec("thetascan", "preset", recipe),
                    ("thetascan", recipe, "partial"),
                )

    def test_suite_and_theta_optimizer_contract_match_manifest(self) -> None:
        runs = {}
        for path in sorted((BENCHMARK / "configs" / "runpod").glob("*.json")):
            config = json.loads(path.read_text(encoding="utf-8"))
            runs.update({run["id"]: run for run in config["runs"]})
        parity = self.manifest["expected_mamba3_parity_delta"]
        theta = self.manifest["expected_theta_profiles"]["mamba-shared"]

        self.assertEqual(
            runs["mamba3-parity"]["expected_model_params"],
            parity["pinned_mamba3_model_params"],
        )
        self.assertEqual(
            runs["attention-official"]["expected_model_params"],
            parity["pinned_attention_model_params"],
        )
        for run_id in ("gn-reference-3000", "gn-expanded-3000"):
            self.assertEqual(runs[run_id]["expected_model_params"], theta["model_params"])
            self.assertEqual(
                runs[run_id]["expected_theta_headwise_muon_params"],
                theta["theta_candidate_parameter_count"],
            )
        self.assertEqual(
            runs["kernel-expanded-3000"]["expected_model_params"],
            17_059_976,
        )
        self.assertEqual(
            runs["kernel-expanded-3000"]["expected_theta_headwise_muon_params"],
            theta["theta_candidate_parameter_count"],
        )

    def test_all_theta_projection_layouts_are_parameter_matched(self) -> None:
        profiles = self.manifest["expected_theta_profiles"]
        expected_hidden = {
            "mamba-shared": 1023,
            "transformer-gqa": 832,
            "independent": 576,
        }
        for layout, hidden in expected_hidden.items():
            with self.subTest(layout=layout):
                args = self._args()
                args.projection_layout = layout
                env = self.runner._base_environment(
                    args, 7, "thetascan", "gn-reference-v0.1"
                )
                self.assertEqual(env["THETA_PROJECTION_LAYOUT"], layout)
                self.assertEqual(env["THETA_MLP_HIDDEN"], str(hidden))
                self.assertEqual(profiles[layout]["swapped_ffn_hidden"], hidden)
                self.assertEqual(profiles[layout]["model_params"], 17_059_928)
                self.assertEqual(profiles[layout]["model_delta_vs_attention"], 16)

    def test_adapter_builds_all_projection_layouts_for_gn_and_kernel(self) -> None:
        expected_families = {
            "gn-reference-v0.1": "gn",
            "kernel-expanded-reference-v0.1": "kernel",
        }
        for recipe, family in expected_families.items():
            for layout in ("mamba-shared", "transformer-gqa", "independent"):
                with self.subTest(recipe=recipe, layout=layout):
                    mixer = self.adapter.build_thetascan_mixer(
                        512,
                        recipe=recipe,
                        rope_mode="partial",
                        backend="quad",
                        projection_layout=layout,
                    )
                    resolved = mixer._parameter_golf_config["resolved"]
                    self.assertEqual(resolved["family"], family)
                    self.assertEqual(
                        resolved["share_key_query"], layout == "mamba-shared"
                    )
                    self.assertEqual(
                        resolved["key_value_heads"],
                        4 if layout == "transformer-gqa" else None,
                    )

    def test_adapter_builds_exactly_the_three_public_recipes(self) -> None:
        expected = {
            "gn-reference-v0.1": (
                787_472, "both_feature_mass", "bank", 1
            ),
            "gn-expanded-reference-v0.1": (
                790_544, "both_feature_mass", "bank", 2
            ),
            "kernel-expanded-reference-v0.1": (
                787_496, "key_mass", "bank", 2
            ),
        }
        self.assertEqual(set(self.runner.RECIPES), set(expected))
        self.assertEqual(set(self.adapter.RECIPES), set(expected))
        for recipe, (
            count,
            normalization,
            temporal,
            feature_expansion,
        ) in expected.items():
            with self.subTest(recipe=recipe):
                mixer = self.adapter.build_thetascan_mixer(
                    512,
                    recipe=recipe,
                    rope_mode="partial",
                    backend="quad",
                    projection_layout="mamba-shared",
                    layer_index=4,
                )
                self.assertEqual(sum(p.numel() for p in mixer.parameters()), count)
                resolved = mixer._parameter_golf_config["resolved"]
                family = resolved["family"]
                self.assertEqual(
                    resolved["gn" if family == "gn" else "kernel"]["read_normalization"],
                    normalization,
                )
                self.assertEqual(resolved["temporal"]["mode"], temporal)
                self.assertEqual(resolved["feature_expansion"], feature_expansion)
                if feature_expansion > 1:
                    self.assertEqual(
                        resolved["expansion_key"], "thetascan:layer-4"
                    )

    def test_runpod_configs_are_one_run_each_with_safe_caps(self) -> None:
        config_paths = sorted((BENCHMARK / "configs" / "runpod").glob("*.json"))
        expected_runs = {
            "01-attention-official.json": ("attention-official", "attention", None),
            "02-mamba3-parity.json": ("mamba3-parity", "mamba3", None),
            "05-gn-reference-3000.json": (
                "gn-reference-3000", "thetascan", "gn-reference-v0.1"
            ),
            "06-gn-expanded-3000.json": (
                "gn-expanded-3000", "thetascan", "gn-expanded-reference-v0.1"
            ),
            "07-kernel-expanded-3000.json": (
                "kernel-expanded-3000", "thetascan", "kernel-expanded-reference-v0.1"
            ),
            "08-attention-continue-3000-to-4000.json": (
                "attention-official", "attention", None
            ),
            "09-mamba3-continue-3000-to-4000.json": (
                "mamba3-parity", "mamba3", None
            ),
            "10-gn-reference-continue-3000-to-4000.json": (
                "gn-reference-3000", "thetascan", "gn-reference-v0.1"
            ),
            "11-gn-expanded-continue-3000-to-4000.json": (
                "gn-expanded-3000", "thetascan", "gn-expanded-reference-v0.1"
            ),
            "12-kernel-expanded-continue-3000-to-4000.json": (
                "kernel-expanded-3000", "thetascan", "kernel-expanded-reference-v0.1"
            ),
            "13-attention-continue-4000-to-7500.json": (
                "attention-official", "attention", None
            ),
            "14-mamba3-continue-4000-to-7500.json": (
                "mamba3-parity", "mamba3", None
            ),
            "15-gn-reference-continue-4000-to-7500.json": (
                "gn-reference-3000", "thetascan", "gn-reference-v0.1"
            ),
            "16-gn-expanded-continue-4000-to-7500.json": (
                "gn-expanded-3000", "thetascan", "gn-expanded-reference-v0.1"
            ),
            "17-kernel-expanded-continue-4000-to-7500.json": (
                "kernel-expanded-3000", "thetascan", "kernel-expanded-reference-v0.1"
            ),
        }
        # Runs whose int8 artifact exceeds the 16MB cap at every stage
        # (research-only expanded arms).
        oversize_runs = {"gn-expanded-3000", "kernel-expanded-3000"}
        self.assertEqual({path.name for path in config_paths}, set(expected_runs))
        suite_ids: set[str] = set()
        for path in config_paths:
            with self.subTest(config=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(len(config["runs"]), 1)
                expected_id, expected_arm, expected_recipe = expected_runs[path.name]
                self.assertEqual(config["runs"][0]["id"], expected_id)
                suite_ids.add(config["suite_id"])
                self.assertEqual(config["seed_policy"], "explicit_shared_across_pods")
                run = config["runs"][0]
                self.assertEqual(run["arm"], expected_arm)
                if expected_recipe is None:
                    self.assertNotIn("recipe", run)
                else:
                    self.assertEqual(run["recipe"], expected_recipe)
                    self.assertIn(run["recipe"], self.runner.RECIPES)
                self.assertEqual(run["artifact_cap_bytes"], 16_000_000)
                self.assertIn("final_int8_zlib_roundtrip_exact", run["required_log_regex"])
                self.assertIn(
                    "Total submission size int8\\+zlib: [0-9]+ bytes",
                    run["required_log_regex"],
                )
                smoke = config["protocols"]["smoke"]
                production = config["protocols"]["production"]
                self.assertEqual(production["val_every"], 250)
                self.assertEqual(smoke["iterations"], 7500)
                self.assertEqual(production["iterations"], 7500)
                self.assertEqual(production["warmdown_iters"], 750)
                self.assertEqual(production["max_wallclock_seconds"], 0.0)
                self.assertTrue(smoke["checkpoint_at_end"])
                self.assertTrue(production["checkpoint_at_end"])
                if run["id"] in oversize_runs:
                    self.assertTrue(smoke["research_allow_oversize_artifact"])
                    self.assertTrue(
                        production["research_allow_oversize_artifact"]
                    )
                else:
                    self.assertNotIn("research_allow_oversize_artifact", smoke)
                    self.assertNotIn(
                        "research_allow_oversize_artifact", production
                    )
                if path.name[:2] in {"01", "02", "05", "06", "07"}:
                    self.assertEqual(smoke["timeout_seconds"], 900)
                    self.assertEqual(smoke["max_wallclock_seconds"], 600.0)
                    self.assertEqual(config["data"]["train_shards"], 16)
                    self.assertEqual(smoke["segment_start_step"], 0)
                    self.assertEqual(smoke["segment_end_step"], 20)
                    self.assertEqual(production["segment_start_step"], 20)
                    self.assertEqual(production["segment_end_step"], 3000)
                    self.assertEqual(production["val_start_step"], 1000)
                elif path.name[:2] in {"08", "09", "10", "11", "12"}:
                    self.assertEqual(smoke["timeout_seconds"], 900)
                    self.assertEqual(smoke["max_wallclock_seconds"], 0.0)
                    self.assertEqual(config["data"]["train_shards"], 21)
                    self.assertEqual(smoke["segment_start_step"], 3000)
                    self.assertEqual(smoke["segment_end_step"], 3020)
                    self.assertEqual(smoke["val_start_step"], 3020)
                    self.assertEqual(smoke["val_every"], 20)
                    self.assertEqual(production["segment_start_step"], 3020)
                    self.assertEqual(production["segment_end_step"], 4000)
                    self.assertEqual(production["val_start_step"], 3250)
                else:
                    self.assertEqual(smoke["timeout_seconds"], 1000)
                    self.assertEqual(smoke["max_wallclock_seconds"], 0.0)
                    self.assertEqual(config["data"]["train_shards"], 40)
                    self.assertEqual(smoke["segment_start_step"], 4000)
                    self.assertEqual(smoke["segment_end_step"], 4020)
                    self.assertEqual(smoke["val_start_step"], 4020)
                    self.assertEqual(smoke["val_every"], 20)
                    self.assertEqual(production["segment_start_step"], 4020)
                    self.assertEqual(production["segment_end_step"], 7500)
                    self.assertEqual(production["val_start_step"], 4250)
        self.assertEqual(len(suite_ids), len(expected_runs))

    def test_remote_suite_requires_exact_roundtrip_and_16mb_artifact(self) -> None:
        valid_log = (
            "model_params:17059928\n"
            "Total submission size int8+zlib: 15999999 bytes\n"
            "final_int8_zlib_roundtrip_exact val_loss:1.23456789 "
            "val_bpb:1.01234567\n"
        )
        metrics = self.remote._parse_metrics(valid_log)
        self.assertEqual(metrics["artifact_bytes"], 15_999_999)
        self.assertEqual(metrics["artifact_cap_bytes"], 16_000_000)
        self.assertTrue(metrics["artifact_within_cap"])
        self.assertTrue(metrics["final_int8_roundtrip_exact"])
        self.assertEqual(self.remote._artifact_contract_failures(metrics), [])

        oversized = self.remote._parse_metrics(
            valid_log.replace("15999999 bytes", "16000001 bytes")
        )
        failures = self.remote._artifact_contract_failures(oversized)
        self.assertFalse(oversized["artifact_within_cap"])
        self.assertTrue(any("exceeds" in failure for failure in failures))
        self.assertEqual(
            self.remote._artifact_contract_failures(
                oversized, allow_oversize=True
            ),
            [],
        )

        inexact = self.remote._parse_metrics(
            valid_log.replace("final_int8_zlib_roundtrip_exact", "final_int8_zlib_roundtrip")
        )
        failures = self.remote._artifact_contract_failures(inexact)
        self.assertTrue(any("exact" in failure for failure in failures))

    def test_remote_suite_resumes_production_from_smoke_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "checkout"
            export = root / "export"
            ours = checkout / "ours"
            ours.mkdir(parents=True)
            export.mkdir()
            (ours / "ADAPTER-MANIFEST.json").write_text(
                json.dumps({"test": True}), encoding="utf-8"
            )
            runner = root / "fake_runner.py"
            runner.write_text(
                '''import hashlib
import sys
from pathlib import Path

args = sys.argv[1:]
checkout = Path(args[0])
def value(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default

start = int(value("--segment-start-step", "0"))
end = int(value("--segment-end-step"))
target = int(value("--iterations"))
resume = value("--resume-checkpoint")
if start:
    assert resume is not None
    resume_path = Path(resume)
    blob = resume_path.read_bytes()
    assert len(blob) == int(value("--resume-checkpoint-bytes"))
    assert hashlib.sha256(blob).hexdigest() == value("--resume-checkpoint-sha256")
output = Path(value("--checkpoint-output"))
output.write_bytes(f"checkpoint-at-{end}".encode())
blob = output.read_bytes()
digest = hashlib.sha256(blob).hexdigest()
print("model_params:1")
print(f"step:{end}/{target} val_loss:1.0000 val_bpb:1.0000 train_time:1ms step_avg:1ms")
print(f"checkpoint_saved:path={output} step:{end} target:{target} bytes:{len(blob)} sha256:{digest}")
print("Total submission size int8+zlib: 100 bytes")
print("final_int8_zlib_roundtrip_exact val_loss:1.00000000 val_bpb:1.00000000")
(checkout / "final_model.int8.ptz").write_bytes(b"artifact")
''',
                encoding="utf-8",
            )
            config = {
                "schema_version": 1,
                "suite_id": "checkpoint-chain-test",
                "data": {"variant": "sp1024", "train_shards": 1},
                "protocols": {
                    "smoke": {
                        "iterations": 100,
                        "segment_start_step": 0,
                        "segment_end_step": 2,
                        "val_start_step": 0,
                        "val_every": 2,
                        "warmup_steps": 0,
                        "warmdown_iters": 10,
                        "max_wallclock_seconds": 0,
                        "timeout_seconds": 30,
                        "checkpoint_at_end": True,
                    },
                    "production": {
                        "iterations": 100,
                        "segment_start_step": 2,
                        "segment_end_step": 4,
                        "val_start_step": 3,
                        "val_every": 1,
                        "warmup_steps": 0,
                        "warmdown_iters": 10,
                        "max_wallclock_seconds": 0,
                        "timeout_seconds": 30,
                        "checkpoint_at_end": True,
                    },
                },
                "runs": [{
                    "id": "attention",
                    "arm": "attention",
                    "rope": "partial",
                    "backend": "quad",
                    "projection_layout": "mamba-shared",
                    "optimizer_policy": "muon-2d",
                    "expected_model_params": 1,
                    "artifact_cap_bytes": 16_000_000,
                    "required_log_regex": [],
                }],
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            argv = [
                "remote_suite.py",
                "--config", str(config_path),
                "--checkout", str(checkout),
                "--benchmark-runner", str(runner),
                "--export-dir", str(export),
                "--seed", "7",
                "--source-sha256", "a" * 64,
                "--git-dirty", "false",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(self.remote.time, "sleep", return_value=None),
            ):
                self.assertEqual(self.remote.main(), 0)
            metadata = json.loads(
                (export / "research-checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["step"], 4)
            self.assertEqual(metadata["grad_accum_steps"], 8)
            self.assertEqual(
                hashlib.sha256((export / "research-checkpoint.pt").read_bytes()).hexdigest(),
                metadata["sha256"],
            )
            production_log = (export / "production-attention.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("--resume-checkpoint", production_log)
            self.assertIn("--grad-accum-steps 8", production_log)

    def test_remote_suite_chains_external_resume_through_smoke_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "checkout"
            export = root / "export"
            input_dir = root / "input"
            (checkout / "ours").mkdir(parents=True)
            export.mkdir()
            input_dir.mkdir()
            (checkout / "ours" / "ADAPTER-MANIFEST.json").write_text(
                json.dumps({"test": True}), encoding="utf-8"
            )
            external = input_dir / "research-checkpoint.pt"
            external.write_bytes(b"external-step-2")
            external_sha = hashlib.sha256(external.read_bytes()).hexdigest()
            config = {
                "schema_version": 1,
                "suite_id": "external-checkpoint-chain-test",
                "data": {"variant": "sp1024", "train_shards": 1},
                "protocols": {
                    "smoke": {
                        "iterations": 10,
                        "segment_start_step": 2,
                        "segment_end_step": 3,
                        "val_start_step": 3,
                        "val_every": 1,
                        "warmup_steps": 0,
                        "warmdown_iters": 2,
                        "max_wallclock_seconds": 0,
                        "timeout_seconds": 30,
                        "checkpoint_at_end": True,
                    },
                    "production": {
                        "iterations": 10,
                        "segment_start_step": 3,
                        "segment_end_step": 4,
                        "val_start_step": 4,
                        "val_every": 1,
                        "warmup_steps": 0,
                        "warmdown_iters": 2,
                        "max_wallclock_seconds": 0,
                        "timeout_seconds": 30,
                        "checkpoint_at_end": True,
                    },
                },
                "runs": [{
                    "id": "attention",
                    "arm": "attention",
                    "rope": "partial",
                    "backend": "quad",
                    "projection_layout": "mamba-shared",
                    "optimizer_policy": "muon-2d",
                    "expected_model_params": 1,
                    "artifact_cap_bytes": 16_000_000,
                    "required_log_regex": [],
                }],
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            runner = root / "unused-runner.py"
            runner.write_text("raise SystemExit(99)\n", encoding="utf-8")
            observed: list[tuple[str, Path, bytes]] = []

            def fake_run_one(**kwargs):
                resume_path = kwargs["resume_checkpoint"]
                self.assertIsNotNone(resume_path)
                resume_path = Path(resume_path)
                observed.append(
                    (kwargs["stage"], resume_path, resume_path.read_bytes())
                )
                output = kwargs["export_dir"] / self.remote.CHECKPOINT_FILENAME
                output.write_bytes(f"{kwargs['stage']}-output".encode())
                blob = output.read_bytes()
                metadata = {
                    "schema_version": 1,
                    "filename": self.remote.CHECKPOINT_FILENAME,
                    "step": int(kwargs["protocol"]["segment_end_step"]),
                    "target_iterations": int(kwargs["protocol"]["iterations"]),
                    "grad_accum_steps": self.remote._resolved_grad_accum_steps(
                        kwargs["protocol"]
                    ),
                    "bytes": len(blob),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "run_id": "attention",
                    "runtime_seed": kwargs["seed"],
                }
                self.remote._write_json(
                    kwargs["export_dir"] / self.remote.CHECKPOINT_METADATA_FILENAME,
                    metadata,
                )
                return {"id": "attention", "stage": kwargs["stage"], "status": "passed"}

            argv = [
                "remote_suite.py",
                "--config", str(config_path),
                "--checkout", str(checkout),
                "--benchmark-runner", str(runner),
                "--export-dir", str(export),
                "--seed", "7",
                "--source-sha256", "a" * 64,
                "--git-dirty", "false",
                "--resume-checkpoint", str(external),
                "--resume-checkpoint-sha256", external_sha,
                "--resume-checkpoint-bytes", str(external.stat().st_size),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(self.remote, "_run_one", side_effect=fake_run_one),
            ):
                self.assertEqual(self.remote.main(), 0)
            self.assertEqual(observed[0][0], "smoke")
            self.assertEqual(observed[0][1], external.resolve())
            self.assertEqual(observed[0][2], b"external-step-2")
            self.assertEqual(observed[1][0], "production")
            self.assertEqual(observed[1][1], (export / "research-checkpoint.pt").resolve())
            self.assertEqual(observed[1][2], b"smoke-output")

    def test_checkpoint_runtime_metadata_is_weights_only_safe(self) -> None:
        generator = (BENCHMARK / "prepare_harness.py").read_text(encoding="utf-8")
        self.assertIn("kernel_relu2_threshold_per_head:", generator)
        self.assertIn("bank_timescales_per_head:", generator)
        self.assertIn("gn_threshold_stats:", generator)
        self.assertIn('"torch": str(torch.__version__)', generator)
        self.assertIn(
            "None if torch.version.cuda is None else str(torch.version.cuda)",
            generator,
        )
        runtime = {
            "torch": str(torch.__version__),
            "torch_cuda": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safe-runtime.pt"
            torch.save({"runtime": runtime}, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(loaded["runtime"], runtime)
        self.assertIs(type(loaded["runtime"]["torch"]), str)

    def test_exact_resume_contract_authenticates_data_and_accumulation(self) -> None:
        generator = (BENCHMARK / "prepare_harness.py").read_text(encoding="utf-8")
        self.assertIn('"sha256": sha256_file(file)', generator)
        self.assertIn('"grad_accum_steps": grad_accum_steps', generator)
        self.assertIn(
            'if saved_prefix != self.metadata[: len(saved_prefix)]:', generator
        )

    def test_failed_result_schema_allows_pre_run_failure_only(self) -> None:
        schema = json.loads(
            (BENCHMARK / "results" / "result.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("minItems", schema["properties"]["runs"])
        passed_rule = schema["allOf"][0]
        self.assertEqual(
            passed_rule["if"]["properties"]["status"]["const"], "passed"
        )
        self.assertEqual(
            passed_rule["then"]["properties"]["runs"]["minItems"], 1
        )


if __name__ == "__main__":
    unittest.main()
