from __future__ import annotations

import base64
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "benchmarks" / "parameter-golf" / "run_runpod.py"
SPEC = importlib.util.spec_from_file_location("thetascan_run_runpod", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
runpod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runpod)


def _config() -> dict:
    return {
        "schema_version": 1,
        "suite_id": "unit-attention",
        "seed_policy": "fresh_shared_per_suite",
        "hardware": {"gpu_count": 1, "gpu_type": runpod.GPU_TYPE_ID},
        "data": {"variant": "sp1024", "train_shards": 16},
        "protocols": {
            "smoke": {
                "iterations": 20,
                "val_every": 20,
                "warmup_steps": 5,
                "warmdown_iters": 5,
                "max_wallclock_seconds": 600,
                "timeout_seconds": 900,
            },
            "production": {
                "iterations": 1000,
                "val_every": 250,
                "warmup_steps": 20,
                "warmdown_iters": 200,
                "max_wallclock_seconds": 1500,
                "timeout_seconds": 1800,
            },
        },
        "runs": [
            {
                "id": "attention",
                "arm": "attention",
                "rope": "partial",
                "backend": "quad",
                "optimizer_policy": "muon-2d",
                "expected_model_params": 17_059_912,
                "artifact_cap_bytes": 16_000_000,
                "required_log_regex": [
                    "model_params:17059912",
                    "final_int8_zlib_roundtrip_exact",
                    "Total submission size int8\\+zlib: [0-9]+ bytes",
                ],
            }
        ],
    }


class RunPodLauncherTests(unittest.TestCase):
    def _config_path(self, directory: str) -> Path:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(_config()), encoding="utf-8")
        return path

    def _resume_fixture(self, directory: str, *, seed: int = 123456):
        root = Path(directory)
        value = _config()
        value["protocols"] = {
            "smoke": {
                "iterations": 7500,
                "segment_start_step": 3000,
                "segment_end_step": 3020,
                "val_start_step": 3020,
                "val_every": 20,
                "warmup_steps": 20,
                "warmdown_iters": 750,
                "max_wallclock_seconds": 600,
                "timeout_seconds": 900,
                "checkpoint_at_end": True,
            },
            "production": {
                "iterations": 7500,
                "segment_start_step": 3020,
                "segment_end_step": 4000,
                "val_start_step": 3250,
                "val_every": 250,
                "warmup_steps": 20,
                "warmdown_iters": 750,
                "max_wallclock_seconds": 1800,
                "timeout_seconds": 2400,
                "checkpoint_at_end": True,
            },
        }
        config_path = root / "resume-config.json"
        config_path.write_text(json.dumps(value), encoding="utf-8")
        checkpoint = root / "research-checkpoint.pt"
        checkpoint.write_bytes(b"external-checkpoint")
        sha256 = runpod._file_sha256(checkpoint)
        metadata = {
            "schema_version": 1,
            "filename": checkpoint.name,
            "step": 3000,
            "target_iterations": 7500,
            "grad_accum_steps": 8,
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256,
            "run_id": "attention",
            "runtime_seed": seed,
        }
        (root / "research-checkpoint.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return config_path, checkpoint, metadata

    def test_config_is_exact_h100_and_one_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._config_path(directory)
            _, run = runpod._validate_config(path)
            self.assertEqual(run["id"], "attention")

            value = _config()
            value["runs"].append(dict(value["runs"][0], id="attention-2"))
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(runpod.LauncherError, "exactly one run"):
                runpod._validate_config(path)

    def test_config_recipe_contract_matches_the_five_retained_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"

            for arm in ("attention", "mamba3"):
                with self.subTest(arm=arm):
                    value = _config()
                    value["runs"][0]["arm"] = arm
                    value["runs"][0].pop("recipe", None)
                    path.write_text(json.dumps(value), encoding="utf-8")
                    _, run = runpod._validate_config(path)
                    self.assertNotIn("recipe", run)

                    value["runs"][0]["recipe"] = "gn-reference-v0.1"
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaisesRegex(runpod.LauncherError, "recipe"):
                        runpod._validate_config(path)

            for recipe in (
                "gn-reference-v0.1",
                "gn-expanded-reference-v0.1",
                "kernel-expanded-reference-v0.1",
            ):
                with self.subTest(recipe=recipe):
                    value = _config()
                    value["runs"][0].update(arm="thetascan", recipe=recipe)
                    path.write_text(json.dumps(value), encoding="utf-8")
                    _, run = runpod._validate_config(path)
                    self.assertEqual(run["recipe"], recipe)

            value = _config()
            value["runs"][0]["arm"] = "thetascan"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(runpod.LauncherError, "recipe"):
                runpod._validate_config(path)

            value["runs"][0]["recipe"] = "retired-recipe"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(runpod.LauncherError, "recipe"):
                runpod._validate_config(path)

    def test_payload_is_deterministic_and_source_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._config_path(directory)
            first = runpod._build_payload(path)
            second = runpod._build_payload(path)
            self.assertEqual(first, second)
            self.assertLessEqual(len(runpod._pod_environment(first)), 45)
            with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
                names = archive.getnames()
            self.assertIn("suite-config.json", names)
            self.assertIn("src/thetascan/__init__.py", names)
            self.assertFalse(any("__pycache__" in name or name.endswith(".pyc") for name in names))

    def test_external_resume_is_verified_against_smoke_target_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path, checkpoint, metadata = self._resume_fixture(directory)
            config, run = runpod._validate_config(config_path)
            verified = runpod._validate_resume_checkpoint(
                checkpoint, config=config, run=run, seed=123456
            )
            self.assertEqual(verified["step"], 3000)
            self.assertEqual(verified["sha256"], metadata["sha256"])
            self.assertEqual(verified["grad_accum_steps"], 8)

            metadata["runtime_seed"] = 7
            (checkpoint.parent / "research-checkpoint.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(runpod.LauncherError, "runtime_seed"):
                runpod._validate_resume_checkpoint(
                    checkpoint, config=config, run=run, seed=123456
                )

            metadata["runtime_seed"] = 123456
            metadata["grad_accum_steps"] = 4
            (checkpoint.parent / "research-checkpoint.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(runpod.LauncherError, "grad_accum_steps"):
                runpod._validate_resume_checkpoint(
                    checkpoint, config=config, run=run, seed=123456
                )

    def test_checkpoint_continuation_requires_same_effective_accumulation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path, _, _ = self._resume_fixture(directory)
            value = json.loads(config_path.read_text(encoding="utf-8"))
            value["protocols"]["smoke"]["grad_accum_steps"] = 8
            value["protocols"]["production"]["grad_accum_steps"] = 4
            config_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(runpod.LauncherError, "grad_accum_steps"):
                runpod._validate_config(config_path)

    def test_oversize_research_exception_is_boolean_and_expanded_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            value = _config()
            value["runs"][0].update(
                arm="thetascan", recipe="gn-expanded-reference-v0.1"
            )
            for protocol in value["protocols"].values():
                protocol["research_allow_oversize_artifact"] = True
            path.write_text(json.dumps(value), encoding="utf-8")
            runpod._validate_config(path)

            value["runs"][0]["recipe"] = "gn-reference-v0.1"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(runpod.LauncherError, "expanded research"):
                runpod._validate_config(path)

            value["runs"][0]["recipe"] = "kernel-expanded-reference-v0.1"
            value["protocols"]["smoke"][
                "research_allow_oversize_artifact"
            ] = "yes"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(runpod.LauncherError, "must be boolean"):
                runpod._validate_config(path)

    def test_external_resume_hash_failure_precedes_billable_operations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path, checkpoint, _ = self._resume_fixture(directory)
            checkpoint.write_bytes(b"tampered")
            with mock.patch.object(runpod, "_load_api_key") as load_key:
                with self.assertRaisesRegex(runpod.LauncherError, "size mismatch"):
                    runpod.main(
                        [
                            "launch",
                            "--config",
                            str(config_path),
                            "--seed",
                            "123456",
                            "--resume-checkpoint",
                            str(checkpoint),
                            "--yes",
                        ]
                    )
            load_key.assert_not_called()

    def test_continuation_requires_resume_before_payload_or_billing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path, _, _ = self._resume_fixture(directory)
            with (
                mock.patch.object(runpod, "_build_payload") as build_payload,
                mock.patch.object(runpod, "_load_api_key") as load_key,
            ):
                with self.assertRaisesRegex(
                    runpod.LauncherError, "--resume-checkpoint is required"
                ):
                    runpod.main(
                        [
                            "launch",
                            "--config",
                            str(config_path),
                            "--seed",
                            "123456",
                            "--yes",
                        ]
                    )
            build_payload.assert_not_called()
            load_key.assert_not_called()

    def test_startup_has_two_self_delete_paths_and_isolates_exports(self) -> None:
        startup = runpod._startup_script(
            run={"arm": "attention"},
            data={"variant": "sp1024", "train_shards": 16},
            seed=123456,
            source_sha256="a" * 64,
            git_revision="b" * 40,
            git_dirty="true",
            hard_timeout_seconds=3900,
            result_grace_seconds=300,
        )
        self.assertIn("RUNPOD_POD_ID", startup)
        self.assertIn("RUNPOD_API_KEY", startup)
        self.assertIn("https://rest.runpod.io/v1/pods/{pod_id}", startup)
        self.assertIn('method="DELETE"', startup)
        self.assertIn("while True:", startup)
        self.assertIn("absolute_watchdog", startup)
        self.assertIn('sleep "$HARD_TIMEOUT_SECONDS"', startup)
        self.assertGreaterEqual(startup.count("self_delete"), 3)
        self.assertIn('--directory "$EXPORT_ROOT"', startup)
        self.assertIn("export HF_HUB_DISABLE_XET=1", startup)
        self.assertIn("export HF_HUB_DOWNLOAD_TIMEOUT=120", startup)
        self.assertIn("export HF_HUB_ETAG_TIMEOUT=30", startup)
        self.assertLess(
            startup.index("export HF_HUB_DISABLE_XET=1"),
            startup.index("download_data.py"),
        )
        self.assertIn("--train-shards 16", startup)
        self.assertIn('/tmp/thetascan-payload.tar.gz', startup)
        self.assertNotIn('--directory "$SOURCE_ROOT"', startup)
        self.assertNotIn("timeout --signal", startup)

        command = runpod._docker_command(startup)
        encoded = command.split("printf %s ", 1)[1].split(" | base64", 1)[0]
        decoded = gzip.decompress(base64.b64decode(encoded)).decode("utf-8")
        self.assertEqual(decoded, startup)

    def test_startup_passes_external_resume_contract_to_remote_suite(self) -> None:
        startup = runpod._startup_script(
            run={"arm": "attention"},
            data={"variant": "sp1024", "train_shards": 16},
            seed=123456,
            source_sha256="a" * 64,
            git_revision=None,
            git_dirty="false",
            hard_timeout_seconds=3900,
            result_grace_seconds=300,
            resume_checkpoint_sha256="b" * 64,
            resume_checkpoint_bytes=135_000_000,
        )
        self.assertIn(
            f"--resume-checkpoint {runpod.REMOTE_RESUME_CHECKPOINT}", startup
        )
        self.assertIn("--resume-checkpoint-sha256 " + "b" * 64, startup)
        self.assertIn("--resume-checkpoint-bytes 135000000", startup)

    def test_external_resume_rejects_environment_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path, checkpoint, _ = self._resume_fixture(directory)
            with (
                mock.patch.object(
                    runpod,
                    "_select_bootstrap_transport",
                    return_value=("environment", "test fallback"),
                ),
                mock.patch.object(runpod, "_load_api_key") as load_key,
            ):
                with self.assertRaisesRegex(runpod.LauncherError, "requires SSH"):
                    runpod.main(
                        [
                            "launch",
                            "--config",
                            str(config_path),
                            "--seed",
                            "123456",
                            "--resume-checkpoint",
                            str(checkpoint),
                            "--dry-run",
                        ]
                    )
            load_key.assert_not_called()

    def test_dry_run_never_requires_credentials_or_creates_a_pod(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._config_path(directory)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = runpod.main(
                    [
                        "launch",
                        "--config",
                        str(path),
                        "--seed",
                        "123456",
                        "--dry-run",
                    ]
                )
            self.assertEqual(code, 0)
            preview = json.loads(output.getvalue())
            self.assertEqual(preview["runtime_seed"], 123456)
            self.assertEqual(preview["hard_timeout_seconds_from_boot"], 3900)
            self.assertEqual(preview["setup_headroom_seconds"], 900)
            self.assertEqual(preview["gpu"], f"1 x {runpod.GPU_TYPE_ID}")
            self.assertEqual(preview["image_name"], runpod.IMAGE_NAME)
            self.assertEqual(preview["template_id"], runpod.TEMPLATE_ID)

    def test_config_launch_guards_are_self_contained_and_cli_can_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = _config()
            value["launcher"] = {
                "hard_timeout_seconds": 3600,
                "result_grace_seconds": 300,
                "max_budget_usd": 3.25,
            }
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(value), encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                runpod.main(
                    ["launch", "--config", str(path), "--seed", "123456", "--dry-run"]
                )
            preview = json.loads(output.getvalue())
            self.assertEqual(preview["hard_timeout_seconds_from_boot"], 3600)
            self.assertEqual(preview["result_grace_seconds"], 300)
            self.assertEqual(preview["setup_headroom_seconds"], 600)
            self.assertEqual(preview["max_budget_usd"], 3.25)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                runpod.main(
                    [
                        "launch",
                        "--config",
                        str(path),
                        "--seed",
                        "123456",
                        "--hard-timeout-seconds",
                        "3900",
                        "--result-grace-seconds",
                        "200",
                        "--max-budget-usd",
                        "3.5",
                        "--dry-run",
                    ]
                )
            preview = json.loads(output.getvalue())
            self.assertEqual(preview["hard_timeout_seconds_from_boot"], 3900)
            self.assertEqual(preview["result_grace_seconds"], 200)
            self.assertEqual(preview["setup_headroom_seconds"], 1000)
            self.assertEqual(preview["max_budget_usd"], 3.5)

    def test_invalid_config_launch_guards_fail_before_billable_operations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = _config()
            value["launcher"] = {
                "hard_timeout_seconds": 3000,
                "result_grace_seconds": 3000,
                "max_budget_usd": 3.25,
            }
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(runpod, "_load_api_key") as load_key:
                with self.assertRaisesRegex(runpod.LauncherError, "grace"):
                    runpod.main(["launch", "--config", str(path), "--yes"])
            load_key.assert_not_called()

    def test_every_checked_in_public_config_dry_runs_with_its_own_guards(self) -> None:
        config_dir = ROOT / "benchmarks" / "parameter-golf" / "configs" / "runpod"
        paths = sorted(config_dir.glob("*.json"))
        self.assertGreaterEqual(len(paths), 12)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "research-checkpoint.pt"
            checkpoint.write_bytes(b"checked-in-config-dry-run")
            for path in paths:
                with self.subTest(config=path.name):
                    config = json.loads(path.read_text(encoding="utf-8"))
                    run = config["runs"][0]
                    smoke = config["protocols"]["smoke"]
                    smoke_start = int(smoke.get("segment_start_step", 0))
                    arguments = [
                        "launch",
                        "--config",
                        str(path),
                        "--seed",
                        "123456",
                        "--dry-run",
                    ]
                    if smoke_start > 0:
                        metadata = {
                            "schema_version": 1,
                            "filename": checkpoint.name,
                            "step": smoke_start,
                            "target_iterations": int(smoke["iterations"]),
                            "grad_accum_steps": runpod._resolved_grad_accum_steps(
                                smoke
                            ),
                            "bytes": checkpoint.stat().st_size,
                            "sha256": hashlib.sha256(
                                checkpoint.read_bytes()
                            ).hexdigest(),
                            "run_id": run["id"],
                            "runtime_seed": 123456,
                        }
                        (checkpoint.parent / "research-checkpoint.json").write_text(
                            json.dumps(metadata), encoding="utf-8"
                        )
                        arguments.extend(
                            ("--resume-checkpoint", str(checkpoint))
                        )
                    output = io.StringIO()
                    # This test validates the checked-in configuration guards,
                    # not host SSH-key discovery.  Keep it hermetic on clean CI
                    # runners, where ``auto`` would otherwise select the legacy
                    # environment transport and correctly reject continuations.
                    with (
                        mock.patch.object(
                            runpod,
                            "_select_bootstrap_transport",
                            return_value=("ssh", "unit-test transport"),
                        ),
                        contextlib.redirect_stdout(output),
                    ):
                        runpod.main(arguments)
                    preview = json.loads(output.getvalue())
                    self.assertGreaterEqual(preview["setup_headroom_seconds"], 0)
                    self.assertGreater(preview["max_budget_usd"], 0)
                    self.assertEqual(
                        preview["resume_checkpoint"] is not None,
                        smoke_start > 0,
                    )

    def test_create_pod_is_minimal_and_has_platform_termination(self) -> None:
        with mock.patch.object(
            runpod,
            "_gql",
            return_value={"podFindAndDeployOnDemand": {"id": "test-pod"}},
        ) as call:
            pod_id = runpod._create_pod(
                "redacted-key",
                name="unit-pod",
                cloud_type="SECURE",
                terminate_after="2026-07-12T20:00:00Z",
            )
        self.assertEqual(pod_id, "test-pod")
        variables = call.call_args.args[2]
        payload = variables["input"]
        self.assertEqual(payload["imageName"], runpod.IMAGE_NAME)
        self.assertNotIn("templateId", payload)
        self.assertEqual(payload["gpuTypeId"], runpod.GPU_TYPE_ID)
        self.assertNotIn("dockerArgs", payload)
        self.assertNotIn("env", payload)
        self.assertTrue(payload["startSsh"])
        self.assertEqual(payload["terminateAfter"], "2026-07-12T20:00:00Z")

    def test_environment_bootstrap_preserves_image_and_platform_termination(self) -> None:
        environment = [{"key": "TS_PAYLOAD_000", "value": "payload"}]
        with mock.patch.object(
            runpod,
            "_gql",
            return_value={"podFindAndDeployOnDemand": {"id": "test-pod"}},
        ) as call:
            runpod._create_pod(
                "redacted-key",
                name="unit-pod",
                cloud_type="SECURE",
                terminate_after="2026-07-12T20:00:00Z",
                bootstrap_transport="environment",
                docker_command="safe-startup-command",
                environment=environment,
            )
        payload = call.call_args.args[2]["input"]
        self.assertEqual(payload["imageName"], runpod.IMAGE_NAME)
        self.assertEqual(payload["terminateAfter"], "2026-07-12T20:00:00Z")
        self.assertEqual(payload["dockerArgs"], "safe-startup-command")
        self.assertEqual(payload["env"], environment)

    def test_auto_bootstrap_uses_readable_project_key_after_bad_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            default = Path(directory) / "RunPod-Key-Go"
            alternate = Path(directory) / "thetascan_runpod_ed25519"
            default.write_text("inaccessible", encoding="utf-8")
            alternate.write_text("private-key", encoding="utf-8")
            original_open = runpod.Path.open

            def selective_open(path, *args, **kwargs):
                if path == default:
                    raise PermissionError("access denied")
                return original_open(path, *args, **kwargs)

            with (
                mock.patch.object(runpod.shutil, "which", return_value="ssh"),
                mock.patch.object(
                    runpod,
                    "_ssh_key_candidates",
                    return_value=(default, alternate),
                ),
                mock.patch.object(runpod.Path, "open", new=selective_open),
            ):
                transport, detail = runpod._select_bootstrap_transport("auto")
        self.assertEqual(transport, "ssh")
        self.assertIn(str(alternate.resolve()), detail)

    def test_oversized_environment_bootstrap_fails_before_graphql(self) -> None:
        environment = [
            {"key": f"TS_PAYLOAD_{index:03d}", "value": "A" * 3500}
            for index in range(30)
        ]
        with mock.patch.object(runpod, "_gql") as gql:
            with self.assertRaisesRegex(runpod.LauncherError, "too large"):
                runpod._create_pod(
                    "redacted-key",
                    name="unit-pod",
                    cloud_type="SECURE",
                    terminate_after="2026-07-12T20:00:00Z",
                    bootstrap_transport="environment",
                    docker_command="safe-startup-command",
                    environment=environment,
                )
        gql.assert_not_called()

    def test_forced_ssh_fails_before_creation_when_key_is_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "RunPod-Key-Go"
            key.write_text("private-key", encoding="utf-8")
            with (
                mock.patch.object(runpod.shutil, "which", return_value="ssh"),
                mock.patch.object(
                    runpod,
                    "_ssh_key_path",
                    side_effect=runpod.LauncherError("unreadable key"),
                ),
            ):
                with self.assertRaisesRegex(runpod.LauncherError, "preflight failed"):
                    runpod._select_bootstrap_transport("ssh")

    def test_monitor_consumes_the_preview_hard_timeout_field(self) -> None:
        state = {
            "pod_id": "test-pod",
            "proxy_url": "https://invalid.example",
            "hard_timeout_seconds_from_boot": 3900,
            "expected_exports": [],
        }
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(runpod, "_pod_info", return_value=None), \
                mock.patch.object(runpod, "_download_exports", return_value={}):
            state_path = Path(tmp) / "launch.json"
            with self.assertRaises(runpod.LauncherError):
                runpod._monitor_and_collect(
                    "unused-key",
                    state,
                    state_path,
                    poll_seconds=1,
                    boot_timeout_seconds=1,
                )

    def test_fetch_treats_connection_reset_as_transient(self) -> None:
        with mock.patch.object(
            runpod.urllib.request,
            "urlopen",
            side_effect=ConnectionResetError("proxy reset"),
        ):
            self.assertIsNone(
                runpod._fetch("https://example.invalid/status.json", timeout=1)
            )

    def test_graphql_http_error_includes_response_body(self) -> None:
        error = runpod.urllib.error.HTTPError(
            runpod.GRAPHQL_URL,
            500,
            "Internal Server Error",
            {},
            io.BytesIO(b"control-plane request too large"),
        )
        with mock.patch.object(runpod.urllib.request, "urlopen", side_effect=error):
            with self.assertRaisesRegex(
                runpod.LauncherError,
                "HTTP 500: control-plane request too large",
            ):
                runpod._gql("redacted", "query { myself { id } }")

    def test_ssh_bootstrap_retries_a_transient_control_plane_timeout(self) -> None:
        endpoint = {
            "runtime": {
                "ports": [{
                    "privatePort": 22,
                    "publicPort": 12345,
                    "ip": "192.0.2.1",
                    "type": "tcp",
                }]
            }
        }
        completed = types.SimpleNamespace(returncode=0, stderr=b"")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / "key"
            key.write_text("test", encoding="utf-8")
            with (
                mock.patch.object(runpod.shutil, "which", return_value="ssh"),
                mock.patch.object(runpod, "_ssh_key_path", return_value=key),
                mock.patch.object(
                    runpod,
                    "_pod_info",
                    side_effect=[TimeoutError("temporary"), endpoint],
                ) as pod_info,
                mock.patch.object(runpod.subprocess, "run", return_value=completed) as run,
                mock.patch.object(runpod.time, "sleep", return_value=None),
            ):
                runpod._bootstrap_via_ssh(
                    "unused-key",
                    "test-pod",
                    payload=b"payload",
                    startup_script="#!/bin/bash\ntrue\n",
                    run_dir=root,
                    timeout_seconds=30,
                )
        self.assertEqual(pod_info.call_count, 2)
        self.assertEqual(run.call_count, 4)

    def test_ssh_bootstrap_streams_resume_checkpoint_with_bounded_timeout(self) -> None:
        endpoint = {
            "runtime": {
                "ports": [{
                    "privatePort": 22,
                    "publicPort": 12345,
                    "ip": "192.0.2.1",
                    "type": "tcp",
                }]
            }
        }
        completed = types.SimpleNamespace(returncode=0, stderr=b"")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / "key"
            key.write_text("test", encoding="utf-8")
            checkpoint = root / "research-checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint-stream")
            with (
                mock.patch.object(runpod.shutil, "which", return_value="ssh"),
                mock.patch.object(runpod, "_ssh_key_path", return_value=key),
                mock.patch.object(runpod, "_pod_info", return_value=endpoint),
                mock.patch.object(runpod.subprocess, "run", return_value=completed) as run,
            ):
                runpod._bootstrap_via_ssh(
                    "unused-key",
                    "test-pod",
                    payload=b"payload",
                    startup_script="#!/bin/bash\ntrue\n",
                    resume_checkpoint=checkpoint,
                    run_dir=root,
                    timeout_seconds=30,
                )
        self.assertEqual(run.call_count, 5)
        upload = run.call_args_list[3]
        self.assertIn(runpod.REMOTE_RESUME_CHECKPOINT, upload.args[0][-1])
        self.assertEqual(
            upload.kwargs["timeout"], runpod.SSH_CHECKPOINT_UPLOAD_TIMEOUT_SECONDS
        )
        self.assertIn("stdin", upload.kwargs)
        self.assertNotIn("input", upload.kwargs)

    def test_monitor_tolerates_a_transient_control_plane_timeout(self) -> None:
        state = {
            "pod_id": "test-pod",
            "proxy_url": "https://invalid.example",
            "hard_timeout_seconds_from_boot": 3900,
            "expected_exports": [],
            "run_id": "attention",
        }
        running = {
            "desiredStatus": "RUNNING",
            "machine": {"gpuTypeId": runpod.GPU_TYPE_ID},
        }

        def fake_fetch(url: str, timeout: int = 10):
            del timeout
            if url.endswith("status.json"):
                return b'{"state":"running"}'
            if url.endswith("done.flag") and fake_fetch.done_calls:
                return b"passed attention"
            if url.endswith("done.flag"):
                fake_fetch.done_calls += 1
            return None

        fake_fetch.done_calls = 0

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "launch.json"

            def fake_download(_state, run_dir, require_complete=False):
                del require_complete
                (run_dir / "suite-result.json").write_text(
                    '{"status":"passed"}\n', encoding="utf-8"
                )
                return {"suite-result.json": 20}

            with (
                mock.patch.object(
                    runpod,
                    "_pod_info",
                    side_effect=[TimeoutError("temporary"), running],
                ) as pod_info,
                mock.patch.object(runpod, "_fetch", side_effect=fake_fetch),
                mock.patch.object(runpod, "_download_exports", side_effect=fake_download),
                mock.patch.object(runpod.time, "sleep", return_value=None),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertTrue(
                    runpod._monitor_and_collect(
                        "unused-key",
                        state,
                        state_path,
                        poll_seconds=1,
                        boot_timeout_seconds=1,
                    )
                )
            self.assertEqual(pod_info.call_count, 2)

    def test_termination_accepts_a_pod_already_deleted_by_its_watchdog(self) -> None:
        def fake_gql(_api_key: str, query: str, _variables=None):
            if "podTerminate" in query:
                raise runpod.LauncherError("pod not found")
            return {"myself": {"pods": []}}

        with (
            mock.patch.object(runpod, "_gql", side_effect=fake_gql),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertTrue(runpod._terminate_pod("redacted", "already-deleted"))

    def test_done_export_requires_result_and_both_stage_logs(self) -> None:
        state = {
            "proxy_url": "https://example.invalid",
            "run_id": "attention",
            "expected_exports": runpod._expected_exports("attention", checkpoint=False),
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(runpod, "_fetch", return_value=None),
                mock.patch.object(runpod.time, "sleep", return_value=None),
            ):
                with self.assertRaisesRegex(runpod.LauncherError, "required exports"):
                    runpod._download_exports(
                        state, Path(directory), require_complete=True
                    )


if __name__ == "__main__":
    unittest.main()
