from __future__ import annotations

import importlib.util
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
BENCHMARK = ROOT / "benchmarks" / "parameter-golf"


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicRecipeBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.configure = _load_module(
            "thetascan_configure_example", EXAMPLES / "configure_mixer.py"
        )
        cls.adapter = _load_module(
            "thetascan_public_parameter_golf_adapter",
            BENCHMARK / "thetascan_benchmark_adapter.py",
        )
        cls.runner = _load_module(
            "thetascan_public_parameter_golf_runner",
            BENCHMARK / "run_benchmark.py",
        )
        cls.remote = _load_module(
            "thetascan_public_parameter_golf_remote",
            BENCHMARK / "remote_suite.py",
        )

    @staticmethod
    def _runner_args(recipe: str) -> types.SimpleNamespace:
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
            optimizer_policy="muon-2d+theta",
            projection_layout="mamba-shared",
            recipe=recipe,
            backend="quad",
            expected_train_shards=16,
            checkpoint_output=None,
            grad_accum_steps=None,
            resume_checkpoint=None,
            resume_checkpoint_sha256="",
            resume_checkpoint_bytes=0,
        )

    def test_documented_configure_recipes_build_and_run(self) -> None:
        reference = self.configure.make_config("gn-reference-v0.1")
        self.assertEqual(reference.family, "gn")
        self.assertEqual(reference.feature_expansion, 1)
        output = self.configure.ThetaScan(reference)(
            torch.randn(1, 4, reference.d_model)
        )
        self.assertEqual(output.shape, (1, 4, reference.d_model))

        kernel = self.configure.make_config("kernel-expanded-reference-v0.1")
        self.assertEqual(kernel.family, "kernel")
        self.assertEqual(kernel.kernel.feature_map, "relu2_ridge")
        self.assertEqual(kernel.feature_expansion, 2)
        output = self.configure.ThetaScan(kernel)(
            torch.randn(1, 4, kernel.d_model)
        )
        self.assertEqual(output.shape, (1, 4, kernel.d_model))

        expanded = self.configure.make_config("gn-expanded-reference-v0.1")
        self.assertEqual(expanded.family, "gn")
        self.assertEqual(expanded.feature_expansion, 2)
        self.assertEqual(expanded.gn.nonlinearity, "relu2_threshold")
        output = self.configure.ThetaScan(expanded)(
            torch.randn(1, 4, expanded.d_model)
        )
        self.assertEqual(output.shape, (1, 4, expanded.d_model))

    def test_remote_suite_preserves_recipe_and_protocol_execution_mapping(self) -> None:
        protocol = {
            "iterations": 20,
            "val_every": 20,
            "warmup_steps": 5,
            "warmdown_iters": 5,
            "grad_accum_steps": 2,
        }
        for recipe in (
            "gn-reference-v0.1",
            "gn-expanded-reference-v0.1",
            "kernel-expanded-reference-v0.1",
        ):
            with self.subTest(recipe=recipe):
                command = self.remote._command(
                    Path("run_benchmark.py"),
                    Path("prepared-checkout"),
                    {
                        "arm": "thetascan",
                        "recipe": recipe,
                        "rope": "partial",
                        "backend": "quad",
                        "projection_layout": "mamba-shared",
                        "optimizer_policy": "muon-2d+theta",
                    },
                    protocol,
                    1,
                    None,
                    None,
                    "",
                    0,
                )
                self.assertEqual(command[command.index("--recipe") + 1], recipe)
                self.assertEqual(
                    command[command.index("--grad-accum-steps") + 1], "2"
                )

        for arm in ("attention", "mamba3"):
            with self.subTest(arm=arm):
                command = self.remote._command(
                    Path("run_benchmark.py"),
                    Path("prepared-checkout"),
                    {
                        "arm": arm,
                        "rope": "partial",
                        "backend": "quad",
                        "projection_layout": "mamba-shared",
                        "optimizer_policy": "muon-2d",
                    },
                    protocol,
                    1,
                    None,
                    None,
                    "",
                    0,
                )
                self.assertNotIn("--recipe", command)

    def test_parameter_golf_expanded_contracts_are_parameter_matched(self) -> None:
        baseline = self.adapter.build_thetascan_mixer(
            512,
            recipe="gn-reference-v0.1",
            rope_mode="partial",
            backend="quad",
            projection_layout="mamba-shared",
        )
        baseline_mixer_params = sum(p.numel() for p in baseline.parameters())
        baseline_full_model_params = 17_059_928

        expected = {
            "gn-expanded-reference-v0.1": {
                "ffn": 1020,
                "full_model": 17_059_928,
                "theta_muon": 393_216,
                "muon_shapes": {
                    "_core.W1.0": (8, 192, 64),
                    "_core.W2.0": (8, 64, 192),
                },
                "control_shapes": {"_core.Wg.0": (8, 384, 1)},
                "expansion_buffers": {
                    "_core._expand_w1_0": (1, 384, 192),
                    "_core._expand_w2_0": (1, 192, 384),
                },
            },
            "kernel-expanded-reference-v0.1": {
                "ffn": 1023,
                "full_model": 17_059_976,
                "theta_muon": 393_216,
                "muon_shapes": {
                    "_core.W1.0": (8, 192, 64),
                    "_core.W2.0": (8, 64, 192),
                },
                "control_shapes": {"_core.kernel_relu2_threshold": (8,)},
                "expansion_buffers": {
                    "_core._expand_w1_0": (1, 384, 192),
                    "_core._expand_w2_0": (1, 192, 384),
                },
            },
        }
        for recipe, contract in expected.items():
            with self.subTest(recipe=recipe):
                mixer = self.adapter.build_thetascan_mixer(
                    512,
                    recipe=recipe,
                    rope_mode="partial",
                    backend="quad",
                    projection_layout="mamba-shared",
                    layer_index=4,
                )
                weights = {
                    name: tuple(parameter.shape)
                    for name, parameter in mixer.named_parameters()
                    if name in contract["muon_shapes"]
                }
                self.assertEqual(weights, contract["muon_shapes"])
                controls = {
                    name: tuple(parameter.shape)
                    for name, parameter in mixer.named_parameters()
                    if name in contract["control_shapes"]
                }
                self.assertEqual(controls, contract["control_shapes"])
                buffers = {
                    name: tuple(buffer.shape)
                    for name, buffer in mixer.named_buffers()
                    if name in contract["expansion_buffers"]
                }
                self.assertEqual(buffers, contract["expansion_buffers"])
                # Fixed expansion maps are derived, not stored.
                self.assertFalse(
                    any("_expand_" in name for name in mixer.state_dict())
                )
                theta_muon_per_block = sum(
                    parameter.numel()
                    for name, parameter in mixer.named_parameters()
                    if name in contract["muon_shapes"]
                )
                self.assertEqual(2 * theta_muon_per_block, contract["theta_muon"])

                args = self._runner_args(recipe)
                environment = self.runner._base_environment(
                    args, 7, "thetascan", recipe
                )
                self.assertEqual(
                    int(environment["THETA_MLP_HIDDEN"]), contract["ffn"]
                )
                mixer_delta = (
                    sum(p.numel() for p in mixer.parameters())
                    - baseline_mixer_params
                )
                ffn_delta = 2 * 512 * (contract["ffn"] - 1023)
                derived_full_model = baseline_full_model_params + 2 * (
                    mixer_delta + ffn_delta
                )
                self.assertEqual(derived_full_model, contract["full_model"])

    def test_expanded_layers_get_distinct_deterministic_maps(self) -> None:
        def build(layer_index: int):
            return self.adapter.build_thetascan_mixer(
                512,
                recipe="gn-expanded-reference-v0.1",
                rope_mode="partial",
                backend="quad",
                projection_layout="mamba-shared",
                layer_index=layer_index,
            )

        first = dict(build(4).named_buffers())
        second = dict(build(5).named_buffers())
        rebuilt = dict(build(4).named_buffers())
        name = "_core._expand_w1_0"
        self.assertTrue(torch.equal(first[name], rebuilt[name]))
        self.assertFalse(torch.equal(first[name], second[name]))

    def test_runner_and_adapter_expose_exactly_three_canonical_recipes(self) -> None:
        expected = {
            "gn-reference-v0.1",
            "gn-expanded-reference-v0.1",
            "kernel-expanded-reference-v0.1",
        }
        self.assertEqual(set(self.runner.RECIPES), expected)
        self.assertEqual(set(self.adapter.RECIPES), expected)
        kernel_recipes = {
            recipe for recipe in self.runner.RECIPES if recipe.startswith("kernel-")
        }
        self.assertEqual(kernel_recipes, {"kernel-expanded-reference-v0.1"})
        with self.assertRaisesRegex(ValueError, "unknown THETA_RECIPE"):
            self.adapter.build_thetascan_mixer(
                512,
                recipe="not-a-public-recipe",
                rope_mode="partial",
                backend="quad",
                projection_layout="mamba-shared",
            )
        for recipe in kernel_recipes:
            with self.subTest(recipe=recipe):
                mixer = self.adapter.build_thetascan_mixer(
                    512,
                    recipe=recipe,
                    rope_mode="partial",
                    backend="quad",
                    projection_layout="mamba-shared",
                )
                self.assertEqual(mixer.config.family, "kernel")

    def test_every_advertised_benchmark_recipe_builds(self) -> None:
        for recipe in self.adapter.RECIPES:
            with self.subTest(recipe=recipe):
                mixer = self.adapter.build_thetascan_mixer(
                    512,
                    recipe=recipe,
                    rope_mode="partial",
                    backend="quad",
                    projection_layout="mamba-shared",
                )
                mixer.config.validate()
                self.assertEqual(mixer.config.d_model, 512)
                self.assertIn(mixer.config.family, ("gn", "kernel"))

    def test_generated_harness_routes_canonical_controls(self) -> None:
        self.assertEqual(set(self.runner.RECIPES), set(self.adapter.RECIPES))
        source = (BENCHMARK / "prepare_harness.py").read_text(encoding="utf-8")
        for recipe, minimum in (
            ("gn-reference-v0.1", 1),
            ("gn-expanded-reference-v0.1", 4),
            ("kernel-expanded-reference-v0.1", 4),
        ):
            with self.subTest(recipe=recipe):
                # Construction, optimizer routing, and validation logging each
                # carry a fail-closed explicit recipe classification.
                self.assertGreaterEqual(source.count(f'"{recipe}"'), minimum)
        self.assertIn("kernel_relu2_threshold_per_head:", source)
        self.assertIn("gn_threshold_stats:", source)
        self.assertIn("bank_timescales_per_head:", source)
        for removed in ("W1b", "W2b", "Wgb", "correction_scale_per_head", "rho2"):
            self.assertNotIn(removed, source)


if __name__ == "__main__":
    unittest.main()
