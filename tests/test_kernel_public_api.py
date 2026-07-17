from __future__ import annotations

import json
import unittest

import torch
import thetascan

from thetascan import (
    KernelConfig,
    RegularizationConfig,
    RuntimeConfig,
    TemporalConfig,
    ThetaScan,
    ThetaScanConfig,
)


class KernelPublicApiTests(unittest.TestCase):
    def _base(self, *, family: str, **kwargs) -> ThetaScanConfig:
        return ThetaScanConfig(
            d_model=32,
            n_heads=2,
            memory_multiplier=2,
            family=family,
            runtime=RuntimeConfig(backend="naive"),
            **kwargs,
        )

    def test_removed_legacy_kernel_spellings_fail_closed(self) -> None:
        self.assertFalse(hasattr(thetascan, "NWConfig"))
        self.assertFalse(hasattr(thetascan, "KernelKind"))
        with self.assertRaisesRegex(ValueError, "'gn' or 'kernel'"):
            self._base(family="nw").validate()  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "unknown kernel feature map"):
            KernelConfig(feature_map="rbf").validate()  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            KernelConfig(kind="relu2_ridge")  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            KernelConfig(key_query_anchors_trainable=False)  # type: ignore[call-arg]
        with self.assertRaisesRegex(ValueError, "learned_per_feature"):
            KernelConfig(
                kernel_sharpness_mode="learned_per_anchor"  # type: ignore[arg-type]
            ).validate()
        with self.assertRaisesRegex(ValueError, "unknown ThetaScan preset"):
            ThetaScanConfig.from_dict({"preset": "nw_reference_v0_1"})

    def test_softmax_partition_feature_map_maps_to_normalized_address(self) -> None:
        config = self._base(
            family="kernel", kernel=KernelConfig(feature_map="softmax_partition")
        )
        core = config._to_core_config()
        self.assertEqual(config.kernel.feature_map, "softmax_partition")
        self.assertEqual(core.kernel_kind, "softmax_partition")
        self.assertEqual(core.nonlin, "softmax_hidden")

    def test_unknown_kernel_feature_map_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown kernel feature map"):
            KernelConfig(feature_map="unknown").validate()  # type: ignore[arg-type]

    def test_discarded_manifest_spellings_fail_closed(self) -> None:
        """Manifests must use canonical field names; no legacy mapping exists."""
        with self.assertRaises(TypeError):
            ThetaScanConfig.from_dict({
                "family": "kernel",
                "kernel": {"kind": "relu2_ridge"},
            })
        with self.assertRaises(TypeError):
            # The research-era correction section was removed outright; manifests
            # carrying it (any spelling) must fail closed instead of resolving.
            ThetaScanConfig.from_dict({
                "family": "kernel",
                "correction": {"mode": "loo", "scale_init": 0.0},
            })
        with self.assertRaises(TypeError):
            ThetaScanConfig.from_dict({
                "family": "kernel",
                "temporal": {"mode": "bank", "fade_branches": 2},
            })
        with self.assertRaisesRegex(ValueError, "unknown temporal mode"):
            ThetaScanConfig.from_dict({
                "family": "kernel",
                "temporal": {"mode": "fade"},
            })
        mixer = ThetaScan(self._base(family="kernel"))
        self.assertFalse(hasattr(mixer, "ortho_loss"))
        self.assertTrue(hasattr(mixer, "regularization_loss"))

    def test_config_round_trip_and_preset_resolution(self) -> None:
        original = ThetaScanConfig.kernel_expanded_reference_v0_1(
            d_model=64, n_heads=4
        )
        restored = ThetaScanConfig.from_dict(
            json.loads(json.dumps(original.to_dict()))
        )
        self.assertEqual(original, restored)
        self.assertIsInstance(restored.temporal.half_life_inits, tuple)

        resolved = ThetaScanConfig.from_dict({
            "preset": "kernel_expanded_reference_v0_1",
            "d_model": 64,
            "n_heads": 4,
            "runtime": {"backend": "naive"},
            "temporal": {"half_life_inits": [4.0, 32.0]},
        })
        self.assertEqual(resolved.family, "kernel")
        self.assertEqual(resolved.kernel.feature_map, "relu2_ridge")
        self.assertEqual(resolved.feature_expansion, 2)
        self.assertEqual(resolved.temporal.half_life_inits, (4.0, 32.0))
        self.assertEqual(resolved.runtime.backend, "naive")

    def test_to_dict_is_detached(self) -> None:
        config = ThetaScanConfig.kernel_expanded_reference_v0_1(
            d_model=64, n_heads=4
        )
        manifest = config.to_dict()
        manifest["temporal"]["half_life_inits"] = [1.0, 2.0]
        self.assertEqual(config.temporal.half_life_inits, (8.0, 64.0))

    def test_module_snapshots_config_and_core_owns_regularization(self) -> None:
        config = self._base(
            family="kernel",
            kernel=KernelConfig(feature_map="softmax_partition", value_representation="value_mlp"),
            regularization=RegularizationConfig(value_mlp_weight=0.25),
        )
        mixer = ThetaScan(config)
        config.regularization.value_mlp_weight = 0.0
        config.kernel.value_representation = "raw"
        self.assertEqual(mixer.config.regularization.value_mlp_weight, 0.25)
        self.assertEqual(mixer.config.kernel.value_representation, "value_mlp")
        self.assertEqual(mixer._core_config.value_mlp_ortho, 0.25)
        self.assertEqual(mixer.regularization_loss(), mixer._core.ortho_loss())
        self.assertGreater(float(mixer.regularization_loss().detach()), 0.0)

    def test_feature_freeze_uses_explicit_core_ownership(self) -> None:
        config = self._base(
            family="kernel",
            kernel=KernelConfig(
                feature_map="projected_bspline",
                bspline_scale_mode="learned_per_head",
                feature_parameters_trainable=False,
            ),
        )
        mixer = ThetaScan(config)
        feature_parameters = list(mixer._core.key_query_feature_parameters())
        self.assertTrue(feature_parameters)
        self.assertTrue(all(not p.requires_grad for p in feature_parameters))
        self.assertTrue(all(p.requires_grad for p in mixer._core.W2))

    def test_reference_preset_and_all_feature_maps_run(self) -> None:
        x = torch.randn(1, 5, 64)
        config = ThetaScanConfig.kernel_expanded_reference_v0_1(
            d_model=64, n_heads=4
        )
        config.runtime = RuntimeConfig(backend="naive")
        y = ThetaScan(config)(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

        for feature_map in ("softmax_partition", "relu2_ridge", "projected_bspline"):
            with self.subTest(feature_map=feature_map):
                config = ThetaScanConfig(
                    d_model=64,
                    n_heads=4,
                    memory_multiplier=2,
                    family="kernel",
                    kernel=KernelConfig(feature_map=feature_map),
                    runtime=RuntimeConfig(backend="naive"),
                )
                y = ThetaScan(config)(x)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
