from __future__ import annotations

import unittest

import torch
import thetascan

from thetascan import (
    GNConfig,
    KernelConfig,
    RegularizationConfig,
    RoPEConfig,
    RuntimeConfig,
    TemporalConfig,
    ThetaScan,
    ThetaScanConfig,
)


class PublicApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def _config(self, **kwargs) -> ThetaScanConfig:
        return ThetaScanConfig(d_model=32, n_heads=2, memory_multiplier=2, **kwargs)

    def test_gn_two_jacobian_steps_forward_and_backward(self) -> None:
        config = self._config(
            family="gn",
            gn=GNConfig(nonlinearity="relu2_threshold", jacobian_steps=2),
        )
        mixer = ThetaScan(config)
        self.assertEqual(mixer._core_config.write_iters, 2)
        x = torch.randn(2, 7, 32, requires_grad=True)
        y = mixer(x)
        self.assertEqual(tuple(y.shape), (2, 7, 32))
        (y.square().mean() + mixer.regularization_loss()).backward()
        self.assertIsNotNone(x.grad)

    def test_kernel_value_representations(self) -> None:
        for representation in ("raw", "value_anchors", "value_mlp"):
            with self.subTest(representation=representation):
                kwargs = {"value_representation": representation}
                if representation == "value_anchors":
                    kwargs["value_anchors"] = 5
                if representation == "value_mlp":
                    kwargs["value_mlp_multiplier"] = 1.5
                config = self._config(
                    family="kernel", kernel=KernelConfig(**kwargs)
                )
                mixer = ThetaScan(config)
                y = mixer(torch.randn(2, 5, 32))
                self.assertTrue(torch.isfinite(y).all())
                self.assertEqual(mixer._core_config.write_rule, "kernel")
                self.assertTrue(mixer._core_config.read_norm)

    def test_kernel_anchor_freeze_and_value_mlp_regularization(self) -> None:
        config = self._config(
            family="kernel",
            kernel=KernelConfig(
                value_representation="value_mlp",
                feature_parameters_trainable=False,
            ),
            regularization=RegularizationConfig(value_mlp_weight=0.1),
        )
        mixer = ThetaScan(config)
        anchors = [parameter for name, parameter in mixer._core.named_parameters() if name.startswith("W1.")]
        self.assertTrue(anchors)
        self.assertTrue(all(not parameter.requires_grad for parameter in anchors))
        self.assertTrue(torch.isfinite(mixer.regularization_loss()))

    def test_rope_modes_map_and_run_for_both_families(self) -> None:
        """RoPE is a shared positional option, rather than a family-specific axis."""
        for family in ("gn", "kernel"):
            for mode in ("none", "partial", "full"):
                with self.subTest(family=family, mode=mode):
                    rope = RoPEConfig(mode=mode, fraction=0.75, base=1_337.0)
                    mixer = ThetaScan(self._config(family=family, rope=rope))
                    self.assertEqual(mixer._core_config.rope, mode)
                    self.assertEqual(mixer._core_config.rope_frac, 0.75)
                    self.assertEqual(mixer._core_config.rope_base, 1_337.0)
                    y = mixer(torch.randn(1, 7, 32))
                    self.assertEqual(tuple(y.shape), (1, 7, 32))
                    self.assertTrue(torch.isfinite(y).all())

    def test_full_rope_keeps_an_unpaired_odd_head_channel(self) -> None:
        config = ThetaScanConfig(
            d_model=30,
            n_heads=2,
            family="gn",
            rope=RoPEConfig(mode="full"),
        )
        y = ThetaScan(config)(torch.randn(1, 5, 30))
        self.assertEqual(tuple(y.shape), (1, 5, 30))
        self.assertTrue(torch.isfinite(y).all())

    def test_normalized_kernel_ema_forgets_state_and_key_mass_together(self) -> None:
        """The EMA numerator and the key-mass denominator must share decay weights."""
        config = self._config(
            family="kernel",
            kernel=KernelConfig(
                read_normalization="key_mass",
                value_representation="value_anchors",
                value_anchors=5,
            ),
            temporal=TemporalConfig(mode="ema"),
        )
        mixer = ThetaScan(config)
        self.assertEqual(mixer._core_config.accumulation, "ema_gate")
        self.assertTrue(mixer._core_config.read_norm)
        self.assertEqual(mixer._core_config.value_centers, 5)
        x = torch.randn(2, 7, 32, requires_grad=True)
        y = mixer(x)
        loss = y.square().mean() + mixer.regularization_loss()
        loss.backward()
        self.assertTrue(torch.isfinite(y).all())
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_gn_ema_runs_with_and_without_feature_mass_normalization(self) -> None:
        """GN EMA filters the LA2 numerator and its optional mass together."""
        for normalization in ("none", "w2_feature_mass"):
            with self.subTest(normalization=normalization):
                config = self._config(
                    family="gn",
                    gn=GNConfig(read_normalization=normalization),
                    temporal=TemporalConfig(mode="ema"),
                )
                mixer = ThetaScan(config)
                self.assertEqual(mixer._core_config.accumulation, "ema_gate")
                self.assertEqual(
                    mixer._core_config.read_norm,
                    normalization == "w2_feature_mass",
                )
                self.assertTrue(mixer._core_config.fast_w1)
                x = torch.randn(2, 7, 32, requires_grad=True)
                y = mixer(x)
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(y).all())
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_unnormalized_kernel_ema_remains_supported(self) -> None:
        """Kernel EMA is not conditional on a key-mass-normalized read."""
        config = self._config(
            family="kernel",
            kernel=KernelConfig(
                read_normalization="none", value_representation="raw"
            ),
            temporal=TemporalConfig(mode="ema"),
        )
        mixer = ThetaScan(config)
        self.assertEqual(mixer._core_config.accumulation, "ema_gate")
        self.assertFalse(mixer._core_config.read_norm)
        x = torch.randn(2, 7, 32, requires_grad=True)
        mixer(x).square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_bank_is_supported_by_both_families(self) -> None:
        for family in ("gn", "kernel"):
            with self.subTest(family=family):
                mixer = ThetaScan(self._config(family=family, temporal=TemporalConfig(mode="bank")))
                y = mixer(torch.randn(1, 9, 32))
                self.assertTrue(torch.isfinite(y).all())

        normalized_gn = ThetaScan(
            self._config(
                family="gn",
                gn=GNConfig(read_normalization="w2_feature_mass"),
                temporal=TemporalConfig(mode="bank"),
            )
        )
        self.assertTrue(normalized_gn._core_config.read_norm)
        self.assertTrue(torch.isfinite(normalized_gn(torch.randn(1, 9, 32))).all())

    def test_invalid_cells_are_explicit(self) -> None:
        with self.assertRaises(ValueError):
            self._config(
                family="kernel",
                kernel=KernelConfig(
                    value_representation="value_anchors",
                    read_normalization="none",
                ),
            ).validate()
        with self.assertRaisesRegex(
            ValueError, "normalization requires jacobian_steps=1"
        ):
            self._config(
                family="gn",
                gn=GNConfig(
                    jacobian_steps=2,
                    read_normalization="w2_feature_mass",
                ),
            ).validate()
        with self.assertRaisesRegex(ValueError, "non-negative relu2"):
            self._config(
                family="gn",
                gn=GNConfig(
                    nonlinearity="silu",
                    read_normalization="w2_feature_mass",
                ),
            ).validate()

    def test_value_anchor_head_regularization_is_finite(self) -> None:
        config = self._config(
            family="kernel",
            kernel=KernelConfig(
                value_representation="value_anchors", value_anchors=4
            ),
            regularization=RegularizationConfig(head_weight=0.1),
        )
        self.assertTrue(torch.isfinite(ThetaScan(config).regularization_loss()))

    def test_all_public_float_fields_reject_non_finite_values(self) -> None:
        kernel_fields = (
            "value_mlp_multiplier",
            "kernel_sharpness",
            "sparse_blend_init",
            "relative_threshold_init",
            "threshold_temperature",
            "bspline_bound",
            "bspline_scale",
        )
        for field in kernel_fields:
            for invalid in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(section="kernel", field=field, invalid=invalid):
                    config = KernelConfig()
                    setattr(config, field, invalid)
                    with self.assertRaisesRegex(ValueError, "must be finite"):
                        config.validate()

        for field in ("fraction", "base"):
            with self.subTest(section="rope", field=field):
                config = RoPEConfig()
                setattr(config, field, float("nan"))
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    config.validate()

        temporal_cases = (
            TemporalConfig(retention_init=float("nan")),
            TemporalConfig(retention_inits=(float("inf"),)),
            TemporalConfig(half_life_inits=(float("nan"),)),
        )
        for config in temporal_cases:
            with self.subTest(section="temporal", config=config):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    config.validate()

        for field in ("feature_weight", "head_weight", "value_mlp_weight"):
            with self.subTest(section="regularization", field=field):
                config = RegularizationConfig()
                setattr(config, field, float("inf"))
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    config.validate()

    def test_structural_integer_fields_reject_floats_and_bools(self) -> None:
        for invalid in (1.0, True):
            with self.subTest(section="gn", invalid=invalid):
                with self.assertRaisesRegex(TypeError, "must be an integer"):
                    GNConfig(jacobian_steps=invalid).validate()  # type: ignore[arg-type]

            for field in ("value_anchors", "bspline_basis_count", "bspline_degree"):
                with self.subTest(section="kernel", field=field, invalid=invalid):
                    config = KernelConfig()
                    setattr(config, field, invalid)
                    with self.assertRaisesRegex(TypeError, "must be an integer"):
                        config.validate()

            with self.subTest(section="temporal", invalid=invalid):
                with self.assertRaisesRegex(TypeError, "must be an integer"):
                    TemporalConfig(recency_branches=invalid).validate()  # type: ignore[arg-type]

            for field in (
                "d_model",
                "n_heads",
                "head_dim",
                "memory_multiplier",
                "depth",
                "key_value_heads",
            ):
                with self.subTest(section="top", field=field, invalid=invalid):
                    config = ThetaScanConfig(d_model=32, n_heads=2)
                    setattr(config, field, invalid)
                    with self.assertRaisesRegex(TypeError, "must be an integer"):
                        config.validate()

    def test_public_boolean_fields_reject_truthy_non_booleans(self) -> None:
        for field in ("score_bias", "feature_parameters_trainable"):
            for invalid in (0, 1, "false", "true"):
                with self.subTest(section="kernel", field=field, invalid=invalid):
                    kernel = KernelConfig()
                    setattr(kernel, field, invalid)
                    with self.assertRaisesRegex(TypeError, "must be a boolean"):
                        kernel.validate()

        for field in ("share_key_query", "output_gate"):
            for invalid in (0, 1, "false", "true"):
                with self.subTest(section="top", field=field, invalid=invalid):
                    config = ThetaScanConfig(d_model=32, n_heads=2)
                    setattr(config, field, invalid)
                    with self.assertRaisesRegex(TypeError, "must be a boolean"):
                        config.validate()

    def test_from_dict_rejects_malformed_boolean_manifests(self) -> None:
        malformed = (
            {"share_key_query": 1},
            {"output_gate": "false"},
            {"family": "kernel", "kernel": {"score_bias": 0}},
            {
                "family": "kernel",
                "kernel": {"feature_parameters_trainable": "true"},
            },
            {
                "preset": "kernel_expanded_reference_v0_1",
                "kernel": {"score_bias": "false"},
            },
        )
        for manifest in malformed:
            with self.subTest(manifest=manifest):
                with self.assertRaisesRegex(TypeError, "must be a boolean"):
                    ThetaScanConfig.from_dict(manifest)

    def test_orthogonality_is_finite_with_single_row_matrices(self) -> None:
        config = ThetaScanConfig(
            d_model=2,
            n_heads=2,
            memory_multiplier=2,
            family="gn",
            gn=GNConfig(nonlinearity="relu2_threshold"),
            regularization=RegularizationConfig(
                feature_weight=0.2,
                head_weight=0.3,
            ),
        )
        model = ThetaScan(config)
        loss = model.regularization_loss()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        finite_grads = [
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(finite_grads)
        self.assertTrue(all(finite_grads))

    def test_gn_threshold_storage_is_not_orthogonality_regularized(self) -> None:
        config = self._config(
            family="gn",
            gn=GNConfig(nonlinearity="relu2_threshold"),
            regularization=RegularizationConfig(
                feature_weight=0.2,
                head_weight=0.3,
            ),
        )
        model = ThetaScan(config)
        before = model.regularization_loss().detach().clone()
        with torch.no_grad():
            for threshold in model._core.Wg:
                threshold.normal_(mean=100.0, std=20.0)
        after = model.regularization_loss().detach()
        torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)

    def test_demo_model_is_not_public_library_api(self) -> None:
        self.assertFalse(hasattr(thetascan, "TinyLM"))
        self.assertNotIn("TinyLM", thetascan.__all__)
        self.assertFalse(hasattr(ThetaScanConfig, "tiny_cpu"))

    def test_kernel_sharpness_maps_to_core_gain(self) -> None:
        config = self._config(
            family="kernel", kernel=KernelConfig(kernel_sharpness=8.0)
        )
        self.assertEqual(config._to_core_config().softmax_gain, 8.0)
        with self.assertRaises(ValueError):
            self._config(
                family="kernel", kernel=KernelConfig(kernel_sharpness=0.0)
            ).validate()
        with self.assertRaisesRegex(TypeError, "bandwidth"):
            KernelConfig(bandwidth=8.0)  # type: ignore[call-arg]

    def test_share_key_query_and_output_gate_map(self) -> None:
        config = self._config(family="gn", share_key_query=True, output_gate=False)
        core = config._to_core_config()
        self.assertTrue(core.share_kq)
        self.assertFalse(core.out_gate)
        y = ThetaScan(config)(torch.randn(2, 6, 32))
        self.assertTrue(torch.isfinite(y).all())

    def test_transformer_gqa_projection_layout_maps_and_runs(self) -> None:
        config = ThetaScanConfig(
            d_model=32,
            n_heads=4,
            key_value_heads=2,
            family="gn",
        )
        core = config._to_core_config()
        self.assertFalse(core.share_kq)
        self.assertEqual(core.key_value_heads, 2)
        mixer = ThetaScan(config)
        self.assertEqual(tuple(mixer._core.proj_q.weight.shape), (32, 32))
        self.assertEqual(tuple(mixer._core.proj_k.weight.shape), (16, 32))
        self.assertEqual(tuple(mixer._core.proj_v.weight.shape), (16, 32))
        self.assertFalse(hasattr(mixer._core, "kq_bias"))
        y = mixer(torch.randn(2, 6, 32))
        self.assertEqual(tuple(y.shape), (2, 6, 32))
        self.assertTrue(torch.isfinite(y).all())

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            ThetaScanConfig(
                d_model=32,
                n_heads=4,
                share_key_query=True,
                key_value_heads=2,
            ).validate()
        with self.assertRaisesRegex(ValueError, "divisible"):
            ThetaScanConfig(d_model=32, n_heads=4, key_value_heads=3).validate()

    def test_transformer_gqa_group_expansion_is_pairwise_and_deterministic(self) -> None:
        from thetascan._core.modules.block import _expand_kv_groups

        groups = torch.tensor([10.0, 20.0]).view(1, 2, 1, 1)
        expanded = _expand_kv_groups(groups, repeats=2)
        self.assertEqual(expanded[:, :, 0, 0].tolist(), [[10.0, 10.0, 20.0, 20.0]])

    def test_zero_output_projection_is_public_and_in_place(self) -> None:
        mixer = ThetaScan(self._config(family="gn"))
        returned = mixer.zero_output_projection_()
        self.assertIs(returned, mixer)
        self.assertEqual(torch.count_nonzero(mixer._core.proj_out.weight).item(), 0)
        self.assertEqual(
            torch.count_nonzero(mixer(torch.randn(1, 5, 32))).item(), 0
        )

    def test_portable_backends_agree(self) -> None:
        """naive, quad, and cumsum are the same scan; their outputs must match."""
        x = torch.randn(2, 12, 32, generator=torch.Generator().manual_seed(11))
        for family in ("gn", "kernel"):
            outputs = {}
            for backend in ("naive", "quad", "cumsum"):
                torch.manual_seed(7)
                config = self._config(family=family, runtime=RuntimeConfig(backend=backend))
                outputs[backend] = ThetaScan(config)(x)
            for backend in ("quad", "cumsum"):
                with self.subTest(family=family, backend=backend):
                    torch.testing.assert_close(
                        outputs[backend], outputs["naive"], rtol=2e-4, atol=2e-5
                    )

    def test_cumsum_backend_is_restricted_to_sum_temporal(self) -> None:
        for mode in ("ema", "bank"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "cumsum"):
                    self._config(
                        family="gn",
                        temporal=TemporalConfig(mode=mode),
                        runtime=RuntimeConfig(backend="cumsum"),
                    ).validate()
        # The undecayed prefix-sum path stays supported and exact.
        self._config(
            family="gn", runtime=RuntimeConfig(backend="cumsum")
        ).validate()

    def test_presets_construct_validate_and_run(self) -> None:
        presets = {
            "gn_reference_v0_1": ThetaScanConfig.gn_reference_v0_1(
                d_model=64, n_heads=4
            ),
            "gn_expanded_reference_v0_1": (
                ThetaScanConfig.gn_expanded_reference_v0_1(
                    d_model=64, n_heads=4
                )
            ),
            "kernel_expanded_reference_v0_1": (
                ThetaScanConfig.kernel_expanded_reference_v0_1(
                    d_model=64, n_heads=4
                )
            ),
        }
        for name, config in presets.items():
            with self.subTest(preset=name):
                config.validate()
                mixer = ThetaScan(config)
                y = mixer(torch.randn(2, 8, config.d_model))
                self.assertEqual(tuple(y.shape), (2, 8, config.d_model))
                self.assertTrue(torch.isfinite(y).all())

    def test_expanded_references_have_the_reported_core_configuration(self) -> None:
        core = ThetaScanConfig.kernel_expanded_reference_v0_1()._to_core_config()
        self.assertEqual(core.write_rule, "kernel")
        self.assertEqual(core.kernel_kind, "relu2_ridge")
        self.assertEqual(core.feature_expansion, 2)
        self.assertEqual(core.expansion_key, "thetascan")
        self.assertTrue(core.share_kq)
        self.assertFalse(core.out_gate)
        self.assertTrue(core.read_norm)
        self.assertEqual(core.mem_mult, 6)
        self.assertTrue(core.read_fade)
        self.assertEqual(core.read_fade_mode, "fast")
        self.assertEqual(core.accumulation, "sum")

        gn_core = ThetaScanConfig.gn_expanded_reference_v0_1()._to_core_config()
        self.assertEqual(gn_core.write_rule, "gn")
        self.assertEqual(gn_core.feature_expansion, 2)
        self.assertEqual(gn_core.mem_mult, 6)
        self.assertTrue(gn_core.learn_thresh)
        self.assertTrue(gn_core.read_norm and gn_core.read_norm_w1)
        self.assertTrue(gn_core.fast_w1)

    def test_versioned_reference_presets_have_exact_public_fields(self) -> None:
        gn = ThetaScanConfig.gn_reference_v0_1(d_model=64, n_heads=4)
        self.assertEqual(
            (gn.family, gn.memory_multiplier, gn.share_key_query, gn.output_gate),
            ("gn", 3, True, False),
        )
        self.assertEqual(
            (
                gn.gn.nonlinearity,
                gn.gn.jacobian_steps,
                gn.gn.read_normalization,
            ),
            ("relu2", 1, "both_feature_mass"),
        )
        self.assertEqual(gn.feature_expansion, 1)
        self.assertEqual(
            (gn.rope.mode, gn.rope.fraction, gn.rope.base),
            ("partial", 0.5, 10_000.0),
        )
        self.assertEqual(
            (
                gn.temporal.mode,
                gn.temporal.bank_mode,
                gn.temporal.recency_branches,
                gn.temporal.retention_inits,
                gn.temporal.blend_mode,
            ),
            ("bank", "fast", 1, (0.9,), "free"),
        )

        expanded = ThetaScanConfig.gn_expanded_reference_v0_1(
            d_model=64, n_heads=4
        )
        self.assertEqual(expanded.memory_multiplier, 6)
        self.assertEqual(expanded.feature_expansion, 2)
        self.assertEqual(expanded.gn.nonlinearity, "relu2_threshold")
        self.assertEqual(expanded.gn.read_normalization, "both_feature_mass")
        self.assertEqual(expanded.temporal.recency_branches, 1)
        self.assertEqual(expanded.temporal.retention_inits, (0.9,))
        self.assertEqual(expanded.temporal.blend_mode, "free")
        self.assertEqual(expanded.rope.mode, "partial")

        kernel = ThetaScanConfig.kernel_expanded_reference_v0_1(
            d_model=64, n_heads=4
        )
        self.assertEqual(
            (
                kernel.family,
                kernel.memory_multiplier,
                kernel.feature_expansion,
                kernel.share_key_query,
                kernel.output_gate,
            ),
            ("kernel", 6, 2, True, False),
        )
        self.assertEqual(kernel.kernel.feature_map, "relu2_ridge")
        self.assertEqual(kernel.kernel.value_representation, "raw")
        self.assertEqual(kernel.kernel.read_normalization, "key_mass")
        self.assertEqual(
            kernel.kernel.relu2_threshold_mode, "learned_per_head"
        )
        self.assertEqual(kernel.temporal.recency_branches, 2)
        self.assertEqual(kernel.temporal.half_life_inits, (8.0, 64.0))
        self.assertEqual(kernel.temporal.blend_mode, "free")
        self.assertEqual(kernel.rope.mode, "partial")

        for config in (gn, expanded, kernel):
            config.validate()

    def test_removed_gn_feature_mass_alias_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "GN read normalization"):
            GNConfig(
                read_normalization="feature_mass"  # type: ignore[arg-type]
            ).validate()

    def test_single_retention_init_configures_one_branch(self) -> None:
        config = ThetaScanConfig(
            d_model=32,
            n_heads=2,
            family="gn",
            gn=GNConfig(read_normalization="w2_feature_mass"),
            temporal=TemporalConfig(mode="bank", retention_init=0.83),
        )
        core = config._to_core_config()
        self.assertTrue(core.read_norm)
        self.assertFalse(core.read_norm_w1)
        self.assertEqual(core.fade_branches, 1)
        self.assertEqual(core.resolved_fade_alphas, (0.83,))
        mixer = ThetaScan(config)
        self.assertEqual(tuple(mixer._core.fade_alpha.shape), (2,))
        self.assertEqual(tuple(mixer._core.fade_eta.shape), (2,))

if __name__ == "__main__":
    unittest.main()
