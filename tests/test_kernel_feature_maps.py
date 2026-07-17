"""Tests for the three public kernel-memory feature geometries."""
from __future__ import annotations

import unittest

import torch

from thetascan import KernelConfig, TemporalConfig, ThetaScan, ThetaScanConfig
from thetascan._core.ops import engine, kernel_features
from thetascan._core.ops.interface import Accumulator


class KernelFeatureMapTests(unittest.TestCase):
    def _config(self, kind: str, normalization: str = "key_mass", **kwargs):
        return ThetaScanConfig(
            d_model=16,
            n_heads=2,
            memory_multiplier=2,
            output_gate=False,
            family="kernel",
            kernel=KernelConfig(
                feature_map=kind,
                read_normalization=normalization,
                **kwargs,
            ),
        )

    def test_softmax_partition_formula_is_normalized(self) -> None:
        cfg = self._config("softmax_partition")._to_core_config()
        pre = torch.randn(2, 2, 5, 16, dtype=torch.float64)
        actual = kernel_features.feature_map(pre, cfg)
        expected = torch.softmax(pre * 8.0, dim=-1)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_relative_st_threshold_is_sparse_normalized_and_trainable(self) -> None:
        pre = torch.tensor(
            [[[[2.0, 1.0, 0.0, -1.0]], [[1.0, 0.9, -2.0, -3.0]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        rho = torch.tensor([0.2, 0.5], dtype=torch.float64, requires_grad=True)
        out = kernel_features.softmax_partition_features(
            pre,
            gain=1.0,
            fallback_gain=1.0,
            score_bias=None,
            sparsity="relative_st",
            relative_threshold=rho,
            threshold_temperature=0.25,
            eps=1e-9,
        )
        torch.testing.assert_close(
            out.sum(-1), torch.ones_like(out.sum(-1)), rtol=1e-12, atol=1e-12
        )
        self.assertGreater((out == 0).sum().item(), 0)
        self.assertTrue(torch.all(out.amax(-1) > 0))
        weights = torch.arange(1, 5, dtype=out.dtype).view(1, 1, 1, 4)
        (out * weights).sum().backward()
        self.assertIsNotNone(rho.grad)
        self.assertTrue(torch.isfinite(rho.grad).all())
        self.assertGreater(rho.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(pre.grad).all())

    def test_relative_st_blend_is_exactly_dense_at_zero_and_can_open(self) -> None:
        mixer = ThetaScan(
            self._config(
                "softmax_partition",
                kernel_sharpness=1.0,
                sparsity="relative_st_blend",
                sparse_blend_init=0.0,
                relative_threshold_init=0.6,
            )
        )._core.double()
        pre = torch.tensor(
            [[[[1.0, 0.5, 0.0, -0.5]], [[0.8, 0.6, 0.0, -1.0]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        controls = mixer.kernel_controls()
        out = kernel_features.feature_map(
            pre,
            mixer.cfg,
            softmax_gain=mixer.kernel_sharpness(),
            score_bias=controls["score_bias"],
            relative_threshold=controls["relative_threshold"],
            sparse_blend_alpha=controls["sparse_blend_alpha"],
        )
        expected = torch.softmax(pre, dim=-1)
        torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            controls["sparse_blend_alpha"],
            torch.zeros(2, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        weights = torch.tensor(
            [1.0, 2.0, 4.0, 8.0], dtype=out.dtype
        ).view(1, 1, 1, 4)
        (out * weights).sum().backward()
        blend_grad = mixer.kernel_sparse_blend_raw.grad
        self.assertIsNotNone(blend_grad)
        self.assertTrue(torch.isfinite(blend_grad).all())
        self.assertGreater(blend_grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(pre.grad).all())
        self.assertGreater(pre.grad.abs().sum().item(), 0.0)

    def test_relative_st_blend_is_a_positive_simplex_with_all_gradients(self) -> None:
        mixer = ThetaScan(
            self._config(
                "softmax_partition",
                kernel_sharpness=1.0,
                sparsity="relative_st_blend",
                sparse_blend_init=0.5,
                relative_threshold_init=0.6,
            )
        )._core.double()
        directions = torch.tensor(
            [
                [[1.0], [0.5], [0.0], [-0.5]],
                [[0.8], [0.6], [0.0], [-1.0]],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        x = torch.ones(1, 2, 1, 1, dtype=torch.float64)
        pre = torch.einsum("bhtd,hmd->bhtm", x, directions)
        controls = mixer.kernel_controls()
        out = kernel_features.feature_map(
            pre,
            mixer.cfg,
            softmax_gain=mixer.kernel_sharpness(),
            score_bias=controls["score_bias"],
            relative_threshold=controls["relative_threshold"],
            sparse_blend_alpha=controls["sparse_blend_alpha"],
        )
        dense = torch.softmax(pre, dim=-1)
        sparse = kernel_features.relative_threshold_features(
            pre,
            controls["relative_threshold"],
            temperature=mixer.cfg.kernel_threshold_temperature,
            straight_through=True,
            eps=mixer.cfg.eps,
        )
        expected = dense + 0.5 * (sparse - dense)
        torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            out.sum(-1), torch.ones_like(out.sum(-1)), rtol=1e-12, atol=1e-12
        )
        self.assertTrue(torch.all(out > 0))
        weights = torch.tensor(
            [1.0, 2.0, 4.0, 8.0], dtype=out.dtype
        ).view(1, 1, 1, 4)
        (out * weights).sum().backward()
        for parameter in (
            directions,
            mixer.kernel_relative_threshold_raw,
            mixer.kernel_sparse_blend_raw,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_relu2_ridge_is_a_positive_sparse_simplex(self) -> None:
        pre = torch.tensor([[[[-2.0, -1.0, 1.0, 2.0]]]], dtype=torch.float64)
        out = kernel_features.relu2_ridge_features(
            pre, score_bias=None, eps=1e-9
        )
        self.assertTrue(torch.all(out >= 0))
        torch.testing.assert_close(out.sum(-1), torch.ones_like(out.sum(-1)))
        self.assertEqual((out == 0).sum().item(), 2)

    def test_relu2_ridge_head_threshold_matches_formula_and_has_gradient(self) -> None:
        pre = torch.tensor(
            [[[[-1.0, 0.5, 2.0]], [[0.0, 2.0, 3.0]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        threshold = torch.tensor(
            [0.25, 0.5], dtype=torch.float64, requires_grad=True
        )
        out = kernel_features.relu2_ridge_features(
            pre,
            score_bias=None,
            relu2_threshold=threshold,
            eps=1e-9,
        )
        shifted = pre - threshold.view(1, 2, 1, 1)
        expected = torch.relu(shifted).square()
        expected = expected / expected.sum(-1, keepdim=True)
        torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)
        weights = torch.tensor([1.0, 2.0, 4.0], dtype=out.dtype)
        (out * weights).sum().backward()
        self.assertIsNotNone(threshold.grad)
        self.assertTrue(torch.isfinite(threshold.grad).all())
        self.assertGreater(threshold.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(pre.grad).all())

    def test_relu2_ridge_threshold_parameter_is_zero_initialized_per_head(self) -> None:
        mixer = ThetaScan(
            self._config(
                "relu2_ridge",
                relu2_threshold_mode="learned_per_head",
            )
        )._core
        self.assertEqual(tuple(mixer.kernel_relu2_threshold.shape), (2,))
        self.assertEqual(
            torch.count_nonzero(mixer.kernel_relu2_threshold).item(), 0
        )
        self.assertIs(
            mixer.kernel_controls()["relu2_threshold"],
            mixer.kernel_relu2_threshold,
        )

    def test_cubic_bspline_partition_support_and_gradient(self) -> None:
        coordinates = torch.linspace(-3.0, 3.0, 31, dtype=torch.float64)
        coordinates.requires_grad_(True)
        basis = kernel_features.open_uniform_bspline_basis(
            coordinates, n_basis=8, degree=3, bound=3.0
        )
        torch.testing.assert_close(
            basis.sum(-1), torch.ones_like(coordinates), rtol=1e-12, atol=1e-12
        )
        self.assertTrue(torch.all(basis >= 0))
        self.assertLessEqual(int((basis > 1e-12).sum(-1).max()), 4)
        weights = torch.arange(8, dtype=basis.dtype)
        (basis * weights).sum().backward()
        self.assertTrue(torch.isfinite(coordinates.grad).all())
        self.assertGreater(coordinates.grad[1:-1].abs().sum().item(), 0.0)

    def test_projected_bspline_uses_expected_direction_and_feature_shapes(self) -> None:
        mixer = ThetaScan(
            self._config(
                "projected_bspline", bspline_scale_mode="learned_per_head"
            )
        )
        core = mixer._core
        # head_dim=8, memory_multiplier=2 => 16 features = 2 directions * 8 basis.
        self.assertEqual(tuple(core.W1[0].shape), (2, 2, 8))
        self.assertEqual(tuple(core.W2[0].shape), (2, 8, 16))
        self.assertEqual(tuple(core.bspline_scale_raw.shape), (2,))

    def test_all_maps_compose_with_dual_recency_and_backward(self) -> None:
        variants = (
            KernelConfig(
                feature_map="softmax_partition",
                kernel_sharpness_mode="learned_per_head",
                sparsity="relative_st",
            ),
            KernelConfig(
                feature_map="softmax_partition",
                kernel_sharpness_mode="learned_per_head",
                sparsity="relative_st_blend",
                sparse_blend_init=0.5,
            ),
            KernelConfig(
                feature_map="relu2_ridge",
                relu2_threshold_mode="learned_per_head",
            ),
            KernelConfig(
                feature_map="projected_bspline",
                bspline_scale_mode="learned_per_head",
            ),
        )
        for kernel in variants:
            with self.subTest(
                feature_map=kernel.feature_map, sparsity=kernel.sparsity
            ):
                config = ThetaScanConfig(
                    d_model=16,
                    n_heads=2,
                    memory_multiplier=2,
                    output_gate=False,
                    family="kernel",
                    kernel=kernel,
                    temporal=TemporalConfig(
                        mode="bank",
                        bank_mode="fast",
                        recency_branches=2,
                        half_life_inits=(8.0, 64.0),
                    ),
                )
                mixer = ThetaScan(config).double()
                x = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
                y = mixer(x)
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue(torch.isfinite(x.grad).all())
                for parameter in mixer.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                if mixer._core.kernel_sparse_blend_raw is not None:
                    for parameter in (
                        mixer._core.W1[0],
                        mixer._core.kernel_relative_threshold_raw,
                        mixer._core.kernel_sparse_blend_raw,
                    ):
                        self.assertIsNotNone(parameter.grad)
                        self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_quad_matches_independent_oracle_for_all_maps_and_normalizers(self) -> None:
        variants = (
            ("softmax_partition", {"sparsity": "relative_st", "score_bias": True}),
            (
                "softmax_partition",
                {
                    "sparsity": "relative_st_blend",
                    "sparse_blend_init": 0.4,
                    "score_bias": True,
                },
            ),
            (
                "relu2_ridge",
                {"relu2_threshold_mode": "learned_per_head"},
            ),
            ("projected_bspline", {"bspline_scale_mode": "learned_per_head"}),
        )
        torch.manual_seed(17)
        for kind, kwargs in variants:
            for normalization in ("key_mass", "feature_mass"):
                with self.subTest(kind=kind, normalization=normalization):
                    mixer = ThetaScan(
                        self._config(kind, normalization, **kwargs)
                    )._core.double()
                    with torch.no_grad():
                        mixer.W2[0].normal_(std=0.1)
                        if mixer.kernel_score_bias is not None:
                            mixer.kernel_score_bias.normal_(std=0.1)
                        if mixer.kernel_relu2_threshold is not None:
                            mixer.kernel_relu2_threshold.normal_(std=0.1)
                    B, H, T, d = 1, 2, 6, 8
                    key = torch.randn(B, H, T, d, dtype=torch.float64)
                    query = torch.randn(B, H, T, d, dtype=torch.float64)
                    value = torch.randn(B, H, T, d, dtype=torch.float64)
                    key = key / key.norm(dim=-1, keepdim=True)
                    query = query / query.norm(dim=-1, keepdim=True)
                    cfg = mixer.cfg
                    weights = mixer._weights()
                    sharpness = mixer.kernel_sharpness()
                    controls = mixer.kernel_controls()
                    streams, _, _ = engine.write_streams(
                        weights,
                        cfg,
                        key,
                        value,
                        softmax_gain=sharpness,
                        kernel_controls=controls,
                    )
                    actual = engine.dual_read(
                        weights,
                        cfg,
                        query,
                        streams,
                        Accumulator("quad", eps=cfg.eps),
                        softmax_gain=sharpness,
                        kernel_controls=controls,
                    )
                    expected = engine.naive_read(
                        weights,
                        cfg,
                        key,
                        value,
                        query,
                        softmax_gain=sharpness,
                        kernel_controls=controls,
                    )
                    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)

    def test_invalid_kernel_feature_map_combinations_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature_map='softmax_partition'"):
            self._config(
                "projected_bspline", sparsity="relative_st"
            ).validate()
        with self.assertRaisesRegex(ValueError, "divisible"):
            ThetaScanConfig(
                d_model=12,
                n_heads=2,
                memory_multiplier=1,
                family="kernel",
                kernel=KernelConfig(
                    feature_map="projected_bspline", bspline_basis_count=8
                ),
            ).validate()
        with self.assertRaisesRegex(ValueError, "relu2_ridge"):
            self._config(
                "softmax_partition",
                relu2_threshold_mode="learned_per_head",
            ).validate()
        with self.assertRaisesRegex(ValueError, "redundant"):
            self._config(
                "relu2_ridge",
                relu2_threshold_mode="learned_per_head",
                score_bias=True,
            ).validate()
        with self.assertRaisesRegex(ValueError, "feature_map='softmax_partition'"):
            self._config(
                "relu2_ridge",
                sparsity="relative_st",
            ).validate()
        with self.assertRaisesRegex(ValueError, "feature_map='softmax_partition'"):
            self._config(
                "relu2_ridge",
                sparsity="relative_st_blend",
            ).validate()
        for blend_init in (-0.1, 1.1):
            with self.subTest(sparse_blend_init=blend_init):
                with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
                    self._config(
                        "softmax_partition",
                        sparsity="relative_st_blend",
                        sparse_blend_init=blend_init,
                    ).validate()
        for sparsity in ("none", "relative_st"):
            with self.subTest(inactive_sparsity=sparsity):
                with self.assertRaisesRegex(ValueError, "active only"):
                    self._config(
                        "softmax_partition",
                        sparsity=sparsity,
                        sparse_blend_init=0.25,
                    ).validate()
        with self.assertRaisesRegex(RuntimeError, "blend tensor"):
            kernel_features.softmax_partition_features(
                torch.zeros(1, 2, 1, 4),
                gain=1.0,
                fallback_gain=1.0,
                score_bias=None,
                sparsity="relative_st_blend",
                relative_threshold=torch.full((2,), 0.5),
                threshold_temperature=0.25,
                eps=1e-8,
            )


if __name__ == "__main__":
    unittest.main()
