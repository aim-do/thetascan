from __future__ import annotations

import unittest

import torch

from thetascan import (
    GNConfig,
    KernelConfig,
    RoPEConfig,
    RuntimeConfig,
    ThetaScan,
    ThetaScanConfig,
)
from thetascan._core.modules.rope import rope_cache
from thetascan._core.ops.engine import _hrot_stream


class FeatureRoPETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    @staticmethod
    def _config(
        family: str,
        *,
        placement: str = "feature",
        mode: str = "partial",
        backend: str = "quad",
        normalization: str = "none",
        jacobian_steps: int = 1,
    ) -> ThetaScanConfig:
        return ThetaScanConfig(
            d_model=16,
            n_heads=2,
            memory_multiplier=2,
            output_gate=False,
            family=family,
            gn=GNConfig(
                nonlinearity="relu2",
                jacobian_steps=jacobian_steps,
                read_normalization=normalization,
            ),
            kernel=KernelConfig(
                value_representation="raw",
                kernel_sharpness=8.0,
                kernel_sharpness_mode="fixed",
                read_normalization=normalization,
            ),
            rope=RoPEConfig(
                mode=mode, fraction=0.5, base=1_337.0, placement=placement
            ),
            runtime=RuntimeConfig(backend=backend),
        )

    def test_public_placement_maps_to_core_and_validates(self) -> None:
        for placement in ("input", "feature"):
            with self.subTest(placement=placement):
                config = self._config("gn", placement=placement)
                config.validate()
                core = config._to_core_config()
                self.assertEqual(core.rope_placement, placement)
                self.assertEqual(core.rope, "partial")
                self.assertEqual(core.rope_frac, 0.5)
                self.assertEqual(core.rope_base, 1_337.0)

        with self.assertRaisesRegex(ValueError, "RoPE placement"):
            self._config("gn", placement="after_output").validate()

    def test_input_and_feature_are_distinct_paths_but_none_is_a_noop(self) -> None:
        torch.manual_seed(11)
        input_rope = ThetaScan(self._config("kernel", placement="input")).double()
        feature_rope = ThetaScan(self._config("kernel", placement="feature")).double()
        feature_rope.load_state_dict(input_rope.state_dict())
        x = torch.linspace(-0.7, 0.9, 7 * 16, dtype=torch.float64).view(1, 7, 16)

        y_input = input_rope(x)
        y_feature = feature_rope(x)
        self.assertGreater((y_input - y_feature).abs().max().item(), 1e-8)

        torch.manual_seed(12)
        no_rope_input = ThetaScan(
            self._config("kernel", placement="input", mode="none")
        ).double()
        no_rope_feature = ThetaScan(
            self._config("kernel", placement="feature", mode="none")
        ).double()
        no_rope_feature.load_state_dict(no_rope_input.state_dict())
        torch.testing.assert_close(
            no_rope_input(x), no_rope_feature(x), rtol=0.0, atol=0.0
        )

    def test_unnormalized_gn_and_kernel_feature_rope_fp64_forward_backward(self) -> None:
        for family in ("gn", "kernel"):
            with self.subTest(family=family):
                torch.manual_seed(20 if family == "gn" else 21)
                model = ThetaScan(self._config(family)).double()
                x = (
                    0.2
                    * torch.randn(2, 7, 16, dtype=torch.float64)
                ).requires_grad_(True)
                y = model(x)
                self.assertEqual(tuple(y.shape), tuple(x.shape))
                self.assertTrue(torch.isfinite(y).all())

                (y.square().mean() + model.regularization_loss()).backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())
                finite_parameter_grads = [
                    torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                self.assertTrue(finite_parameter_grads)
                self.assertTrue(all(finite_parameter_grads))

    def test_signed_feature_rope_denominators_fail_closed(self) -> None:
        cases = {
            "gn": ("w2_feature_mass", "both_feature_mass"),
            "kernel": ("key_mass", "feature_mass"),
        }
        for family, normalizations in cases.items():
            for mode in ("partial", "full"):
                for normalization in normalizations:
                    with self.subTest(
                        family=family, mode=mode, normalization=normalization
                    ):
                        invalid = self._config(
                            family,
                            placement="feature",
                            mode=mode,
                            normalization=normalization,
                        )
                        with self.assertRaisesRegex(ValueError, "denominator signed"):
                            invalid.validate()

                        # Moving RoPE before the feature map preserves the
                        # non-negative mass; disabling RoPE is also safe even
                        # when the placement field remains "feature".
                        self._config(
                            family,
                            placement="input",
                            mode=mode,
                            normalization=normalization,
                        ).validate()
                        self._config(
                            family,
                            placement="feature",
                            mode="none",
                            normalization=normalization,
                        ).validate()

    def test_long_adversarial_unnormalized_feature_rope_stays_finite(self) -> None:
        values = torch.linspace(-1.0e4, 1.0e4, 257 * 16, dtype=torch.float64)
        x = values.view(1, 257, 16)
        x[:, 1::2].neg_()
        for family in ("gn", "kernel"):
            for mode in ("partial", "full"):
                with self.subTest(family=family, mode=mode):
                    torch.manual_seed(25)
                    model = ThetaScan(
                        self._config(
                            family,
                            placement="feature",
                            mode=mode,
                            normalization="none",
                        )
                    ).double()
                    with torch.no_grad():
                        output = model(x)
                    self.assertEqual(output.shape, x.shape)
                    self.assertTrue(torch.isfinite(output).all())

    def test_feature_rope_naive_and_quad_forward_backward_parity(self) -> None:
        for family, jacobian_steps in (("gn", 1), ("kernel", 1), ("gn", 2)):
            with self.subTest(family=family, jacobian_steps=jacobian_steps):
                torch.manual_seed(30 if family == "gn" else 31)
                naive = ThetaScan(
                    self._config(
                        family, backend="naive", jacobian_steps=jacobian_steps
                    )
                ).double()
                quad = ThetaScan(
                    self._config(
                        family, backend="quad", jacobian_steps=jacobian_steps
                    )
                ).double()
                quad.load_state_dict(naive.state_dict())

                source = 0.15 * torch.randn(2, 8, 16, dtype=torch.float64)
                x_naive = source.clone().requires_grad_(True)
                x_quad = source.clone().requires_grad_(True)
                y_naive = naive(x_naive)
                y_quad = quad(x_quad)
                torch.testing.assert_close(y_quad, y_naive, rtol=2e-8, atol=2e-9)

                probe = torch.linspace(
                    -0.4, 0.6, y_naive.numel(), dtype=torch.float64
                ).view_as(y_naive)
                (y_naive * probe).sum().backward()
                (y_quad * probe).sum().backward()
                torch.testing.assert_close(
                    x_quad.grad, x_naive.grad, rtol=5e-8, atol=5e-9
                )

                naive_parameters = dict(naive.named_parameters())
                quad_parameters = dict(quad.named_parameters())
                self.assertEqual(naive_parameters.keys(), quad_parameters.keys())
                for name, parameter in naive_parameters.items():
                    other = quad_parameters[name]
                    if parameter.grad is None or other.grad is None:
                        self.assertIsNone(parameter.grad)
                        self.assertIsNone(other.grad)
                        continue
                    torch.testing.assert_close(
                        other.grad,
                        parameter.grad,
                        rtol=8e-8,
                        atol=8e-9,
                        msg=lambda message, name=name: f"{name}: {message}",
                    )

    def test_feature_placement_does_not_add_parameters(self) -> None:
        for family in ("gn", "kernel"):
            with self.subTest(family=family):
                torch.manual_seed(40)
                input_rope = ThetaScan(self._config(family, placement="input"))
                torch.manual_seed(40)
                feature_rope = ThetaScan(self._config(family, placement="feature"))
                input_signature = {
                    name: tuple(parameter.shape)
                    for name, parameter in input_rope.named_parameters()
                }
                feature_signature = {
                    name: tuple(parameter.shape)
                    for name, parameter in feature_rope.named_parameters()
                }
                self.assertEqual(feature_signature, input_signature)
                self.assertEqual(
                    sum(parameter.numel() for parameter in feature_rope.parameters()),
                    sum(parameter.numel() for parameter in input_rope.parameters()),
                )

    def test_second_jacobian_step_keys_carry_the_same_rotation(self) -> None:
        """Both LA2 write streams must see one h-space rotation convention.

        The read rotates its query once, so an unrotated second-step key
        stream would be scored at absolute instead of relative positions.
        """
        from thetascan._core.config import ThetaScanConfig as CoreConfig
        from thetascan._core.ops import engine

        dtype = torch.float64
        torch.manual_seed(21)
        heads, time, dim = 2, 6, 4
        cfg = CoreConfig(
            d_model=heads * dim,
            n_heads=heads,
            head_dim=dim,
            mem_mult=2,
            backend="naive",
            rope="partial",
            rope_placement="feature",
            write_iters=2,
            qk_norm=True,
        )
        hidden = cfg.mem_hidden
        weights = [tuple(
            torch.randn(heads, *shape, dtype=dtype) * 0.3
            for shape in ((hidden, dim), (dim, hidden), (hidden, dim))
        )]
        key = torch.randn(1, heads, time, dim, dtype=dtype)
        value = torch.randn(1, heads, time, dim, dtype=dtype)
        rot_dim = max(2, int(hidden * cfg.rope_frac) // 2 * 2)
        cos, sin = rope_cache(time, rot_dim, cfg.rope_base, key.device, dtype)
        hrot = (
            cos.view(1, 1, time, -1).expand(1, heads, time, -1),
            sin.view(1, 1, time, -1).expand(1, heads, time, -1),
            rot_dim,
        )

        plain, _, _ = engine.write_streams(weights, cfg, key, value)
        rotated, _, _ = engine.write_streams(weights, cfg, key, value, hrot=hrot)
        self.assertEqual(len(plain.la2), 2)
        for step, (plain_stream, rotated_stream) in enumerate(
            zip(plain.la2, rotated.la2)
        ):
            with self.subTest(step=step + 1):
                # The payload does not depend on the rotation; the key must
                # carry it — on the second stream as well as the first.
                torch.testing.assert_close(rotated_stream[1], plain_stream[1])
                expected = _hrot_stream(plain_stream[0], hrot)
                torch.testing.assert_close(rotated_stream[0], expected)
                self.assertGreater(
                    (rotated_stream[0] - plain_stream[0]).abs().max().item(),
                    1e-3,
                )

    def test_feature_rotation_has_the_expected_relative_dot_product(self) -> None:
        dtype = torch.float64
        time, width = 5, 6
        cos, sin = rope_cache(time, width, 1_337.0, "cpu", dtype)
        hrot = (
            cos.view(1, 1, time, -1),
            sin.view(1, 1, time, -1),
            width,
        )
        q0 = torch.tensor([0.2, -0.7, 0.5, 0.4, -0.1, 0.9], dtype=dtype)
        k0 = torch.tensor([-0.3, 0.8, 0.6, -0.2, 0.7, 0.1], dtype=dtype)
        q = q0.view(1, 1, 1, width).expand(1, 1, time, width)
        k = k0.view(1, 1, 1, width).expand(1, 1, time, width)
        q_rot = _hrot_stream(q, hrot)
        k_rot = _hrot_stream(k, hrot)

        query_position, key_position = 4, 1
        actual = (
            q_rot[0, 0, query_position] * k_rot[0, 0, key_position]
        ).sum()
        cos_delta = (
            cos[key_position] * cos[query_position]
            + sin[key_position] * sin[query_position]
        )
        sin_delta = (
            sin[key_position] * cos[query_position]
            - cos[key_position] * sin[query_position]
        )
        k1, k2 = k0[0::2], k0[1::2]
        relative_k = torch.stack(
            (
                k1 * cos_delta - k2 * sin_delta,
                k1 * sin_delta + k2 * cos_delta,
            ),
            dim=-1,
        ).flatten()
        expected = (q0 * relative_k).sum()
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
