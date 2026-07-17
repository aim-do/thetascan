import math
import unittest

import torch

from thetascan import (
    GNConfig,
    RuntimeConfig,
    TemporalConfig,
    ThetaScan,
    ThetaScanConfig,
)
from thetascan._core.config import ThetaScanConfig as CoreConfig


class TemporalBankTests(unittest.TestCase):
    def _config(
        self,
        family: str,
        *,
        temporal: TemporalConfig,
    ) -> ThetaScanConfig:
        return ThetaScanConfig(
            d_model=16,
            n_heads=2,
            memory_multiplier=2,
            output_gate=False,
            family=family,
            gn=GNConfig(read_normalization="both_feature_mass"),
            temporal=temporal,
            runtime=RuntimeConfig(backend="naive"),
        )

    @staticmethod
    def _same_slow_parameters(source: ThetaScan, target: ThetaScan) -> None:
        source_state = source.state_dict()
        target_state = target.state_dict()
        with torch.no_grad():
            for name, value in target_state.items():
                if "fade_alpha" in name or "fade_eta" in name:
                    continue
                value.copy_(source_state[name])

    def test_default_preserves_legacy_shapes_and_exact_output(self) -> None:
        default_temporal = TemporalConfig(mode="bank")
        explicit_temporal = TemporalConfig(
            mode="bank",
            recency_branches=1,
            retention_inits=(0.9,),
            blend_mode="free",
        )
        torch.manual_seed(7)
        default = ThetaScan(self._config("kernel", temporal=default_temporal)).double()
        torch.manual_seed(7)
        explicit = ThetaScan(self._config("kernel", temporal=explicit_temporal)).double()

        self.assertEqual(tuple(default._core.fade_alpha.shape), (2,))
        self.assertEqual(tuple(default._core.fade_eta.shape), (2,))
        self.assertEqual(default.state_dict().keys(), explicit.state_dict().keys())
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)

        x = torch.randn(2, 7, 16, dtype=torch.float64)
        torch.testing.assert_close(default(x), explicit(x), rtol=0, atol=0)

    def test_two_branch_fp64_matches_independent_read_oracle(self) -> None:
        # Since both branches read the same slow state independently, the exact
        # oracle is the sum of two one-branch bank outputs minus one slow output.
        # This remains a strong check under normalized kernel/GN because each fast
        # read must filter its numerator and feature mass with the same alpha.
        for family in ("kernel", "gn"):
            for bank_mode in ("fast", "stale"):
                with self.subTest(family=family, bank_mode=bank_mode):
                    retentions = (0.72, 0.96)
                    dual_t = TemporalConfig(
                        mode="bank",
                        bank_mode=bank_mode,
                        recency_branches=2,
                        retention_inits=retentions,
                    )
                    one_ts = [
                        TemporalConfig(
                            mode="bank",
                            bank_mode=bank_mode,
                            retention_inits=(alpha,),
                        )
                        for alpha in retentions
                    ]
                    slow_t = TemporalConfig(mode="sum")

                    torch.manual_seed(11)
                    dual = ThetaScan(self._config(family, temporal=dual_t)).double()
                    singles = []
                    for temporal in one_ts:
                        torch.manual_seed(19)
                        model = ThetaScan(self._config(family, temporal=temporal)).double()
                        self._same_slow_parameters(dual, model)
                        singles.append(model)
                    torch.manual_seed(23)
                    slow = ThetaScan(self._config(family, temporal=slow_t)).double()
                    self._same_slow_parameters(dual, slow)

                    eta = torch.tensor([[0.23, -0.17], [0.41, 0.08]], dtype=torch.float64)
                    with torch.no_grad():
                        dual._core.fade_eta.copy_(eta)
                        for branch, model in enumerate(singles):
                            model._core.fade_eta.copy_(eta[branch])

                    x = torch.randn(2, 6, 16, dtype=torch.float64)
                    actual = dual(x)
                    expected = singles[0](x) + singles[1](x) - slow(x)
                    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)

    def test_half_life_conversion_tanh_constraint_and_gradients(self) -> None:
        temporal = TemporalConfig(
            mode="bank",
            recency_branches=2,
            half_life_inits=(3.0, 17.0),
            blend_mode="tanh",
        )
        expected = tuple(math.pow(2.0, -1.0 / half_life) for half_life in (3.0, 17.0))
        self.assertEqual(temporal.resolved_retentions(), expected)

        for family in ("kernel", "gn"):
            with self.subTest(family=family):
                torch.manual_seed(31)
                model = ThetaScan(self._config(family, temporal=temporal)).double()
                self.assertEqual(tuple(model._core.fade_alpha.shape), (2, 2))
                self.assertEqual(tuple(model._core.fade_eta.shape), (2, 2))
                torch.testing.assert_close(
                    torch.sigmoid(model._core.fade_alpha)[0],
                    torch.full((2,), expected[0], dtype=torch.float64),
                    rtol=2e-8,
                    atol=2e-8,
                )
                with torch.no_grad():
                    model._core.fade_eta.copy_(torch.tensor(
                        [[0.3, -0.2], [0.15, 0.4]], dtype=torch.float64
                    ))
                blends = model._core.fade_blends()
                self.assertTrue(torch.all(blends.abs() < 1.0))

                x = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
                model(x).square().mean().backward()
                for parameter in (model._core.fade_alpha, model._core.fade_eta):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_multi_branch_trains_for_both_families(self) -> None:
        temporal = TemporalConfig(
            mode="bank",
            recency_branches=2,
            retention_inits=(0.8, 0.97),
            blend_mode="free",
        )
        for family in ("kernel", "gn"):
            with self.subTest(family=family):
                model = ThetaScan(self._config(family, temporal=temporal)).double()
                with torch.no_grad():
                    model._core.fade_eta.copy_(torch.tensor(
                        [[0.2, -0.1], [0.05, 0.3]], dtype=torch.float64
                    ))
                x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
                y = model(x)
                self.assertTrue(torch.isfinite(y).all())
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(model._core.fade_eta.grad).all())
                self.assertTrue(torch.isfinite(model._core.fade_alpha.grad).all())

    def test_validation_rejects_ambiguous_or_unsupported_settings(self) -> None:
        invalid = (
            TemporalConfig(mode="bank", recency_branches=3),
            TemporalConfig(
                mode="bank", recency_branches=2, retention_inits=(0.9,)
            ),
            TemporalConfig(
                mode="bank", recency_branches=2, retention_inits=(0.9, 0.9)
            ),
            TemporalConfig(
                mode="bank", recency_branches=2, half_life_inits=(4.0, -1.0)
            ),
            TemporalConfig(
                mode="bank",
                retention_inits=(0.9,),
                half_life_inits=(4.0,),
            ),
        )
        for temporal in invalid:
            with self.subTest(temporal=temporal), self.assertRaises(ValueError):
                temporal.validate()

        with self.assertRaisesRegex(ValueError, "fade_alpha_inits"):
            CoreConfig(
                read_fade=True,
                accumulation="sum",
                fade_branches=2,
                fade_alpha_inits=(0.8,),
            )


if __name__ == "__main__":
    unittest.main()
