"""Tests for fixed and learned kernel sharpness controls."""
from __future__ import annotations

import unittest

import torch

from thetascan import KernelConfig, TemporalConfig, ThetaScan, ThetaScanConfig


class KernelSharpnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    @staticmethod
    def _config(
        mode: str = "fixed",
        sharpness: float = 8.0,
        *,
        bank: bool = False,
    ) -> ThetaScanConfig:
        return ThetaScanConfig(
            d_model=32,
            n_heads=2,
            memory_multiplier=2,
            output_gate=False,
            family="kernel",
            kernel=KernelConfig(
                kernel_sharpness=sharpness,
                kernel_sharpness_mode=mode,  # type: ignore[arg-type]
            ),
            temporal=TemporalConfig(mode="bank" if bank else "sum"),
        )

    def test_fixed_mode_is_default_and_adds_no_parameter(self) -> None:
        config = self._config()
        core = config._to_core_config()
        self.assertEqual(core.softmax_gain, 8.0)
        self.assertEqual(core.softmax_gain_mode, "fixed")
        mixer = ThetaScan(config)
        self.assertIsNone(mixer._core.kernel_sharpness_raw)
        self.assertIsNone(mixer._core.kernel_sharpness())
        self.assertNotIn(
            "kernel_sharpness_raw", dict(mixer._core.named_parameters())
        )

    def test_learned_mode_has_positive_per_head_exact_initialization(self) -> None:
        mixer = ThetaScan(self._config("learned_per_head", 2.5))
        raw = mixer._core.kernel_sharpness_raw
        self.assertIsNotNone(raw)
        self.assertEqual(tuple(raw.shape), (2,))
        actual = mixer._core.kernel_sharpness()
        torch.testing.assert_close(actual, torch.full((2,), 2.5))
        self.assertTrue(torch.all(actual > 0))

        # Even an extreme optimizer update cannot make the effective gain
        # negative or non-finite in float32.
        with torch.no_grad():
            raw.fill_(-1_000.0)
        actual = mixer._core.kernel_sharpness()
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.all(actual > 0))

    def test_learned_initialization_matches_fixed_forward(self) -> None:
        x = torch.randn(2, 11, 32, generator=torch.Generator().manual_seed(91))
        torch.manual_seed(17)
        fixed = ThetaScan(self._config("fixed", 4.0))
        torch.manual_seed(17)
        learned = ThetaScan(self._config("learned_per_head", 4.0))
        torch.testing.assert_close(
            learned(x), fixed(x), rtol=2e-6, atol=2e-7
        )

    def test_learned_per_head_backpropagates_with_bank(self) -> None:
        torch.manual_seed(29)
        mixer = ThetaScan(self._config("learned_per_head", bank=True))
        x = torch.randn(2, 13, 32, requires_grad=True)
        loss = mixer(x).square().mean()
        loss.backward()

        grad = mixer._core.kernel_sharpness_raw.grad
        self.assertIsNotNone(grad)
        self.assertEqual(tuple(grad.shape), (2,))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_learned_per_feature_matches_fixed_initialization(self) -> None:
        x = torch.randn(2, 11, 32, generator=torch.Generator().manual_seed(92))
        torch.manual_seed(18)
        fixed = ThetaScan(self._config("fixed", 4.0))
        torch.manual_seed(18)
        learned = ThetaScan(self._config("learned_per_feature", 4.0))

        raw = learned._core.kernel_sharpness_raw
        self.assertIsNotNone(raw)
        self.assertEqual(tuple(raw.shape), (2, 32))
        actual = learned._core.kernel_sharpness()
        torch.testing.assert_close(actual, torch.full((2, 32), 4.0))
        torch.testing.assert_close(learned(x), fixed(x), rtol=2e-6, atol=2e-7)

    def test_learned_per_feature_is_positive_and_backpropagates_with_bank(self) -> None:
        torch.manual_seed(30)
        mixer = ThetaScan(self._config("learned_per_feature", bank=True))
        x = torch.randn(2, 13, 32, requires_grad=True)
        mixer(x).square().mean().backward()

        raw = mixer._core.kernel_sharpness_raw
        self.assertIsNotNone(raw)
        self.assertIsNotNone(raw.grad)
        self.assertEqual(tuple(raw.grad.shape), (2, 32))
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(raw.grad.abs().sum().item(), 0.0)
        with torch.no_grad():
            raw.fill_(-1_000.0)
        actual = mixer._core.kernel_sharpness()
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.all(actual > 0))

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "kernel_sharpness_mode"):
            self._config("learned_global").validate()


if __name__ == "__main__":
    unittest.main()
