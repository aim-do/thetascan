from __future__ import annotations

import unittest

import torch

from thetascan import (
    GNConfig,
    KernelConfig,
    TemporalConfig,
    ThetaScan,
    ThetaScanConfig,
)
from thetascan._core.config import ThetaScanConfig as CoreThetaScanConfig
from thetascan._core.modules.block import ThetaScan as CoreThetaScan


BASELINE_MODEL_PARAMS = 17_059_912
ATTENTION_MIXER_PARAMS = 786_440
D_MODEL = 512


def _benchmark_config(family: str, layout: str) -> ThetaScanConfig:
    config = ThetaScanConfig(
        d_model=D_MODEL,
        n_heads=8,
        memory_multiplier=3,
        share_key_query=layout == "mamba_shared_kq",
        key_value_heads=4 if layout == "transformer_gqa" else None,
        output_gate=False,
        family=family,
        gn=GNConfig(nonlinearity="relu2", jacobian_steps=1),
        kernel=KernelConfig(read_normalization="key_mass", kernel_sharpness=8.0),
        temporal=TemporalConfig(mode="bank", bank_mode="fast"),
    )
    return config


class ProjectionLayoutTests(unittest.TestCase):
    def test_layout_validation_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            ThetaScanConfig(
                n_heads=8,
                share_key_query=True,
                key_value_heads=4,
            ).validate()
        with self.assertRaisesRegex(ValueError, "divisible"):
            ThetaScanConfig(n_heads=8, key_value_heads=3).validate()
        for invalid in (True, 4.0, "4"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(TypeError, "integer or None"):
                    ThetaScanConfig(key_value_heads=invalid).validate()
                with self.assertRaisesRegex(TypeError, "integer or None"):
                    CoreThetaScanConfig(key_value_heads=invalid)

    def test_transformer_gqa_has_pair_shared_kv_and_per_head_q(self) -> None:
        mixer = ThetaScan(_benchmark_config("gn", "transformer_gqa"))
        core = mixer._core
        self.assertEqual(tuple(core.proj_q.weight.shape), (512, 512))
        self.assertEqual(tuple(core.proj_k.weight.shape), (256, 512))
        self.assertEqual(tuple(core.proj_v.weight.shape), (256, 512))
        self.assertFalse(hasattr(core, "kq_bias"))

        output = mixer(torch.randn(1, 4, 512))
        self.assertEqual(tuple(output.shape), (1, 4, 512))

    def test_mamba_shared_layout_keeps_per_head_values(self) -> None:
        mixer = ThetaScan(_benchmark_config("gn", "mamba_shared_kq"))
        core = mixer._core
        self.assertEqual(tuple(core.proj_q1.weight.shape), (64, 512))
        self.assertEqual(tuple(core.proj_k1.weight.shape), (64, 512))
        self.assertEqual(tuple(core.proj_v.weight.shape), (512, 512))
        self.assertEqual(tuple(core.kq_bias.shape), (2, 8, 64))

    def test_independent_layout_has_full_per_head_qkv(self) -> None:
        mixer = ThetaScan(_benchmark_config("gn", "independent"))
        core = mixer._core
        self.assertEqual(tuple(core.proj_qkv.weight.shape), (1536, 512))

    def test_gn_and_kernel_counts_match_for_both_layouts(self) -> None:
        expected = {
            "independent": 1_245_200,
            "mamba_shared_kq": 787_472,
            "transformer_gqa": 983_056,
        }
        for layout, expected_count in expected.items():
            with self.subTest(layout=layout):
                counts = {
                    family: sum(
                        parameter.numel()
                        for parameter in ThetaScan(
                            _benchmark_config(family, layout)
                        ).parameters()
                    )
                    for family in ("gn", "kernel")
                }
                self.assertEqual(
                    counts, {"gn": expected_count, "kernel": expected_count}
                )

    def test_17m_model_count_contract(self) -> None:
        cases = (
            (1_245_200, 576),
            (787_472, 1023),
            (983_056, 832),
        )
        for mixer_params, ffn_hidden in cases:
            with self.subTest(mixer_params=mixer_params, ffn_hidden=ffn_hidden):
                ffn_delta_per_swapped_block = 2 * D_MODEL * (ffn_hidden - 1024)
                total = BASELINE_MODEL_PARAMS + 2 * (
                    mixer_params - ATTENTION_MIXER_PARAMS
                    + ffn_delta_per_swapped_block
                )
                self.assertEqual(total, 17_059_928)


if __name__ == "__main__":
    unittest.main()
