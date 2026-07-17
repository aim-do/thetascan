from __future__ import annotations

import unittest

import torch

from thetascan._core.ops.interface import Accumulator


class EMANormalizationTests(unittest.TestCase):
    def test_key_mass_uses_the_same_ema_weights_as_memory_numerator(self) -> None:
        """Verify the normalized read's two streams against the direct recurrence."""
        torch.manual_seed(7)
        batch, heads, time, key_dim, value_dim = 2, 2, 5, 3, 4
        keys = torch.randn(batch, heads, time, key_dim)
        values = torch.randn(batch, heads, time, value_dim)
        queries = torch.randn(batch, heads, time, key_dim)
        log_alpha = -torch.rand(batch, heads, time, 1)

        accumulator = Accumulator("naive", decay_m=log_alpha)
        numerator = accumulator(queries, keys, values, "m")
        mass = accumulator.mass_cum(keys)

        expected_numerator = torch.empty_like(numerator)
        expected_mass = torch.empty_like(mass)
        state = torch.zeros(batch, heads, key_dim, value_dim)
        key_mass = torch.zeros(batch, heads, key_dim)
        for t in range(time):
            alpha = log_alpha[:, :, t].exp()
            state = state * alpha.unsqueeze(-1) + torch.einsum(
                "bhk,bhv->bhkv", keys[:, :, t], values[:, :, t]
            )
            key_mass = key_mass * alpha + keys[:, :, t]
            expected_numerator[:, :, t] = torch.einsum(
                "bhk,bhkv->bhv", queries[:, :, t], state
            )
            expected_mass[:, :, t] = key_mass

        torch.testing.assert_close(numerator, expected_numerator)
        torch.testing.assert_close(mass, expected_mass)
        torch.testing.assert_close(
            (queries * mass).sum(-1, keepdim=True),
            (queries * expected_mass).sum(-1, keepdim=True),
        )


if __name__ == "__main__":
    unittest.main()
