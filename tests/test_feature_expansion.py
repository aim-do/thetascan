from __future__ import annotations

import unittest

import torch

from thetascan import (
    GNConfig,
    KernelConfig,
    RuntimeConfig,
    ThetaScan,
    ThetaScanConfig,
)


def _config(
    family: str = "gn",
    *,
    memory_multiplier: int = 6,
    feature_expansion: int = 2,
    expansion_key: str = "thetascan",
    backend: str = "quad",
    **kwargs,
) -> ThetaScanConfig:
    sections = {}
    if family == "gn":
        sections["gn"] = kwargs.pop(
            "gn",
            GNConfig(
                nonlinearity="relu2_threshold",
                read_normalization="both_feature_mass",
            ),
        )
    else:
        sections["kernel"] = kwargs.pop(
            "kernel",
            KernelConfig(
                feature_map="relu2_ridge",
                read_normalization="key_mass",
            ),
        )
    return ThetaScanConfig(
        d_model=128,
        n_heads=4,
        memory_multiplier=memory_multiplier,
        feature_expansion=feature_expansion,
        expansion_key=expansion_key,
        family=family,
        runtime=RuntimeConfig(backend=backend),
        **sections,
        **kwargs,
    )


class FeatureExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_trainable_cores_shrink_to_the_base_width(self) -> None:
        # d_model=128, n_heads=4 -> head_dim=32; mem_mult=6 -> effective 192;
        # expansion 2 -> trainable base width 96.
        mixer = ThetaScan(_config("gn"))
        params = dict(mixer.named_parameters())
        self.assertEqual(tuple(params["_core.W1.0"].shape), (4, 96, 32))
        self.assertEqual(tuple(params["_core.W2.0"].shape), (4, 32, 96))
        # Width-sized controls stay at the effective width.
        self.assertEqual(tuple(params["_core.Wg.0"].shape), (4, 192, 1))
        buffers = dict(mixer.named_buffers())
        self.assertEqual(tuple(buffers["_core._expand_w1_0"].shape), (1, 192, 96))
        self.assertEqual(tuple(buffers["_core._expand_w2_0"].shape), (1, 96, 192))

        dense = ThetaScan(_config("gn", feature_expansion=1))
        expanded_total = sum(p.numel() for p in mixer.parameters())
        dense_total = sum(p.numel() for p in dense.parameters())
        self.assertLess(expanded_total, dense_total)

    def test_expansion_maps_are_deterministic_in_the_key(self) -> None:
        first = dict(ThetaScan(_config("gn")).named_buffers())
        second = dict(ThetaScan(_config("gn")).named_buffers())
        other = dict(
            ThetaScan(_config("gn", expansion_key="other")).named_buffers()
        )
        for name in ("_core._expand_w1_0", "_core._expand_w2_0"):
            self.assertTrue(torch.equal(first[name], second[name]))
            self.assertFalse(torch.equal(first[name], other[name]))
        # Rademacher rows normalized to unit L2: entries are +-1/sqrt(base).
        w1_map = first["_core._expand_w1_0"]
        self.assertTrue(
            torch.allclose(
                w1_map.abs(), torch.full_like(w1_map, 1.0 / 96 ** 0.5)
            )
        )

    def test_state_dict_round_trip_excludes_maps_and_carries_fingerprint(self) -> None:
        torch.manual_seed(11)
        source = ThetaScan(_config("kernel", backend="naive"))
        state = source.state_dict()
        self.assertFalse(any("_expand_" in key for key in state))
        fingerprint = state["_core._expansion_fingerprint"]
        self.assertEqual(fingerprint.dtype, torch.uint8)
        self.assertEqual(tuple(fingerprint.shape), (40,))
        target = ThetaScan(_config("kernel", backend="naive"))
        target.load_state_dict(state, strict=True)
        x = torch.randn(2, 9, 128)
        torch.testing.assert_close(source(x), target(x))

    def test_load_rejects_a_present_different_expansion_key(self) -> None:
        source = ThetaScan(_config("gn", expansion_key="source-key"))
        for strict in (True, False):
            with self.subTest(strict=strict):
                target = ThetaScan(_config("gn", expansion_key="target-key"))
                before = {
                    name: tensor.detach().clone()
                    for name, tensor in target.state_dict().items()
                }
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"fingerprint mismatch \(different expansion_key",
                ):
                    target.load_state_dict(source.state_dict(), strict=strict)
                after = target.state_dict()
                self.assertEqual(set(after), set(before))
                for name, expected in before.items():
                    torch.testing.assert_close(after[name], expected)

    def test_load_rejects_an_unknown_fingerprint_schema(self) -> None:
        source = ThetaScan(_config("gn"))
        state = source.state_dict()
        fingerprint = state["_core._expansion_fingerprint"].clone()
        fingerprint[4:8] = torch.tensor((0, 0, 0, 2), dtype=torch.uint8)
        state["_core._expansion_fingerprint"] = fingerprint
        target = ThetaScan(_config("gn"))
        with self.assertRaisesRegex(
            RuntimeError, "unsupported fingerprint schema version 2"
        ):
            target.load_state_dict(state, strict=True)

    def test_legacy_expanded_checkpoint_requires_explicit_non_strict_load(
        self,
    ) -> None:
        source = ThetaScan(_config("kernel", backend="naive"))
        legacy_state = source.state_dict()
        del legacy_state["_core._expansion_fingerprint"]

        strict_target = ThetaScan(_config("kernel", backend="naive"))
        with self.assertRaisesRegex(RuntimeError, "_expansion_fingerprint"):
            strict_target.load_state_dict(legacy_state, strict=True)

        compatible_target = ThetaScan(_config("kernel", backend="naive"))
        incompatible = compatible_target.load_state_dict(
            legacy_state, strict=False
        )
        self.assertEqual(
            incompatible.missing_keys, ["_core._expansion_fingerprint"]
        )
        x = torch.randn(2, 9, 128)
        torch.testing.assert_close(source(x), compatible_target(x))

    def test_forward_backward_finite_and_backend_parity(self) -> None:
        x = torch.randn(2, 12, 128, dtype=torch.float64)
        for family in ("gn", "kernel"):
            outputs = {}
            for backend in ("naive", "quad"):
                torch.manual_seed(7)
                mixer = ThetaScan(_config(family, backend=backend)).double()
                y = mixer(x)
                outputs[backend] = y
                loss = y.square().mean()
                mixer.zero_grad()
                loss.backward()
                grads_finite = all(
                    torch.isfinite(p.grad).all()
                    for p in mixer.parameters()
                    if p.grad is not None
                )
                with self.subTest(family=family, backend=backend):
                    self.assertTrue(torch.isfinite(y).all())
                    self.assertTrue(grads_finite)
            with self.subTest(family=family, backend="parity"):
                torch.testing.assert_close(
                    outputs["naive"], outputs["quad"], atol=1e-9, rtol=1e-9
                )

    def test_expansion_one_is_bitwise_identical_to_dense(self) -> None:
        x = torch.randn(2, 10, 128)
        for family in ("gn", "kernel"):
            with self.subTest(family=family):
                torch.manual_seed(3)
                explicit = ThetaScan(
                    _config(family, feature_expansion=1, backend="naive")
                )
                torch.manual_seed(3)
                default = ThetaScan(
                    _config(
                        family,
                        feature_expansion=1,
                        expansion_key="unused-at-one",
                        backend="naive",
                    )
                )
                self.assertFalse(
                    any("_expand_" in k for k, _ in explicit.named_buffers())
                )
                self.assertNotIn(
                    "_core._expansion_fingerprint", explicit.state_dict()
                )
                torch.testing.assert_close(explicit(x), default(x))

    def test_validation_fails_closed(self) -> None:
        cases = (
            dict(feature_expansion=0),
            dict(feature_expansion=True),
            dict(feature_expansion=2.0),
            dict(memory_multiplier=1, feature_expansion=64),
            dict(expansion_key=""),
            dict(gn=GNConfig(nonlinearity="silu")),
            dict(
                family="kernel",
                kernel=KernelConfig(feature_map="softmax_partition"),
            ),
            dict(
                family="kernel",
                kernel=KernelConfig(feature_map="projected_bspline"),
            ),
        )
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises((ValueError, TypeError)):
                    _config(**case).validate()

    def test_effective_weights_compose_through_the_fixed_maps(self) -> None:
        mixer = ThetaScan(_config("gn", backend="naive")).double()
        core = mixer._core
        w1_trainable = core.W1[0].double()
        w2_trainable = core.W2[0].double()
        up = core._expand_w1_0.double()
        down = core._expand_w2_0.double()
        weights = core._weights()
        w1_effective, w2_effective = weights[0][0], weights[0][1]
        torch.testing.assert_close(
            w1_effective, torch.matmul(up, w1_trainable)
        )
        torch.testing.assert_close(
            w2_effective, torch.matmul(w2_trainable, down)
        )
        self.assertEqual(tuple(w1_effective.shape), (4, 192, 32))
        self.assertEqual(tuple(w2_effective.shape), (4, 32, 192))


if __name__ == "__main__":
    unittest.main()
