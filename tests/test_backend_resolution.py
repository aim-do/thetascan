from __future__ import annotations

import unittest
from unittest import mock

import torch

from thetascan._core.ops import scan_fla
from thetascan._core.ops.interface import resolve_backend


class BackendResolutionTests(unittest.TestCase):
    def test_auto_always_selects_the_portable_chunk_backend(self) -> None:
        """The default is a measured choice, not a capability probe.

        It must not depend on the device, on whether CUDA is present, or on
        whether an optional package happens to be importable, because none of
        those made the faster choice in measurement.
        """
        for available in (False, True):
            with mock.patch.object(
                torch.cuda, "is_available", return_value=available
            ):
                self.assertEqual(resolve_backend("auto"), "chunk")
                self.assertEqual(resolve_backend("auto", device="cpu"), "chunk")
                self.assertEqual(resolve_backend("auto", device="cuda"), "chunk")

    def test_auto_never_selects_fla_for_any_temporal_mode(self) -> None:
        """FLA refuses its gated backward for a retention, and loses anyway.

        Its chunked gated backward is refused for any retention on Hopper with
        Triton >= 3.4, so every reference preset would fail on its first
        backward. For the ungated plain sum it does support, it measured slower
        than the portable tile: a compiled model breaks into several graphs
        around its compiler-disabled wrapper, and substituting the scan leaves
        the rest of a long-context read untouched.
        """
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(scan_fla, "supports", return_value=True) as supports,
        ):
            for kwargs in (
                {},
                {"require_scalar_decay": True},
                {"decay_gate": "scalar"},
                {"accumulation": "ema_gate"},
            ):
                with self.subTest(**kwargs):
                    self.assertEqual(
                        resolve_backend("auto", device="cuda", **kwargs), "chunk"
                    )
        supports.assert_not_called()

    def test_explicit_backends_are_never_overridden(self) -> None:
        for name in ("naive", "quad", "chunk", "cumsum", "fla"):
            with self.subTest(name=name):
                self.assertEqual(resolve_backend(name, device="cpu"), name)

    def test_fla_capability_check_matches_required_kernel_kind(self) -> None:
        """Still used by callers deciding whether an explicit `fla` can run."""
        fake_kernels = {"linear": object(), "simple_gla": None, "gla": None}
        with mock.patch.object(scan_fla, "_load", return_value=fake_kernels):
            self.assertTrue(scan_fla.supports())
            self.assertFalse(scan_fla.supports("scalar"))
            self.assertFalse(scan_fla.supports(require_scalar_decay=True))
            self.assertFalse(scan_fla.supports("channel"))


if __name__ == "__main__":
    unittest.main()
