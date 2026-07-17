from __future__ import annotations

import unittest
from unittest import mock

import torch

from thetascan._core.ops import scan_fla
from thetascan._core.ops.interface import resolve_backend


class BackendResolutionTests(unittest.TestCase):
    def test_auto_uses_the_actual_input_device(self) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(scan_fla, "supports", return_value=True),
        ):
            self.assertEqual(resolve_backend("auto", device="cpu"), "quad")
            self.assertEqual(resolve_backend("auto", device="cuda"), "fla")

    def test_auto_falls_back_when_required_fla_kernel_is_missing(self) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(scan_fla, "supports", return_value=False) as supports,
        ):
            self.assertEqual(resolve_backend("auto", device="cuda"), "quad")
            self.assertEqual(
                resolve_backend(
                    "auto",
                    device="cuda",
                    require_scalar_decay=True,
                ),
                "quad",
            )
        supports.assert_called_with("off", require_scalar_decay=True)

    def test_fla_capability_check_matches_required_kernel_kind(self) -> None:
        fake_kernels = {"linear": object(), "simple_gla": None, "gla": None}
        with mock.patch.object(scan_fla, "_load", return_value=fake_kernels):
            self.assertTrue(scan_fla.supports())
            self.assertFalse(scan_fla.supports("scalar"))
            self.assertFalse(scan_fla.supports(require_scalar_decay=True))
            self.assertFalse(scan_fla.supports("channel"))


if __name__ == "__main__":
    unittest.main()
