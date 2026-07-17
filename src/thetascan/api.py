"""Stable PyTorch modules exposed by the public ThetaScan package."""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from .config import ThetaScanConfig
from ._core.modules.block import ThetaScan as _CoreThetaScan


class ThetaScan(nn.Module):
    """A token mixer with the simple ``[batch, time, channels]`` contract."""

    def __init__(self, config: ThetaScanConfig):
        super().__init__()
        # A module is a snapshot of its construction config. Mutating the caller's
        # dataclass later cannot partially reconfigure already-created parameters.
        self.config = deepcopy(config)
        self.config.validate()
        self._core_config = self.config._to_core_config()
        self._core = _CoreThetaScan(self._core_config)

        if (self.config.family == "kernel"
                and not self.config.kernel.feature_parameters_trainable):
            frozen = list(self._core.key_query_feature_parameters())
            for parameter in frozen:
                parameter.requires_grad_(False)
            if not frozen:
                raise RuntimeError("could not locate kernel feature parameters to freeze")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._core(x)

    def zero_output_projection_(self) -> "ThetaScan":
        """Zero the final mixer projection in place and return ``self``.

        This is useful when a residual host requires every substituted mixer to
        start as the identity branch. It changes initialization only, not the
        memory algorithm or configuration.
        """
        nn.init.zeros_(self._core.proj_out.weight)
        return self

    def regularization_loss(self) -> torch.Tensor:
        """Return all optional regularization terms for adding to task loss."""
        return self._core.ortho_loss()
