"""Learnable scale alignment for relative depth maps."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleFactorAligner(nn.Module):
    """Align relative depth to a positive metric-like scale.

    The module owns a single learnable scalar and multiplies incoming relative
    depths by the corresponding positive scale factor.  Two parameterizations
    are supported:

    * ``"log_scale"``: stores ``log(scale)`` and returns ``exp(log_scale)``.
    * ``"softplus"``: stores an unconstrained value initialized with the
      inverse-softplus transform and returns ``softplus(raw_scale)``.
    """

    def __init__(
        self,
        init_scale: float = 1.0,
        parameterization: str = "log_scale",
        eps: float = 1e-6,
    ):
        super().__init__()
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        if parameterization not in {"log_scale", "softplus"}:
            raise ValueError("parameterization must be 'log_scale' or 'softplus'.")

        self.parameterization = parameterization
        self.eps = eps

        if parameterization == "log_scale":
            init_value = math.log(init_scale)
            self.log_scale = nn.Parameter(torch.tensor(float(init_value)))
            self.raw_scale: Optional[nn.Parameter] = None
        else:
            init_value = self._inverse_softplus(init_scale)
            self.raw_scale = nn.Parameter(torch.tensor(float(init_value)))
            self.log_scale = None

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        """Numerically stable inverse of ``softplus`` for positive scalars."""

        if value > 20.0:
            return value
        return math.log(math.expm1(value))

    @property
    def scale_factor(self) -> torch.Tensor:
        """Return the positive scale factor used for depth alignment."""

        if self.parameterization == "log_scale":
            if self.log_scale is None:
                raise RuntimeError("log_scale parameter is not initialized.")
            return torch.exp(self.log_scale).clamp_min(self.eps)
        if self.raw_scale is None:
            raise RuntimeError("raw_scale parameter is not initialized.")
        return F.softplus(self.raw_scale).clamp_min(self.eps)

    def forward(self, relative_depth: torch.Tensor) -> torch.Tensor:
        """Scale relative depth into an aligned metric-like depth map."""

        return relative_depth * self.scale_factor.to(
            device=relative_depth.device,
            dtype=relative_depth.dtype,
        )
