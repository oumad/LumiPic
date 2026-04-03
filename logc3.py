"""ARRI LogC3 (EI 800) transfer function.

Provides compression/decompression between scene-linear HDR values
and a perceptually uniform [0, 1] range suitable for VAE encoding.

Reference: ARRI LogC3 specification
https://www.arri.com/en/learn-help/learn-help-camera-system/technical-information/about-log-c
"""

import torch
import numpy as np


class LogC3:
    """ARRI LogC3 transfer function (Exposure Index 800).

    Representable range: [0, ~55.1] scene-linear
    Encoded range: [0, 1]

    Key values:
        0.18 linear (mid-gray)  → 0.391 LogC3
        1.0 linear (white)      → 0.525 LogC3
        10.0 linear             → 0.727 LogC3
        55.1 linear (ceiling)   → 1.0 LogC3
    """
    A = 5.555556
    B = 0.052272
    C = 0.247190
    D = 0.385537
    E = 5.367655
    F = 0.092809
    CUT = 0.010591

    def compress(self, linear: torch.Tensor) -> torch.Tensor:
        """Scene-linear → LogC3 [0, 1]."""
        linear = torch.clamp(linear, min=0.0)
        log_val = self.C * torch.log10(self.A * linear + self.B) + self.D
        lin_val = self.E * linear + self.F
        return torch.where(linear > self.CUT, log_val, lin_val)

    def decompress(self, logc: torch.Tensor) -> torch.Tensor:
        """LogC3 [0, 1] → scene-linear."""
        logc = torch.clamp(logc, 0.0, 1.0)
        cut_log = self.E * self.CUT + self.F
        linear_from_log = (torch.pow(10.0, (logc - self.D) / self.C) - self.B) / self.A
        linear_from_lin = (logc - self.F) / self.E
        return torch.where(logc >= cut_log, linear_from_log, linear_from_lin)

    def compress_numpy(self, linear: np.ndarray) -> np.ndarray:
        """Scene-linear → LogC3 [0, 1] (numpy version)."""
        linear = np.maximum(linear, 0.0)
        log_val = self.C * np.log10(self.A * linear + self.B) + self.D
        lin_val = self.E * linear + self.F
        return np.where(linear > self.CUT, log_val, lin_val)

    def decompress_numpy(self, logc: np.ndarray) -> np.ndarray:
        """LogC3 [0, 1] → scene-linear (numpy version)."""
        logc = np.clip(logc, 0.0, 1.0)
        cut_log = self.E * self.CUT + self.F
        linear_from_log = (np.power(10.0, (logc - self.D) / self.C) - self.B) / self.A
        linear_from_lin = (logc - self.F) / self.E
        return np.where(logc >= cut_log, linear_from_log, linear_from_lin)
