"""Fourier Neural Operator, 2D, self-contained.

Kept dependency-free (torch only) rather than importing `neuraloperator` so the
repo clones and runs without resolving a research-package dependency tree.  If
you prefer the reference implementation:

    pip install neuraloperator
    from neuralop.models import FNO
    model = FNO(n_modes=(32, 32), hidden_channels=64, in_channels=2, out_channels=2)

Either is fine.  The surrogate is deliberately NOT the contribution here -- the
point is that a *standard, competently trained* model fails the acoustics, so
using a stock architecture strengthens the argument.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )
        self.w2 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _mul(a, b):
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x):
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros(b, self.out_ch, h, w // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, h // 2)
        m2 = min(self.modes2, w // 2 + 1)
        out[:, :, :m1, :m2] = self._mul(x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = self._mul(x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(h, w), norm="ortho")


class FNO2d(nn.Module):
    """Maps a 2-channel velocity field to the next one."""

    def __init__(self, modes=24, width=48, n_layers=4, in_ch=2, out_ch=2):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, out_ch, 1)
        )

    def forward(self, x):
        x = self.lift(x)
        for s, p in zip(self.spectral, self.pointwise):
            x = torch.nn.functional.gelu(s(x) + p(x))
        return self.head(x)
