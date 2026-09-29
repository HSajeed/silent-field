"""Circular-padded 2D UNet baseline.

Included so the result is not architecture-specific: if both a spectral model
and a convolutional one lose the acoustics, the cause is the objective, not the
inductive bias. That distinction is the whole claim, so the second architecture
is not optional.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular"),
        nn.GroupNorm(min(8, cout), cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, padding_mode="circular"),
        nn.GroupNorm(min(8, cout), cout), nn.GELU(),
    )


class UNet2d(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, base=48, depth=3):
        super().__init__()
        chans = [base * 2 ** i for i in range(depth)]
        self.downs = nn.ModuleList()
        c = in_ch
        for ch in chans:
            self.downs.append(block(c, ch)); c = ch
        self.pool = nn.AvgPool2d(2)
        self.mid = block(c, c * 2); c = c * 2
        self.ups, self.decs = nn.ModuleList(), nn.ModuleList()
        for ch in reversed(chans):
            self.ups.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.decs.append(block(ch * 2, ch)); c = ch
        self.head = nn.Conv2d(c, out_ch, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x); skips.append(x); x = self.pool(x)
        x = self.mid(x)
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)
