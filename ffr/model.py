"""A small 3D U-Net that maps a skull signed-distance field to a skin signed-distance field."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _norm(kind: str, ch: int) -> nn.Module:
    if kind == "instance":
        return nn.InstanceNorm3d(ch, affine=True)
    if kind == "group":
        return nn.GroupNorm(min(8, ch), ch)
    if kind == "batch":
        return nn.BatchNorm3d(ch)
    raise ValueError(kind)


class ConvBlock(nn.Module):
    """Two 3x3x3 convs; optional residual shortcut."""

    def __init__(self, cin: int, cout: int, norm: str = "instance", residual: bool = True, dropout: float = 0.0):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1, bias=False)
        self.n1 = _norm(norm, cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.n2 = _norm(norm, cout)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (nn.Conv3d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()) if residual else None

    def forward(self, x):
        h = F.leaky_relu(self.n1(self.c1(x)), 0.01)
        h = self.n2(self.c2(self.drop(h)))
        if self.skip is not None:
            h = h + self.skip(x)
        return F.leaky_relu(h, 0.01)


class UNet3D(nn.Module):
    """
    Encoder-decoder with skip connections (Cicek et al., MICCAI 2016) in a compact form.

    in_channels:  1 (skull SDF) or 4 with CoordConv channels (z, y, x in [-1, 1]).
    out:          skin SDF in units of the truncation distance, squashed with tanh.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 16, depth: int = 4,
                 norm: str = "instance", residual: bool = True, skip_connections: bool = True,
                 dropout: float = 0.0, predict_residual: bool = False):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth)]
        self.skip_connections = skip_connections
        self.predict_residual = predict_residual
        self.enc = nn.ModuleList()
        c = in_channels
        for i, ch in enumerate(chs):
            self.enc.append(ConvBlock(c, ch, norm, residual, dropout if i == depth - 1 else 0.0))
            c = ch
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose3d(c, ch, 2, stride=2))
            self.dec.append(ConvBlock(ch * 2 if skip_connections else ch, ch, norm, residual))
            c = ch
        self.head = nn.Conv3d(c, out_channels, 1)

    def forward(self, x):
        skips = []
        h = x
        for i, block in enumerate(self.enc):
            h = block(h)
            if i < len(self.enc) - 1:
                skips.append(h)
                h = F.max_pool3d(h, 2)
        for up, block in zip(self.up, self.dec):
            h = up(h)
            s = skips.pop()
            h = block(torch.cat([h, s], 1) if self.skip_connections else h)
        out = self.head(h)
        if self.predict_residual:
            # Predict skin SDF as skull SDF minus a learned soft-tissue offset map.
            return torch.tanh(torch.atanh(x[:, :1].clamp(-0.999, 0.999)) + out)
        return torch.tanh(out)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
