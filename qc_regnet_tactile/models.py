"""QCRegNet-style models for tactile image registration.

This file adapts the public QCRegNet idea to this repo while keeping the code
self-contained.  The model does not directly predict a free dense flow field.
Instead:

    Estimator([fixed, moving]) -> bounded Beltrami coefficient mu
    BSNet(mu)                 -> quasiconformal mapping in normalized coords

The mapping can be passed to torch.nn.functional.grid_sample after conversion
from [0, 1] coordinates to [-1, 1] coordinates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None, *, leaky: bool = False):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        activation: nn.Module
        if leaky:
            activation = nn.LeakyReLU(inplace=True)
        else:
            activation = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            activation,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, leaky: bool = False):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, leaky=leaky))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, bilinear: bool = True, leaky: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, leaky=leaky)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, leaky=leaky)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor | None = None) -> torch.Tensor:
        x1 = self.up(x1)
        if x2 is not None:
            diff_y = x2.size(2) - x1.size(2)
            diff_x = x2.size(3) - x1.size(3)
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Estimator(nn.Module):
    """Predict a bounded Beltrami coefficient field from a fixed/moving pair.

    Output:
        mu:     B x 2 x H/2 x W/2, real/imag channels, with |mu| < 1 by tanh.
        raw_mu: unconstrained raw field before tanh-normalization.

    The half-resolution output matches the original QCRegNet training design,
    where BSNet produces a half-resolution mapping and the caller upsamples it.
    """

    def __init__(self, n_channels: int = 2, base_channels: int = 16, bilinear: bool = True):
        super().__init__()
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear=bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear=bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear=bilinear)
        # Deliberately omit the final up block: output is half resolution.
        self.outc = OutConv(base_channels * 2 // factor, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        raw_mu = self.outc(x)

        norm = torch.linalg.norm(raw_mu, dim=1, keepdim=True).clamp_min(1e-6)
        direction = raw_mu / norm
        mu = torch.tanh(norm) * direction
        return mu, raw_mu


class FFTCrop(nn.Module):
    """Take a centered low-frequency crop of a two-channel complex field."""

    def __init__(self, crop_h: int, crop_w: int):
        super().__init__()
        self.crop_h = crop_h
        self.crop_w = crop_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 2:
            raise ValueError(f"FFTCrop expects 2 channels, got {x.size(1)}")
        _, _, rows, cols = x.shape
        complex_x = torch.complex(x[:, 0], x[:, 1])
        freq = torch.fft.fftshift(torch.fft.fft2(complex_x), dim=(-2, -1))
        cy, cx = rows // 2, cols // 2
        rh, rw = self.crop_h // 2, self.crop_w // 2
        freq = freq[:, cy - rh : cy - rh + self.crop_h, cx - rw : cx - rw + self.crop_w]
        return torch.stack([freq.real, freq.imag], dim=1)


class DTL(nn.Module):
    """Lightweight complex-valued mixing layer used by QCRegNet BSNet."""

    def __init__(self, height: int, width: int):
        super().__init__()
        self.conv1r = nn.Conv2d(height, height, kernel_size=1, padding=0, bias=False)
        self.conv1i = nn.Conv2d(height, height, kernel_size=1, padding=0, bias=False)
        self.conv2r = nn.Conv2d(width, width, kernel_size=1, padding=0, bias=False)
        self.conv2i = nn.Conv2d(width, width, kernel_size=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x 2 x H x W, interpreted as complex HxW coefficients.
        x_pmt = x.permute(0, 2, 3, 1)  # B x H x W x 2
        x1r = self.conv1r(x_pmt)
        x1i = self.conv1i(x_pmt)
        real = x1r[..., 0] - x1i[..., 1]
        imag = x1i[..., 0] + x1r[..., 1]
        x2 = torch.stack((real, imag), dim=1)  # B x 2 x H x W

        x3 = x2.permute(0, 3, 2, 1)  # B x W x H x 2
        x4r = self.conv2r(x3)
        x4i = self.conv2i(x3)
        real = x4r[..., 0] - x4i[..., 1]
        imag = x4i[..., 0] + x4r[..., 1]
        x5 = torch.stack((real, imag), dim=1)  # B x 2 x W x H
        return x5.permute(0, 1, 3, 2)  # B x 2 x H x W


class BSNet(nn.Module):
    """Map a Beltrami coefficient field to a normalized coordinate mapping.

    This adapts the public QCRegNet BSNet.  It is meant to be loaded from a
    pretrained BSNet checkpoint when possible.  If no pretrained checkpoint is
    available, train_bsnet_synthetic.py should be used first.

    Args:
        mu_size: half-resolution size of the mu field, e.g. (112, 112) for a
            224x224 registration image.
    """

    def __init__(self, mu_size: tuple[int, int], n_channels: int = 2, n_classes: int = 2, bilinear: bool = True):
        super().__init__()
        h, w = mu_size
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError(f"mu_size should be divisible by 8, got {mu_size}")
        crop_h, crop_w = h // 8, w // 8
        self.fft = FFTCrop(crop_h, crop_w)
        self.dtl = DTL(crop_h, crop_w)

        self.inc2 = DoubleConv(n_channels, 16, leaky=True)
        self.down = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(16), nn.LeakyReLU(inplace=True))
        self.inc = DoubleConv(2, 64, leaky=True)
        self.up1 = Up(64, 64, bilinear=bilinear, leaky=True)
        self.up2 = Up(64 + 16, 64, bilinear=bilinear, leaky=True)
        self.up3 = Up(64, 48, bilinear=bilinear, leaky=True)
        self.outc = OutConv(48, n_classes, kernel_size=3)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        # Low-frequency complex features from mu.
        x1 = self.fft(mu)
        x1 = self.dtl(x1)

        # Local spatial branch.
        x2 = self.inc2(mu)
        x2 = self.down(x2)

        x = self.inc(x1)
        x = self.up1(x)
        x = self.up2(x, x2)
        x = self.up3(x)
        mapping = self.outc(x)

        # Boundary anchoring, following QCRegNet. Mapping is in [0, 1].
        mapping = mapping.clone()
        mapping[:, 0, :, 0] = -0.5
        mapping[:, 0, :, -1] = 0.5
        mapping[:, 1, 0, :] = 0.5
        mapping[:, 1, -1, :] = -0.5
        return mapping + 0.5


def mapping01_to_grid(mapping: torch.Tensor) -> torch.Tensor:
    """Convert QCRegNet mapping in [0, 1] to grid_sample grid in [-1, 1].

    QCRegNet stores y with image-up convention, while grid_sample uses image-down
    coordinates.  Therefore the y channel is sign-flipped after scaling.
    """
    grid = mapping.permute(0, 2, 3, 1) * 2.0 - 1.0
    grid = grid.clone()
    grid[..., 1] = -grid[..., 1]
    return grid


def identity_mapping(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return B x 2 x H x W normalized identity mapping in QCRegNet convention."""
    y, x = torch.meshgrid(
        torch.linspace(1.0, 0.0, height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    mapping = torch.stack([x, y], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mapping


def mapping_to_flow_pixels(mapping: torch.Tensor) -> torch.Tensor:
    """Convert normalized mapping Bx2xHxW to pixel displacement Bx2xHxW."""
    b, _, h, w = mapping.shape
    ident = identity_mapping(b, h, w, mapping.device, mapping.dtype)
    flow_norm = mapping - ident
    flow = torch.empty_like(flow_norm)
    flow[:, 0] = flow_norm[:, 0] * (w - 1)
    # y coordinate in mapping is image-up, so positive pixel flow is image-down.
    flow[:, 1] = -flow_norm[:, 1] * (h - 1)
    return flow
