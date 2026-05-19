"""Losses and image-warp helpers for QCRegNet tactile registration."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .models import mapping01_to_grid


def warp_with_mapping(image: torch.Tensor, mapping01: torch.Tensor, *, mode: str = "bilinear") -> torch.Tensor:
    """Warp image with a QCRegNet mapping in [0, 1] coordinates.

    image:     B x C x H x W
    mapping01: B x 2 x h x w. It is upsampled to image size if needed.
    """
    if mapping01.shape[-2:] != image.shape[-2:]:
        mapping01 = F.interpolate(mapping01, size=image.shape[-2:], mode="bilinear", align_corners=True)
    grid = mapping01_to_grid(mapping01)
    return F.grid_sample(image, grid, mode=mode, padding_mode="border", align_corners=True)


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def image_similarity_loss(warped: torch.Tensor, fixed: torch.Tensor, mode: str = "charbonnier") -> torch.Tensor:
    """Similarity between warped image and fixed image."""
    if mode == "mse":
        return F.mse_loss(warped, fixed)
    if mode == "l1":
        return F.l1_loss(warped, fixed)
    if mode == "charbonnier":
        return charbonnier(warped - fixed).mean()
    raise ValueError(f"Unknown similarity mode: {mode}")


def mu_magnitude_loss(raw_mu: torch.Tensor) -> torch.Tensor:
    """Suppress excessive distortion in the unconstrained mu logits."""
    return (raw_mu * raw_mu).mean()


def laplacian_loss(field: torch.Tensor) -> torch.Tensor:
    """Laplacian smoothness on a BxCxHxW field."""
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=field.dtype,
        device=field.device,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(field.size(1), 1, 1, 1)
    lap = F.conv2d(field, kernel, padding=1, groups=field.size(1))
    return (lap * lap).mean()


def mapping_det_jacobian_loss(mapping01: torch.Tensor) -> torch.Tensor:
    """Penalize local orientation reversal in the normalized mapping.

    This is optional because the Beltrami parameterization already encourages
    quasiconformal maps.  It is useful when BSNet is trained from scratch.
    """
    x = mapping01[:, 0:1]
    y = mapping01[:, 1:2]

    x_x = (x[:, :, :, 2:] - x[:, :, :, :-2]) / 2.0
    y_x = (y[:, :, :, 2:] - y[:, :, :, :-2]) / 2.0
    x_y = (x[:, :, 2:, :] - x[:, :, :-2, :]) / 2.0
    y_y = (y[:, :, 2:, :] - y[:, :, :-2, :]) / 2.0

    # Crop to common interior region.
    x_x = x_x[:, :, 1:-1, :]
    y_x = y_x[:, :, 1:-1, :]
    x_y = x_y[:, :, :, 1:-1]
    y_y = y_y[:, :, :, 1:-1]

    det = x_x * y_y - x_y * y_x
    return F.relu(-det).mean()


def ncc_loss(warped: torch.Tensor, fixed: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Global normalized cross-correlation loss, useful for heatmaps."""
    w = warped - warped.mean(dim=(-2, -1), keepdim=True)
    f = fixed - fixed.mean(dim=(-2, -1), keepdim=True)
    numerator = (w * f).mean(dim=(-2, -1))
    denominator = torch.sqrt((w * w).mean(dim=(-2, -1)) * (f * f).mean(dim=(-2, -1)) + eps)
    return 1.0 - (numerator / denominator).mean()
