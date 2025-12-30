import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels=2, base_channels=16):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.flow_head = nn.Conv2d(base_channels, 2, kernel_size=3, padding=1)

    @staticmethod
    def _match_size(src, ref):
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.up3(bottleneck)
        dec3 = self.dec3(torch.cat([self._match_size(dec3, enc3), enc3], dim=1))
        dec2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([self._match_size(dec2, enc2), enc2], dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([self._match_size(dec1, enc1), enc1], dim=1))
        return self.flow_head(dec1)


class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _meshgrid(height, width, device):
        ys = torch.linspace(-1.0, 1.0, steps=height, device=device)
        xs = torch.linspace(-1.0, 1.0, steps=width, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return grid

    def forward(self, moving, flow):
        batch_size, _, height, width = moving.shape
        base_grid = self._meshgrid(height, width, moving.device)
        base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        flow_norm = torch.zeros_like(flow)
        flow_norm[:, 0, :, :] = flow[:, 0, :, :] / ((width - 1.0) / 2.0)
        flow_norm[:, 1, :, :] = flow[:, 1, :, :] / ((height - 1.0) / 2.0)
        flow_norm = flow_norm.permute(0, 2, 3, 1)

        sampling_grid = base_grid + flow_norm
        warped = F.grid_sample(moving, sampling_grid, align_corners=True)
        return warped


class VoxelMorph2D(nn.Module):
    def __init__(self, in_channels=2, base_channels=16):
        super().__init__()
        self.unet = UNet2D(in_channels=in_channels, base_channels=base_channels)
        self.transformer = SpatialTransformer()

    def forward(self, moving, fixed):
        x = torch.cat([moving, fixed], dim=1)
        flow = self.unet(x)
        warped = self.transformer(moving, flow)
        return warped, flow
