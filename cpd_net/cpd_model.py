import torch
import torch.nn as nn

class SharedMLP(nn.Module):
    """
    Shared MLP implemented with 1x1 convolutions.

    In the original TensorFlow+sugartensor code, the network used sg_conv with
    kernel size (1, 1) applied to per-point features. A 1x1 Conv1d does the
    same thing when the data is shaped as (B, C, N).
    """

    def __init__(self, in_channels: int, channels: list[int]) -> None:
        super().__init__()
        layers = []
        for out_channels in channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the stacked 1x1 convolutions."""
        return self.net(x)

class PointRegressor(nn.Module):
    """
    PyTorch reimplementation of the original TensorFlow CPD-Net.

    The model takes a source point set and a target point set and predicts a
    per-point displacement for the source. The architecture preserves the two
    separate MLP branches from the original code (gen* and gen9* blocks), then
    fuses the point-wise source coordinates with global features from both
    branches before regressing the displacement vectors.
    """

    def __init__(self) -> None:
        super().__init__()
        # Target branch (corresponds to gen9/gen1/gen2/gen3/gen4 in TF code).
        self.target_mlp = SharedMLP(2, [16, 64, 128, 256, 512])
        # Source branch (corresponds to gen99/gen11/gen22/gen33/gen44 in TF code).
        self.source_mlp = SharedMLP(2, [16, 64, 128, 256, 512])

        # Fusion head (f1/f2/f3 in TF code).
        self.fusion = nn.Sequential(
            nn.Conv1d(2 + 512 + 512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(128, 2, kernel_size=1),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: (B, N, 2) tensor of source points.
            target: (B, N, 2) tensor of target points.

        Returns:
            (B, N, 2) tensor of displacement vectors to apply to source points.
        """
        # Convert to (B, C, N) layout for Conv1d layers.
        source_channels = source.transpose(1, 2)
        target_channels = target.transpose(1, 2)

        # Extract per-point features for the target branch.
        target_features = self.target_mlp(target_channels)
        target_global = torch.max(target_features, dim=2, keepdim=True).values
        target_global = target_global.expand(-1, -1, target_features.shape[2])

        # Extract per-point features for the source branch.
        source_features = self.source_mlp(source_channels)
        source_global = torch.max(source_features, dim=2, keepdim=True).values
        source_global = source_global.expand(-1, -1, source_features.shape[2])

        # Fuse raw coordinates with global features from both branches.
        fused = torch.cat([source_channels, source_global, target_global], dim=1)
        displacement = self.fusion(fused)

        # Return to (B, N, 2) layout.
        return displacement.transpose(1, 2)
    
def chamfer_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the symmetric Chamfer distance between two point sets.

    This mirrors the TensorFlow implementation in the original repo by
    calculating pairwise distances, taking the minimum in each direction, and
    averaging the results.
    """
    # pred/target: (B, N, 2)
    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    dist = (diff ** 2).sum(dim=-1)
    min_pred = dist.min(dim=2).values
    min_target = dist.min(dim=1).values
    return ((min_pred + min_target) / 2.0).mean()