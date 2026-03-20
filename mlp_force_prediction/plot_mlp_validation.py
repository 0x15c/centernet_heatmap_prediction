import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

hidden_dims = (128, 64, 16)
class ForceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, int, int] = hidden_dims):
        super().__init__()
        h1, h2, h3= hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LeakyReLU(),
            nn.Linear(h1, h2),
            nn.LeakyReLU(),
            nn.Linear(h2, h3),
            nn.LeakyReLU(),
            nn.Linear(h3, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PointNetRegressor(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()

        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim,16),
            nn.LeakyReLU(),
            nn.Linear(16,32),
            nn.LeakyReLU(),
            nn.Linear(32,32)
        )

        self.regressor = nn.Sequential(
            nn.Linear(32,16),
            nn.LeakyReLU(),
            nn.Linear(16,3)
        )

    def forward(self,x):
        # x shape: (batch, n_points, 4) (2048, 50, 4)
        pass
        feat = self.point_encoder(x)        # (B,N,128)
        global_feat = feat.max(dim=1)[0]    # max pooling

        out = self.regressor(global_feat)
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot validation predicted force vs ground truth force and MAE."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset .pt from generate_mlp_dataset.py")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint from train_mlp_force_regressor.py")
    parser.add_argument("--output", type=Path, required=True, help="Output image path (.png)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    return parser.parse_args()


def select_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    data: Dict = torch.load(args.dataset, map_location="cpu")
    ckpt: Dict = torch.load(args.checkpoint, map_location="cpu")

    # x_val = data["val"]["x"].reshape(data["val"]["x"].shape[0], -1).float()
    x_val = data["val"]["x"].float()
    y_val = data["val"]["y"].float()

    # data["x"] = torch.cat((data["train"]["x"], data["val"]["x"]), dim=0)
    # data["y"] = torch.cat((data["train"]["y"], data["val"]["y"]), dim=0)
    # x_val = data["x"].float()
    # y_val = data["y"].float()
    

    x_mean = ckpt["norm"]["x_mean"].float()
    x_std = ckpt["norm"]["x_std"].float().clamp_min(1e-6)
    y_mean = ckpt["norm"]["y_mean"].float()
    y_std = ckpt["norm"]["y_std"].float().clamp_min(1e-6)

    x_val_norm = (x_val - x_mean) / x_std

    cfg = ckpt["model_config"]
    # model = PointNetRegressor(input_dim=int(cfg["input_dim"]), hidden_dims=tuple(cfg["hidden_dims"]))
    model = PointNetRegressor(input_dim=int(cfg["input_dim"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred_norm = model(x_val_norm.to(device)).cpu()
    pred = pred_norm * y_std + y_mean

    # with torch.no_grad():
    #     pred = model(x_val.to(device)).cpu()

    mae_axes = (pred - y_val).abs().mean(dim=0)
    mae_all = (pred - y_val).abs().mean()

    pred_np = pred.numpy()
    y_np = y_val.numpy()

    labels = ["Fx", "Fy", "Fz"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, ax in enumerate(axes):
        gt = y_np[:, i]
        pr = pred_np[:, i]
        low = min(gt.min(), pr.min())
        high = max(gt.max(), pr.max())
        ax.scatter(gt, pr, s=8, alpha=0.5)
        ax.plot([low, high], [low, high], "r--", linewidth=1.5)
        ax.set_title(f"{labels[i]}  MAE={mae_axes[i].item():.4f}")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Validation Predicted vs Ground Truth | Overall MAE={mae_all.item():.4f}", fontsize=13)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=220)
    plt.close(fig)

    print(f"Validation samples: {x_val.shape[0]}")
    print(f"MAE Fx={mae_axes[0].item():.6f}, Fy={mae_axes[1].item():.6f}, Fz={mae_axes[2].item():.6f}")
    print(f"Overall MAE: {mae_all.item():.6f}")
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
