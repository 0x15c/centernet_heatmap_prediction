import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
z_axis_loss_coeff = 0.0
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
        pass
        return self.net(x)
    
    
class PointNetRegressor(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()

        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.LeakyReLU(),
            nn.Linear(64,128),
            nn.LeakyReLU(),
            nn.Linear(128,128)
        )

        self.regressor = nn.Sequential(
            nn.Linear(128,64),
            nn.LeakyReLU(),
            nn.Linear(64,3)
        )

    def forward(self,x):
        # x shape: (batch, n_points, 4) (2048, 50, 4)
        pass
        feat = self.point_encoder(x)        # (B,N,128)
        global_feat = feat.max(dim=1)[0]    # max pooling

        out = self.regressor(global_feat)
        return out

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MLP force regressor with MSE loss.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset .pt from generate_mlp_dataset.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save checkpoint/history/plots")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs (default: 120)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay (default: 1e-5)")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size. 0 uses dataset batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    return parser.parse_args()


def select_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def standardize(train_x: torch.Tensor, val_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (val_x - mean) / std, mean, std


def standardize_targets(train_y: torch.Tensor, val_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_y.mean(dim=0, keepdim=True)
    std = train_y.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_y - mean) / std, (val_y - mean) / std, mean, std


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            # loss = criterion(pred[:,0:2], yb[:,0:2])+z_axis_loss_coeff*criterion(pred[:,2],yb[:,2])
            bs = xb.shape[0]
            total_loss += loss.item() * bs
            total_count += bs
    return total_loss / max(total_count, 1)



def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    data: Dict = torch.load(args.dataset, map_location="cpu")

    ds_batch = int(data["meta"]["batch_size"])
    batch_size = args.batch_size if args.batch_size > 0 else ds_batch

    # train_x = data["train"]["x"].reshape(data["train"]["x"].shape[0], -1).float()
    train_x = data["train"]["x"].float()
    train_y = data["train"]["y"].float()
    # val_x = data["val"]["x"].reshape(data["val"]["x"].shape[0], -1).float()
    val_x = data["val"]["x"].float()
    val_y = data["val"]["y"].float()

    train_x, val_x, x_mean, x_std = standardize(train_x, val_x)
    train_y, val_y, y_mean, y_std = standardize_targets(train_y, val_y)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = PointNetRegressor(input_dim=train_x.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            # loss = criterion(pred[:,0:2], yb[:,0:2])+z_axis_loss_coeff*criterion(pred[:,2],yb[:,2])
            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            running += loss.item() * bs
            count += bs

        train_loss = running / max(count, 1)
        val_loss = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train_mse={train_loss:.6f} | val_mse={val_loss:.6f}"
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "mlp_force_model.pt"
    history_path = args.output_dir / "training_history.json"
    curve_path = args.output_dir / "learning_curve.png"

    checkpoint = {
        "model_state": best_state,
        "model_config": {
            "input_dim": int(train_x.shape[2]),
            "hidden_dims": hidden_dims,
        },
        "norm": {
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
        },
        "train_config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": batch_size,
            "seed": args.seed,
            "best_val_mse": best_val,
        },
        "dataset_meta": data["meta"],
    }

    torch.save(checkpoint, ckpt_path)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train MSE")
    plt.plot(history["val_loss"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("MLP Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=180)
    plt.close()

    print(f"Best validation MSE: {best_val:.6f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"History:    {history_path}")
    print(f"Curve:      {curve_path}")


if __name__ == "__main__":
    main()
