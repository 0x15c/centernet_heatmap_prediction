import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    import wandb
except ImportError as exc:
    raise RuntimeError(
        "wandb is required for this script. Install it with `pip install wandb`."
    ) from exc


DATA_DIR = Path("CNN_dataset/data")
IMG_DIR = Path("CNN_dataset/raw_frames")
OUTPUT_DIR = Path("CNN_dataset/ckpt")

DEFAULT_TRAIN_RATIO = 0.75
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 50
DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 224


class CNNForceRegressor(nn.Module):
    def __init__(self, output_dim: int = 3, use_pretrained: bool = True):
        super().__init__()
        weights = None
        if use_pretrained:
            try:
                weights = models.ResNet34_Weights.DEFAULT
            except Exception:
                weights = None

        try:
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            self.backbone = models.resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class CNNRegressorDataset(Dataset):
    def __init__(self, img_paths: list[Path], labels: list[list[float]], transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        force_label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, force_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN regressor for 3D contact force prediction."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--img-dir", type=Path, default=IMG_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--project", type=str, default="cnn-force-regression")
    parser.add_argument("--run-name", type=str, default="resnet18-force-regressor")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights for ResNet-18.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(data_dir: Path, img_dir: Path) -> tuple[list[Path], list[list[float]]]:
    txt_files = sorted(data_dir.glob("data_*.txt"))
    img_paths: list[Path] = []
    force_labels: list[list[float]] = []

    for txt_file in txt_files:
        img_name = txt_file.name.replace("data_", "raw_").replace(".txt", ".png")
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        line = txt_file.read_text(encoding="utf-8").strip()
        if not line:
            continue

        values = [float(x) for x in line.split(",")]
        if len(values) < 3:
            continue

        img_paths.append(img_path)
        force_labels.append(values[:3])

    return img_paths, force_labels


def split_data(
    img_paths: list[Path],
    force_labels: list[list[float]],
    train_ratio: float,
    seed: int,
) -> tuple[list[Path], list[list[float]], list[Path], list[list[float]]]:
    if len(img_paths) < 2:
        raise RuntimeError("Need at least 2 samples to build train/validation splits.")

    indices = list(range(len(img_paths)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    split_idx = int(len(indices) * train_ratio)
    split_idx = min(max(split_idx, 1), len(indices) - 1)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_imgs = [img_paths[i] for i in train_idx]
    train_labels = [force_labels[i] for i in train_idx]
    val_imgs = [img_paths[i] for i in val_idx]
    val_labels = [force_labels[i] for i in val_idx]
    return train_imgs, train_labels, val_imgs, val_labels


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_axis_sq_error = torch.zeros(3, dtype=torch.float64)

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            predictions = model(images)
            loss = criterion(predictions, targets)

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size
            total_axis_sq_error += ((predictions - targets) ** 2).sum(dim=0).cpu().double()

    mean_loss = total_loss / max(total_count, 1)
    axis_mse = (total_axis_sq_error / max(total_count, 1)).numpy()
    return mean_loss, axis_mse


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    gt_batches = []
    pred_batches = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            predictions = model(images).cpu().numpy()
            gt_batches.append(targets.numpy())
            pred_batches.append(predictions)

    return np.concatenate(gt_batches, axis=0), np.concatenate(pred_batches, axis=0)


def plot_learning_curve(history: dict[str, list[float]], save_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train MSE")
    plt.plot(history["val_loss"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("CNN Force Regression Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_gt_vs_prediction(ground_truth: np.ndarray, predictions: np.ndarray, save_path: Path) -> None:
    axis_names = ["fx", "fy", "fz"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    overall_mae = float(np.mean(np.abs(predictions - ground_truth)))

    for idx, axis_name in enumerate(axis_names):
        gt_axis = ground_truth[:, idx]
        pred_axis = predictions[:, idx]
        axis_min = min(gt_axis.min(), pred_axis.min())
        axis_max = max(gt_axis.max(), pred_axis.max())
        axis_mae = float(np.mean(np.abs(pred_axis - gt_axis)))

        axes[idx].scatter(gt_axis, pred_axis, s=18, alpha=0.7)
        axes[idx].plot([axis_min, axis_max], [axis_min, axis_max], "r--", linewidth=1.5)
        axes[idx].set_xlabel(f"Ground Truth {axis_name}")
        axes[idx].set_ylabel(f"Prediction {axis_name}")
        axes[idx].set_title(f"{axis_name.upper()} Prediction")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].text(
            0.05,
            0.95,
            f"MAE = {axis_mae:.4f}",
            transform=axes[idx].transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    fig.suptitle(f"Ground Truth vs Prediction on Validation Set | Overall MAE = {overall_mae:.4f}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    img_paths, force_labels = load_data(args.data_dir, args.img_dir)
    if not img_paths:
        raise RuntimeError(
            f"No valid training pairs found in {args.data_dir} and {args.img_dir}."
        )

    train_imgs, train_labels, val_imgs, val_labels = split_data(
        img_paths, force_labels, args.train_ratio, args.seed
    )

    transform = build_transform(args.image_size)
    train_ds = CNNRegressorDataset(train_imgs, train_labels, transform=transform)
    val_ds = CNNRegressorDataset(val_imgs, val_labels, transform=transform)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = CNNForceRegressor(output_dim=3, use_pretrained=not args.no_pretrained).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=args.project,
        name=args.run_name,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "train_ratio": args.train_ratio,
            "image_size": args.image_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "num_train_samples": len(train_ds),
            "num_val_samples": len(val_ds),
            "label_format": ["fx", "fy", "fz"],
            "model": "resnet18",
            "pretrained": not args.no_pretrained,
        },
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            batch_size = images.shape[0]
            running_loss += loss.item() * batch_size
            sample_count += batch_size

        train_loss = running_loss / max(sample_count, 1)
        val_loss, val_axis_mse = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        wandb.log(
            {
                "epoch": epoch,
                "train/mse": train_loss,
                "val/mse": val_loss,
                "val/mse_fx": float(val_axis_mse[0]),
                "val/mse_fy": float(val_axis_mse[1]),
                "val/mse_fz": float(val_axis_mse[2]),
                "train/lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_mse={train_loss:.6f} | "
            f"val_mse={val_loss:.6f} | "
            f"val_axis_mse=[{val_axis_mse[0]:.6f}, {val_axis_mse[1]:.6f}, {val_axis_mse[2]:.6f}]"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    checkpoint_path = args.output_dir / "cnn_force_regressor_best.pt"
    history_path = args.output_dir / "cnn_training_history.json"
    curve_path = args.output_dir / "cnn_learning_curve.png"
    scatter_path = args.output_dir / "cnn_gt_vs_prediction.png"

    torch.save(
        {
            "model_state": best_state,
            "model_config": {
                "output_dim": 3,
                "model_name": "resnet18",
                "image_size": args.image_size,
                "pretrained": not args.no_pretrained,
            },
            "train_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "train_ratio": args.train_ratio,
                "seed": args.seed,
                "best_val_mse": best_val_loss,
                "label_format": ["fx", "fy", "fz"],
            },
        },
        checkpoint_path,
    )

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_learning_curve(history, curve_path)

    val_ground_truth, val_predictions = collect_predictions(model, val_loader, device)
    plot_gt_vs_prediction(val_ground_truth, val_predictions, scatter_path)

    wandb.log(
        {
            "best/val_mse": best_val_loss,
            "viz/learning_curve": wandb.Image(str(curve_path)),
            "viz/gt_vs_prediction": wandb.Image(str(scatter_path)),
        }
    )
    wandb.finish()

    print(f"Best validation MSE: {best_val_loss:.6f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History:    {history_path}")
    print(f"Curve:      {curve_path}")
    print(f"Scatter:    {scatter_path}")


if __name__ == "__main__":
    main()
