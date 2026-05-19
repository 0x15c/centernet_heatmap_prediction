"""Train a QCRegNet-style tactile registration model.

This script deliberately does not use the existing VoxelMorph path.  It follows:

    fixed/moving image pair -> Estimator -> mu -> BSNet -> mapping -> warp moving to fixed

The default training objective is self-supervised image similarity, with small
regularizers on raw mu and mu Laplacian.  A pretrained BSNet checkpoint is
recommended.  If --freeze-bsnet is set, only the Estimator is trained.

Example:
    python train_qcreg_tactile.py \
        --data-root data/registration_pairs \
        --image-size 224 224 \
        --bsnet-checkpoint bsnet4000.pth \
        --freeze-bsnet \
        --epochs 100 \
        --batch-size 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from qc_regnet_tactile.dataset import TactilePairDataset
from qc_regnet_tactile.losses import (
    image_similarity_loss,
    laplacian_loss,
    mapping_det_jacobian_loss,
    mu_magnitude_loss,
    ncc_loss,
    warp_with_mapping,
)
from qc_regnet_tactile.models import BSNet, Estimator, mapping_to_flow_pixels


def load_state_dict_flexible(module: nn.Module, path: str | Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    module.load_state_dict(state)


def save_debug_images(out_dir: Path, fixed: torch.Tensor, moving: torch.Tensor, warped: torch.Tensor, flow: torch.Tensor, step: int) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_np = fixed[0, 0].detach().cpu().numpy()
    moving_np = moving[0, 0].detach().cpu().numpy()
    warped_np = warped[0, 0].detach().cpu().numpy()
    flow_np = flow[0].detach().cpu()
    mag = torch.linalg.norm(flow_np, dim=0).numpy()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(fixed_np, cmap="gray")
    ax.set_title("fixed")
    ax.axis("off")
    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(moving_np, cmap="gray")
    ax.set_title("moving")
    ax.axis("off")
    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(warped_np, cmap="gray")
    ax.set_title("warped moving")
    ax.axis("off")
    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(mag)
    ax.set_title("flow magnitude / px")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"debug_{step:06d}.png", dpi=150)
    plt.close(fig)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    image_size = tuple(args.image_size)
    if image_size[0] % 16 != 0 or image_size[1] % 16 != 0:
        raise ValueError("--image-size height and width should be divisible by 16 for this prototype")

    dataset = TactilePairDataset(args.data_root, size=image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    estimator = Estimator(n_channels=2, base_channels=args.base_channels).to(device)
    bsnet = BSNet((image_size[0] // 2, image_size[1] // 2)).to(device)

    if args.bsnet_checkpoint:
        load_state_dict_flexible(bsnet, args.bsnet_checkpoint, device)
        print(f"Loaded BSNet checkpoint: {args.bsnet_checkpoint}")
    else:
        print("WARNING: no --bsnet-checkpoint provided. BSNet will train from scratch, which may be unstable.")

    if args.estimator_checkpoint:
        load_state_dict_flexible(estimator, args.estimator_checkpoint, device)
        print(f"Loaded Estimator checkpoint: {args.estimator_checkpoint}")

    if args.freeze_bsnet:
        bsnet.eval()
        for p in bsnet.parameters():
            p.requires_grad_(False)

    params = list(estimator.parameters()) + ([] if args.freeze_bsnet else list(bsnet.parameters()))
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        estimator.train()
        if args.freeze_bsnet:
            bsnet.eval()
        else:
            bsnet.train()

        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            fixed = batch["fixed"].to(device, non_blocking=True)
            moving = batch["moving"].to(device, non_blocking=True)
            pair = torch.cat([fixed, moving], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                mu, raw_mu = estimator(pair)
                mapping_half = bsnet(mu)
                mapping = F.interpolate(mapping_half, size=image_size, mode="bilinear", align_corners=True)
                warped = warp_with_mapping(moving, mapping)

                if args.similarity == "ncc":
                    loss_img = ncc_loss(warped, fixed)
                else:
                    loss_img = image_similarity_loss(warped, fixed, mode=args.similarity)

                loss_mu = mu_magnitude_loss(raw_mu)
                loss_lap = laplacian_loss(mu)
                loss_det = mapping_det_jacobian_loss(mapping)

                loss = (
                    args.lambda_img * loss_img
                    + args.lambda_mu * loss_mu
                    + args.lambda_lap * loss_lap
                    + args.lambda_det * loss_det
                )

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            global_step += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                img=f"{loss_img.item():.4f}",
                mu=f"{loss_mu.item():.4f}",
                lap=f"{loss_lap.item():.4f}",
                det=f"{loss_det.item():.4f}",
            )

            if args.debug_every > 0 and global_step % args.debug_every == 0:
                flow = mapping_to_flow_pixels(mapping)
                save_debug_images(out_dir / "debug", fixed, moving, warped, flow, global_step)

        avg_loss = running / max(len(loader), 1)
        print(f"Epoch {epoch + 1}: avg loss = {avg_loss:.6f}")

        if (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "estimator": estimator.state_dict(),
                    "bsnet": bsnet.state_dict(),
                    "args": vars(args),
                },
                ckpt_dir / f"qcreg_tactile_epoch_{epoch + 1:04d}.pth",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QCRegNet-style tactile registration")
    parser.add_argument("--data-root", required=True, help="Folder containing fixed/moving pairs or seq/ref + frames")
    parser.add_argument("--image-size", nargs=2, type=int, default=[224, 224], metavar=("H", "W"))
    parser.add_argument("--output-dir", default="runs/qcreg_tactile")
    parser.add_argument("--bsnet-checkpoint", default=None, help="Optional pretrained BSNet .pth")
    parser.add_argument("--estimator-checkpoint", default=None, help="Optional Estimator checkpoint")
    parser.add_argument("--freeze-bsnet", action="store_true", help="Train Estimator only; recommended with pretrained BSNet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--similarity", choices=["charbonnier", "mse", "l1", "ncc"], default="charbonnier")
    parser.add_argument("--lambda-img", type=float, default=1.0)
    parser.add_argument("--lambda-mu", type=float, default=1e-3)
    parser.add_argument("--lambda-lap", type=float, default=1e-3)
    parser.add_argument("--lambda-det", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--debug-every", type=int, default=200)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
