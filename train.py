# train.py
from __future__ import annotations
import argparse
import os
import torch
from torch.utils.data import DataLoader
import wandb

from model import CenterNetModel
from dataloader import CenterNetDataset
from losses import combined_heatmap_loss


def unpack_batch(batch):
    """
    Handles dataset returning (img, hm) or (img, hm, meta) or more.
    """
    if isinstance(batch, (list, tuple)):
        images = batch[0]
        heatmaps = batch[1]
        meta = batch[2] if len(batch) > 2 else None
        return images, heatmaps, meta
    raise ValueError("Unexpected batch type")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--format", choices=["yolo", "coco"], required=True)
    ap.add_argument("--val_images", default=None)
    ap.add_argument("--val_annotations", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lambda_sparse", type=float, default=0.05)
    ap.add_argument("--lambda_tv", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--project", type=str, default="centernet-resnet9")
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CenterNetDataset(args.images, args.annotations, fmt=args.format, augment=True)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_dl = None
    if args.val_images and args.val_annotations:
        val_ds = CenterNetDataset(args.val_images, args.val_annotations, fmt=args.format, augment=False)
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    model = CenterNetModel(num_classes=train_ds.num_classes).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    wandb.init(
        project=args.project,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_classes": train_ds.num_classes,
            "lambda_sparse": args.lambda_sparse,
            "lambda_tv": args.lambda_tv,
        },
    )
    wandb.watch(model, log="gradients", log_freq=200)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for batch in train_dl:
            images, heatmaps, _ = unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(images)

            loss, logs = combined_heatmap_loss(
                logits, heatmaps,
                lambda_sparse=args.lambda_sparse,
                lambda_tv=args.lambda_tv,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            running += loss.item()
            global_step += 1

            if global_step % 20 == 0:
                wandb.log({"train/lr": opt.param_groups[0]["lr"], "step": global_step, "epoch": epoch, **{f"train/{k}": v for k, v in logs.items()}})

            # log a visualization sometimes
            if global_step % 200 == 0:
                with torch.no_grad():
                    pred = torch.sigmoid(logits[0]).detach().cpu()   # (C,128,128)
                    gt = heatmaps[0].detach().cpu()
                    pred_max = pred.max(dim=0).values
                    gt_max = gt.max(dim=0).values
                    wandb.log({
                        "viz/pred_heatmap": wandb.Image(pred_max.numpy()),
                        "viz/gt_heatmap": wandb.Image(gt_max.numpy()),
                        "step": global_step,
                        "epoch": epoch
                    })

        train_loss_epoch = running / max(1, len(train_dl))
        wandb.log({"train/epoch_loss": train_loss_epoch, "epoch": epoch})

        # ---- validation ----
        if val_dl is not None:
            model.eval()
            val_running = 0.0
            val_logs_accum = {"loss_focal": 0.0, "loss_sparse": 0.0, "loss_tv": 0.0, "loss_total": 0.0}
            n_batches = 0

            with torch.no_grad():
                for batch in val_dl:
                    images, heatmaps, _ = unpack_batch(batch)
                    images = images.to(device, non_blocking=True)
                    heatmaps = heatmaps.to(device, non_blocking=True)

                    logits = model(images)
                    loss, logs = combined_heatmap_loss(
                        logits, heatmaps,
                        lambda_sparse=args.lambda_sparse,
                        lambda_tv=args.lambda_tv,
                    )

                    val_running += loss.item()
                    for k in val_logs_accum:
                        val_logs_accum[k] += logs[k]
                    n_batches += 1

            val_loss_epoch = val_running / max(1, n_batches)
            for k in val_logs_accum:
                val_logs_accum[k] /= max(1, n_batches)

            wandb.log({
                "val/epoch_loss": val_loss_epoch,
                **{f"val/{k}": v for k, v in val_logs_accum.items()},
                "epoch": epoch
            })

        sched.step()

        # save checkpoint
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt = os.path.join(args.save_dir, f"centernet_resnet9_e{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            wandb.save(ckpt)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss_epoch:.4f}" + (f"  val_loss={val_loss_epoch:.4f}" if val_dl else ""))


if __name__ == "__main__":
    main()
