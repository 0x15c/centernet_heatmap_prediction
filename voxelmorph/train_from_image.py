import os
import random

import cv2
import numpy as np
import torch

from loss import *
from model import VoxelMorph2D

try:
    import wandb
except ImportError as exc:
    raise ImportError(
        "wandb is required for this script. Install with `pip install wandb`."
    ) from exc


IMAGE_DIR = "../screen_shots"
# e.g. "fixed.png" or None for random fixed images
FIXED_IMAGE_NAME = "frame_0.jpg"
RESIZE_TO = None  # (width, height) or None to keep original size
checkpoint_path = "./ckpt"

batch_size = 32
epochs = 15
learning_rate = 1e-3
smoothness_weight = 0.1
seed = 13
drop_last = True

WANDB_PROJECT = "voxelmorph"
WANDB_RUN_NAME = "train_from_image_MSE_LOSS_RANDOM_FLIP"
HFLIP_PROB = 0.65
ROTATE_CHOICES = (0, 1, 3)  # 0, +90, -90 (k=3)


def load_gray(path, resize_to=None):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if resize_to is not None:
        image = cv2.resize(image, resize_to, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return image


def load_images_in_memory(image_dir, resize_to=None):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [
        f
        for f in sorted(os.listdir(image_dir))
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not files:
        raise ValueError(f"No images found in {image_dir}")

    images = []
    for fname in files:
        path = os.path.join(image_dir, fname)
        img = load_gray(path, resize_to=resize_to)
        images.append((fname, img))

    if resize_to is None:
        h0, w0 = images[0][1].shape
        for name, img in images:
            if img.shape != (h0, w0):
                raise ValueError(
                    "Images have different sizes. Set RESIZE_TO to enforce a size."
                )

    tensors = []
    for name, img in images:
        tensors.append(torch.from_numpy(img)[None, ...])  # (1, H, W)
    return files, tensors


def build_batch(moving_tensors, batch_indices, fixed_tensor=None):
    moving = torch.stack([moving_tensors[i] for i in batch_indices], dim=0)
    if fixed_tensor is not None:
        fixed = fixed_tensor.expand(moving.shape[0], -1, -1, -1)
    else:
        rand_idx = [random.randint(0, len(moving_tensors) - 1)
                    for _ in batch_indices]
        fixed = torch.stack([moving_tensors[i] for i in rand_idx], dim=0)
    return moving, fixed


def _resize_to_square(img, size):
    return F.interpolate(
        img.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def augment_pairs(moving, fixed, square_size=None, hflip_prob=0.5, rotate_choices=(0, 1, 3)):
    batch = moving.shape[0]
    new_moving = []
    new_fixed = []
    if square_size is None:
        # expecting [1, 2, H, W] tensor
        square_size = np.min(moving.size()[2:])
    for i in range(batch):
        m = moving[i]
        f = fixed[i]
        k = random.choice(rotate_choices)
        if k != 0:
            m = torch.rot90(m, k=k, dims=(1, 2))
            f = torch.rot90(f, k=k, dims=(1, 2))
        if random.random() < hflip_prob:
            m = torch.flip(m, dims=(2,))
            f = torch.flip(f, dims=(2,))

        m = _resize_to_square(m, square_size)
        f = _resize_to_square(f, square_size)
        new_moving.append(m)
        new_fixed.append(f)
    return torch.stack(new_moving, dim=0), torch.stack(new_fixed, dim=0)


def main():
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    names, tensors = load_images_in_memory(IMAGE_DIR, resize_to=RESIZE_TO)
    fixed_tensor = None
    if FIXED_IMAGE_NAME is not None:
        try:
            fixed_index = names.index(FIXED_IMAGE_NAME)
        except ValueError as exc:
            raise ValueError(
                f"Fixed image '{FIXED_IMAGE_NAME}' not found in {IMAGE_DIR}"
            ) from exc
        fixed_tensor = tensors[fixed_index]

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "image_dir": IMAGE_DIR,
            "fixed_image_name": FIXED_IMAGE_NAME,
            "resize_to": RESIZE_TO,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "smoothness_weight": smoothness_weight,
            "seed": seed,
        },
    )

    model = VoxelMorph2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    global_step = 0
    num_samples = len(tensors)
    for epoch in range(1, epochs + 1):
        perm = list(range(num_samples))
        random.shuffle(perm)
        if drop_last:
            last = (num_samples // batch_size) * batch_size
            perm = perm[:last]

        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, len(perm), batch_size):
            batch_indices = perm[start: start + batch_size]
            if len(batch_indices) < batch_size:
                continue
            moving, fixed = build_batch(
                tensors, batch_indices, fixed_tensor=fixed_tensor)
            moving = moving.to(device)
            fixed = fixed.to(device)
            moving, fixed = augment_pairs(
                moving, fixed, hflip_prob=HFLIP_PROB, rotate_choices=ROTATE_CHOICES
            )

            warped, flow = model(moving, fixed)
            # loss = total_loss(fixed, warped, flow, smoothness_weight=smoothness_weight)
            sim_loss = similarity_loss(fixed, warped, loss_type="MSE")
            smooth_loss = smoothness_loss(flow)
            loss = sim_loss + smooth_loss*smoothness_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            wandb.log({"train/loss": loss.item(), "train/sim_loss": sim_loss,
                      "train/smooth_loss": smooth_loss, "epoch": epoch}, step=global_step)
            global_step += 1

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_loss,
                  "epoch": epoch}, step=global_step)
        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                       f"{checkpoint_path}/voxelmorph2d_images_{epoch}.pt")
            print(f"Saved checkpoint to {checkpoint_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
