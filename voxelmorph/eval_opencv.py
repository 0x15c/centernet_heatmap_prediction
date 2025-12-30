import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from centernet.centernet_model import VoxelMorph2D

MOVING_PATH = "screen_shots/65.png"
FIXED_PATH = "screen_shots/0.png"
CHECKPOINT_PATH = "voxelmorph2d_mnist.pt"
OUT_PATH = "warped.png"
FLOW_PATH = "flow.npy"
RESIZE_FIXED = False
DEVICE = None  # "cpu", "cuda", or None for auto


def load_gray(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return image


def draw_displacement_vectors(
    image: np.ndarray,
    base_points: np.ndarray,
    displacement: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    tip_length: float = 0.2,
    copy: bool = True,
) -> np.ndarray:
    """
    Overlay displacement vectors on an image.

    Args:
        image: HxWx3 (BGR) image as numpy array.
        base_points: (N, 2) array of base points (x, y) in pixel coords.
        displacement: (N, 2) array of displacement vectors (dx, dy) in pixels.
        color: Arrow color in BGR.
        thickness: Line thickness for arrows.
        tip_length: Arrow tip length (OpenCV parameter, 0-1).
        copy: If True, draw on a copy of the image.

    Returns:
        Image with vector overlays.
    """
    if copy:
        img = image.copy()
    else:
        img = image

    base_points = np.asarray(base_points, dtype=np.float32)
    displacement = np.asarray(displacement, dtype=np.float32)

    if base_points.shape != displacement.shape or base_points.shape[1] != 2:
        raise ValueError("base_points and displacement must be shape (N, 2).")

    for (x, y), (dx, dy) in zip(base_points, displacement):
        start = (int(round(x)), int(round(y)))
        end = (int(round(x + dx)), int(round(y + dy)))
        cv2.arrowedLine(img, start, end, color, thickness, tipLength=tip_length)

    return img


def sample_regular_grid(height: int, width: int, step: int) -> np.ndarray:
    ys = np.arange(0, height, step, dtype=np.int32)
    xs = np.arange(0, width, step, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    base_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return base_points, grid_x, grid_y


def main():
    moving = load_gray(MOVING_PATH)
    fixed = load_gray(FIXED_PATH)

    if moving.shape != fixed.shape:
        if not RESIZE_FIXED:
            raise ValueError(
                "Input sizes differ. Set RESIZE_FIXED=True or pre-resize images to match."
            )
        fixed = cv2.resize(fixed, (moving.shape[1], moving.shape[0]), interpolation=cv2.INTER_LINEAR)

    device = DEVICE
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = VoxelMorph2D()
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    moving_t = torch.from_numpy(moving)[None, None].to(device)
    fixed_t = torch.from_numpy(fixed)[None, None].to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        warped, flow = model(moving_t, fixed_t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    warped_np = warped.squeeze().cpu().numpy()
    warped_np = np.clip(warped_np, 0.0, 1.0)
    flow_np = flow.squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].imshow(moving, cmap="gray")
    axes[0, 0].set_title("Moving")
    axes[0, 1].imshow(fixed, cmap="gray")
    axes[0, 1].set_title("Fixed")
    axes[1, 0].imshow(warped_np, cmap="gray")
    axes[1, 0].set_title("Warped")
    axes[1, 1].set_title("Flow (OpenCV)")

    height, width = flow_np.shape[1], flow_np.shape[2]
    step = max(1, min(height, width) // 20)
    base_points, grid_x, grid_y = sample_regular_grid(height, width, step)
    displacement = np.stack(
        (flow_np[0, grid_y, grid_x], flow_np[1, grid_y, grid_x]), axis=-1
    ).reshape(-1, 2)
    flow_bg = cv2.cvtColor((fixed * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    flow_vis = draw_displacement_vectors(flow_bg, base_points, displacement*5)
    axes[1, 1].imshow(cv2.cvtColor(flow_vis, cv2.COLOR_BGR2RGB))

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(f"Registration time: {elapsed:.4f} seconds")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
