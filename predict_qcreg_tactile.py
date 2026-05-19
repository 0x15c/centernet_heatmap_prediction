"""Run QCRegNet-style tactile registration on one image pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from qc_regnet_tactile.dataset import read_gray_tensor
from qc_regnet_tactile.losses import warp_with_mapping
from qc_regnet_tactile.models import BSNet, Estimator, mapping_to_flow_pixels


def load_checkpoint(path: str | Path, estimator: Estimator, bsnet: BSNet, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    if "estimator" in state:
        estimator.load_state_dict(state["estimator"])
    else:
        estimator.load_state_dict(state)
    if "bsnet" in state:
        bsnet.load_state_dict(state["bsnet"])


def flow_to_hsv(flow: torch.Tensor) -> torch.Tensor:
    """flow: 2xHxW tensor in pixels. Return HxWx3 uint8 BGR for OpenCV."""
    flow_np = flow.detach().cpu().permute(1, 2, 0).numpy()
    fx = flow_np[..., 0]
    fy = flow_np[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    hsv = cv2.merge([
        (ang / 2).astype("uint8"),
        (255 * mag / (mag.max() + 1e-6)).astype("uint8"),
        (255 * torch.ones_like(torch.from_numpy(mag)).numpy()).astype("uint8"),
    ])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    image_size = tuple(args.image_size)

    fixed = read_gray_tensor(Path(args.fixed), image_size).unsqueeze(0).to(device)
    moving = read_gray_tensor(Path(args.moving), image_size).unsqueeze(0).to(device)

    estimator = Estimator(n_channels=2, base_channels=args.base_channels).to(device)
    bsnet = BSNet((image_size[0] // 2, image_size[1] // 2)).to(device)
    load_checkpoint(args.checkpoint, estimator, bsnet, device)
    estimator.eval()
    bsnet.eval()

    with torch.no_grad():
        pair = torch.cat([fixed, moving], dim=1)
        mu, _ = estimator(pair)
        mapping_half = bsnet(mu)
        mapping = F.interpolate(mapping_half, size=image_size, mode="bilinear", align_corners=True)
        warped = warp_with_mapping(moving, mapping)
        flow = mapping_to_flow_pixels(mapping)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"mapping": mapping.cpu(), "flow_pixels": flow.cpu(), "mu": mu.cpu()}, out_dir / "qcreg_result.pt")

    fixed_np = fixed[0, 0].cpu().numpy()
    moving_np = moving[0, 0].cpu().numpy()
    warped_np = warped[0, 0].cpu().numpy()
    flow_rgb = flow_to_hsv(flow[0])

    fig = plt.figure(figsize=(10, 8))
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
    ax.imshow(flow_rgb)
    ax.set_title("flow direction/magnitude")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "qcreg_prediction.png", dpi=160)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with QCRegNet tactile registration")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--image-size", nargs=2, type=int, default=[224, 224], metavar=("H", "W"))
    parser.add_argument("--output-dir", default="runs/qcreg_predict")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
