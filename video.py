"""
CenterNet video heatmap visualization (editable config version)

Edit the variables in the CONFIG section, then run:
    python video_heatmap.py
"""

import os
import time
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from model import CenterNetModel


# ============================================================
# CONFIG — EDIT THESE
# ============================================================

VIDEO_SOURCE = "video/1.mp4"   # path to video file, or 0 for webcam
WEIGHTS_PATH = "checkpoints/centernet_resnet9_e90.pth"

INPUT_SIZE = (640,480)             # model input resolution
HEATMAP_THRESHOLD = 0.2      # set to 0.0 to disable thresholding

OVERLAY_ALPHA = 0.6          # original frame weight
OVERLAY_BETA = 0.4           # heatmap weight

SHOW_FPS = True
MAX_DISPLAY_FPS = 0.0        # 0 = uncapped

SAVE_OUTPUT = True
OUTPUT_VIDEO_PATH = "video_with_marker.mp4"

COLORMAP = cv2.COLORMAP_JET  # OpenCV colormap

# ============================================================


def build_model_from_weights(weights_path: str, device: torch.device) -> CenterNetModel:
    state = torch.load(weights_path, map_location=device)

    if "head.2.bias" in state:
        num_classes = state["head.2.bias"].shape[0]
        model = CenterNetModel(num_classes=num_classes)
        model.load_state_dict(state, strict=True)
    else:
        model = state
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError("Unsupported checkpoint format")

    model.to(device)
    model.eval()
    return model


def preprocess_frame(
    frame_bgr: np.ndarray,
    input_size: int,
) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if input_size is not None:
        frame_rgb = cv2.resize(frame_rgb, INPUT_SIZE)

    x = T.functional.to_tensor(frame_rgb)
    x = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(x)

    return x.unsqueeze(0)


@torch.no_grad()
def infer_heatmap(
    model: CenterNetModel,
    x: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    logits = model(x.to(device))
    prob = torch.sigmoid(logits)[0]  # CxHxW

    if prob.shape[0] > 1:
        heat = prob.max(dim=0).values
    else:
        heat = prob[0]

    return heat.cpu().numpy()


def render_heatmap(
    heat: np.ndarray,
    out_shape: Tuple[int, int],
) -> np.ndarray:
    h, w = out_shape
    heat = cv2.resize(heat, (w, h))

    if HEATMAP_THRESHOLD > 0:
        heat = np.where(heat >= HEATMAP_THRESHOLD, heat, 0.0)

    heat_u8 = np.uint8(np.clip(heat * 255, 0, 255))
    return cv2.applyColorMap(heat_u8, COLORMAP)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_model_from_weights(WEIGHTS_PATH, device)

    cap = cv2.VideoCapture(0 if VIDEO_SOURCE == 0 else VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {VIDEO_SOURCE}")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read video")

    H, W = frame.shape[:2]

    writer = None
    if SAVE_OUTPUT:
        os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH) or ".", exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1 or np.isnan(fps):
            fps = 30.0
        writer = cv2.VideoWriter(
            OUTPUT_VIDEO_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
        )

    prev_time = time.time()
    last_show = 0.0

    while True:
        x = preprocess_frame(frame, INPUT_SIZE)
        heat_128 = infer_heatmap(model, x, device)

        heat_color = render_heatmap(heat_128, (H, W))
        overlay = cv2.addWeighted(frame, OVERLAY_ALPHA, heat_color, OVERLAY_BETA, 0)

        if SHOW_FPS:
            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now
            cv2.putText(
                overlay,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

        if MAX_DISPLAY_FPS > 0:
            if time.time() - last_show >= 1.0 / MAX_DISPLAY_FPS:
                # cv2.imshow("Frame", frame)
                cv2.imshow("Heatmap", heat_color)
                # cv2.imshow("Overlay", overlay)
                last_show = time.time()
        else:
            # cv2.imshow("Frame", frame)
            cv2.imshow("Heatmap", heat_color)
            # cv2.imshow("Overlay", overlay)

        if writer is not None:
            writer.write(overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        ret, frame = cap.read()
        if not ret:
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
