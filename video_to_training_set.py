"""
CenterNet video heatmap visualization (editable config version)

Edit the variables in the CONFIG section, then run:
    python video_heatmap.py
"""

import os
import time
from typing import Tuple
import h5py

from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import torch
import torchvision.transforms as T

from model import CenterNetModel
# from cpd_net.cpd_model import PointRegressor
from cpd_net.pred import displacement_predictor

# ============================================================
# CONFIG — EDIT THESE
# ============================================================

# path to video file, or 0 for webcam
VIDEO_SOURCE = "video/video_with_marker2.mp4"
WEIGHTS_PATH = "checkpoints/latest_model.pth"

INPUT_SIZE = (640, 360)             # model input resolution
HEATMAP_THRESHOLD = 0.2      # set to 0.0 to disable thresholding

OVERLAY_ALPHA = 0.5          # original frame weight
OVERLAY_BETA = 0.5          # heatmap weight

SHOW_FPS = True
MAX_DISPLAY_FPS = 0.0        # 0 = uncapped

SAVE_OUTPUT = True
OUTPUT_VIDEO_PATH = "video_with_marker.mp4"

COLORMAP = cv2.COLORMAP_JET  # OpenCV colormap

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
        frame_rgb = cv2.resize(frame_rgb, INPUT_SIZE, cv2.INTER_NEAREST)

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
    heat = cv2.resize(heat, (w, h), cv2.INTER_NEAREST)

    if HEATMAP_THRESHOLD > 0:
        heat = np.where(heat >= HEATMAP_THRESHOLD, heat, 0.0)

    heat_u8 = np.uint8(np.clip(heat * 255, 0, 255))
    return cv2.applyColorMap(heat_u8, COLORMAP)


def get_heatmap_raw(
    heat: np.ndarray,
        out_shape: Tuple[int, int],
) -> np.ndarray:
    h, w = out_shape
    heat = cv2.resize(heat, (w, h), cv2.INTER_NEAREST)
    return heat

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load centernet weights
    model = build_model_from_weights(WEIGHTS_PATH, device)

    cap = cv2.VideoCapture(0 if VIDEO_SOURCE == 0 else VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {VIDEO_SOURCE}")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read video")

    H, W = frame.shape[:2]

    prev_time = time.time()
    last_show = 0.0

    # image features, conventional CV
    # orb_extractor = cv2.ORB_create(nfeatures=2000, edgeThreshold=0)
    frame_count = 0
    training_file_path = 'training_pointset.h5'
    while True:
        # resize x to INPUT_SIZE tensor, if input_size = None, it will do no resize on input.
        x = preprocess_frame(frame, input_size=INPUT_SIZE)
        # get inference heatmap
        heat_128 = infer_heatmap(model, x, device)
        # find the point of interest
        heat_raw = get_heatmap_raw(heat_128, (INPUT_SIZE[1], INPUT_SIZE[0]))
        # convert into grayscale
        heat_gray = np.uint8(heat_raw*255.0)
        # get cluster centroids
        cv2.imshow("heat_gray",heat_gray)
        cv2.imwrite(f"screen_shots/frame_{frame_count}.jpg",cv2.cvtColor(heat_gray,cv2.COLOR_GRAY2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    cap.release()

if __name__ == "__main__":
    main()
