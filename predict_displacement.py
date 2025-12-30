"""
CenterNet video heatmap visualization (editable config version)

Edit the variables in the CONFIG section, then run:
    python video_heatmap.py
"""

import os
import time
from typing import Tuple

from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import torch
import torchvision.transforms as T

from centernet.centernet_model import CenterNetModel
# from cpd_net.cpd_model import PointRegressor
# from cpd_net.pred import displacement_predictor
from voxelmorph.model import VoxelMorph2D

# ============================================================
# CONFIG — EDIT THESE
# ============================================================

# path to video file, or 0 for webcam
VIDEO_SOURCE = "video/video_with_marker2.mp4"
WEIGHTS_PATH = "checkpoints/latest_model.pth"
WEIGHTS_PATH_VOXELMORPH = "voxelmorph/voxelmorph2d_images_25.pt"
# CPD_WEIGHTS_PATH = 'cpd_net/rect_noise_step_15000.pt'

INPUT_SIZE = (640, 360)             # model input resolution
HEATMAP_THRESHOLD = 0.2      # set to 0.0 to disable thresholding

OVERLAY_ALPHA = 0.5          # original frame weight
OVERLAY_BETA = 0.5          # heatmap weight

SHOW_FPS = True
MAX_DISPLAY_FPS = 0.0        # 0 = uncapped

SAVE_OUTPUT = True
OUTPUT_VIDEO_PATH = "video_with_marker.mp4"

COLORMAP = cv2.COLORMAP_JET  # OpenCV colormap

# ============================================================


def get_centernet_model(weights_path: str, device: torch.device) -> CenterNetModel:
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

def get_voxelmorph_model(weights_path: str, device: torch.device) -> VoxelMorph2D:
    state = torch.load(weights_path, map_location=device)
    model = VoxelMorph2D()
    model.load_state_dict(state)
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
def centernet_infer(
    model: CenterNetModel,
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    logits = model(x.to(device))
    prob = torch.sigmoid(logits)[0]  # CxHxW

    if prob.shape[0] > 1:
        heat = prob.max(dim=0).values
    else:
        heat = prob[0]

    return heat

@torch.no_grad()
def voxelmorph_infer(
    model: VoxelMorph2D,
    moving: torch.tensor,
    fixed: torch.tensor,
    # device: torch.device,
)-> np.ndarray:
    wraped, flow = model(moving, fixed)
    wraped_np = wraped.squeeze().cpu().numpy()
    cv2.imshow("wraped",np.uint8(wraped_np*255))
    return flow.squeeze().cpu().numpy()


def render_heatmap(
    heat: np.ndarray,
    out_shape: Tuple[int, int],
) -> np.ndarray:
    if out_shape != (None, None):
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


def draw_keypoints(img_color, keypoints, color=(0, 0, 255)):
    for kp in keypoints:
        x, y = int(round(kp[0])), int(round(kp[1]))
        cv2.circle(img_color, (x, y), radius=3, color=color, thickness=1)
    return img_color


def dbscan_extractor(dbscan_result, points):
    labels = dbscan_result.labels_
    points = np.array(points)

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    if len(unique_labels) == 0:
        return []

    cluster_info = []
    for cluster_id in unique_labels:
        cluster_mask = (labels == cluster_id)
        cluster_points = points[cluster_mask]
        cluster_info.append(cluster_points)

    return cluster_info


def centroids_calc(cluster_array):
    result = np.zeros((0, 2))
    intsty = np.zeros((0)).astype(np.uint16)
    for cluster in cluster_array:
        centroid = np.mean(cluster, axis=0)
        n_pts = cluster.shape[0]
        intensity = n_pts
        result = np.append(result, [centroid], axis=0)
        intsty = np.append(intsty, [intensity], axis=0)
    return result, intsty


def get_pointset(heatmap_uint8: np.ndarray):
    thres = 50
    eps = 3
    min_samples = 2
    _, activate = cv2.threshold(heatmap_uint8, thres, 255, cv2.THRESH_BINARY)
    # find those activated points above certain threshlod
    pts = cv2.findNonZero(activate)
    if pts is None:
        centroids = np.zeros((0, 2))
    else:
        cluster_coordinates = pts.reshape(-1, 2)
        cluster_data = DBSCAN(eps=eps, min_samples=min_samples).fit(
            cluster_coordinates)
        clusters = dbscan_extractor(cluster_data, cluster_coordinates)
        centroids, _ = centroids_calc(clusters)
    return centroids

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
        cv2.arrowedLine(img, start, end, color,
                        thickness, tipLength=tip_length)

    return img

def sample_flow_at_points(flow: np.ndarray, c: np.ndarray, radius: int = 3) -> np.ndarray:
    # flow: (2, H, W), c: (N, 2) in (x, y)
    h, w = flow.shape[1], flow.shape[2]
    pts = np.asarray(c, dtype=np.float32)

    xs = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, h - 1)

    if radius <= 0:
        u = flow[0, ys, xs]
        v = flow[1, ys, xs]
    else:
        ksize = 2 * radius + 1
        u_blur = cv2.blur(flow[0], (ksize, ksize), borderType=cv2.BORDER_REFLECT)
        v_blur = cv2.blur(flow[1], (ksize, ksize), borderType=cv2.BORDER_REFLECT)
        u = u_blur[ys, xs]
        v = v_blur[ys, xs]

    return np.stack([u, v], axis=1)

def sample_regular_grid(height: int, width: int, step: int) -> np.ndarray:
    ys = np.arange(0, height, step, dtype=np.int32)
    xs = np.arange(0, width, step, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    base_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return base_points, grid_x, grid_y



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load cpd_net weights for displacement field prediction
    # cpd_net_predictor = displacement_predictor(CPD_WEIGHTS_PATH, device)
    # load centernet weights
    centernet_model = get_centernet_model(WEIGHTS_PATH, device)
    voxelmorph_model = get_voxelmorph_model(WEIGHTS_PATH_VOXELMORPH, device)

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
            INPUT_SIZE,
        )

    prev_time = time.time()
    last_show = 0.0

    # image features, conventional CV
    # orb_extractor = cv2.ORB_create(nfeatures=2000, edgeThreshold=0)
    frame_count = 0
    # grid sampling flow
    height, width = INPUT_SIZE[1], INPUT_SIZE[0]
    step = max(1, min(height, width) // 12)
    base_points, grid_x, grid_y = sample_regular_grid(height, width, step)

    while True:
        # resize x to INPUT_SIZE tensor, if input_size = None, it will do no resize on input.
        x = preprocess_frame(frame, input_size=INPUT_SIZE) # INPUT_SIZE
        frame_downsampled = cv2.resize(frame, INPUT_SIZE, cv2.INTER_NEAREST)
        # get inference probability map
        # please be noted that the outputed probability map will be downsampled by 4x
        # that's why we have resize everywhere
        probmap_inferred = centernet_infer(centernet_model, x, device)
        # find the point of interest
        heat_raw = get_heatmap_raw(probmap_inferred.cpu().numpy(), (INPUT_SIZE[1], INPUT_SIZE[0]))
        # convert into grayscale
        heat_gray = np.uint8(heat_raw*255.0)
        # heat_clahe = clahe.apply(heat_gray)
        # orb_keypoints = orb_extractor.detect(heat_gray, None)
        # frame_show = draw_keypoints(cv2.cvtColor(
        #     heat_gray, cv2.COLOR_GRAY2BGR), orb_keypoints)
        # cv2.imshow("grayscale_heatmap", frame_show)
        # model outputs the resized tensor compared to input, thus the output should be resized

        # get cluster centroids
        c = get_pointset(heat_gray)
        if frame_count <= 0:
            c0 = c
            d = None
            frame0_tensor = probmap_inferred
        # if frame_count % 10 == 0:
        #     cv2.imwrite(f"screen_shots/{frame_count//10}.png",cv2.cvtColor(heat_gray,cv2.COLOR_GRAY2BGR))
        flow = voxelmorph_infer(voxelmorph_model, probmap_inferred[None, None], frame0_tensor[None, None])
        # after we obtain the flow, let's do some upsampling to match the dimension:
        h, w = flow.shape[1], flow.shape[2]
        scale_x = INPUT_SIZE[0] / w
        scale_y = INPUT_SIZE[1] / h

        u = cv2.resize(flow[0], (INPUT_SIZE[0], INPUT_SIZE[1]), interpolation=cv2.INTER_LINEAR) * scale_x
        v = cv2.resize(flow[1], (INPUT_SIZE[0], INPUT_SIZE[1]), interpolation=cv2.INTER_LINEAR) * scale_y
        flow_upsampled = np.stack([u, v], axis=0)
        d = sample_flow_at_points(flow_upsampled, c0, radius=2)
        
        heat_color = render_heatmap(probmap_inferred.cpu().numpy(), (INPUT_SIZE[1], INPUT_SIZE[0]))
        overlay = cv2.addWeighted(
            frame_downsampled, OVERLAY_ALPHA, heat_color, OVERLAY_BETA, 0)
        overlay = draw_keypoints(overlay, c)
        overlay = draw_keypoints(overlay, c0, color=(0, 255, 0))
        if d is not None:
            overlay = draw_displacement_vectors(overlay, c0, d*5)
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
                cv2.imshow("Overlay", overlay)
                last_show = time.time()
        else:
            # cv2.imshow("Frame", frame)
            cv2.imshow("Heatmap", heat_color)
            cv2.imshow("Overlay", overlay)

        if writer is not None:
            writer.write(overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
