"""
CenterNet video heatmap visualization (editable config version)

Edit the variables in the CONFIG section, then run:
    python video_heatmap.py
"""

import os
import time
from typing import Tuple
import json

from sklearn.cluster import DBSCAN, KMeans

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import gaussian_blur
# import skimage

from centernet.centernet_model import CenterNetModel
import matplotlib.pyplot as plt 

# ============================================================
# CONFIG — EDIT THESE
# ============================================================

# path to video file, or 0 for webcam
# "force_regression_test/Raw_Session_20260205_234104.avi"  # "video/eval3.mp4"
VIDEO_SOURCE = "force_regression_test/Raw_Session_20260311_223951.avi"
WEIGHTS_PATH = "centernet/checkpoints/centernet_resnet9_e35.pth"  # centernet
WEIGHTS_PATH_VOXELMORPH = "voxelmorph/ckpt/voxelmorph2d_images_20_new_sensor.pt"
# CPD_WEIGHTS_PATH = 'cpd_net/rect_noise_step_15000.pt'

INPUT_SIZE = (600, 460)      # model input resolution, (W, H)
CONCAT_SIZE = (INPUT_SIZE[0]*2, INPUT_SIZE[1]*2)
HEATMAP_THRESHOLD = 0.2      # set to 0.0 to disable thresholding

OVERLAY_ALPHA = 0.5          # original frame weight
OVERLAY_BETA = 0.5          # heatmap weight

SHOW_FPS = True
MAX_DISPLAY_FPS = 0.0        # 0 = uncapped

SAVE_OUTPUT = True
OUTPUT_VIDEO_PATH = "Session_20260311_223951_simple_search.mp4"
DISPLACEMENT_OUTPUT_JSON_PATH = "mlp_force_prediction/Session_20260311_223951_MLP_simple_search.jsonl"

COLORMAP = cv2.COLORMAP_JET  # OpenCV colormap

DISP_AMP_COEFF = 1




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
    thres = 30
    eps = 3
    min_samples = 5
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

def get_pointset_kmeans(n_clusters, heatmap_uint8: np.ndarray):
    thres = 30
    eps = 3
    min_samples = 5
    _, activate = cv2.threshold(heatmap_uint8, thres, 255, cv2.THRESH_BINARY)
    # find those activated points above certain threshlod
    pts = cv2.findNonZero(activate)
    if pts is None:
        centroids = np.zeros((0, 2))
    else:
        cluster_coordinates = pts.reshape(-1, 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(cluster_coordinates)
    return kmeans.cluster_centers_


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


# flow: [2, H, W]
def blur_flow_field(flow: np.ndarray, ksize=3) -> np.ndarray:
    u = cv2.blur(flow[0], (ksize, ksize))
    v = cv2.blur(flow[1], (ksize, ksize))
    return np.stack([u, v], axis=0)


def sample_flow_at_points(flow: np.ndarray, c: np.ndarray, radius: int = 5) -> np.ndarray:
    # flow: (2, H, W), c: (N, 2) in (x, y)
    h, w = flow.shape[1], flow.shape[2]
    pts = np.asarray(c, dtype=np.float32)

    xs = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, h - 1)

    if radius <= 0:
        u = flow[0, ys, xs]
        v = flow[1, ys, xs]
    else:
        # sampling from adjecent volume, here using Gaussian blur
        ksize = 2 * radius + 1
        u_blur = cv2.blur(flow[0], (ksize, ksize),
                          borderType=cv2.BORDER_REFLECT)
        v_blur = cv2.blur(flow[1], (ksize, ksize),
                          borderType=cv2.BORDER_REFLECT)
        u = u_blur[ys, xs]
        v = v_blur[ys, xs]

    return np.stack([u, v], axis=1)


def sample_regular_grid(height: int, width: int, step: int) -> np.ndarray:
    ys = np.arange(0, height, step, dtype=np.int32)
    xs = np.arange(0, width, step, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    base_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return base_points, grid_x, grid_y

def match_displacements_silly(c0, c, max_dist=100.0, return_indices=False, return_distances=False):
    """
    Exhaustive independent nearest-neighbor matching from c0 to c.

    For each point c0[j], search all points in c independently.
    Previously matched points in c are STILL allowed to be reused.

    Parameters
    ----------
    c0 : (N, D) array
        Source points.
    c : (M, D) array
        Target points.
    max_dist : float or None
        If not None, reject matches with distance > max_dist.
        Rejected points get zero displacement and index -1.
    return_indices : bool
        If True, also return matched indices into c.
    return_distances : bool
        If True, also return nearest distances.

    Returns
    -------
    disp : (N, D) ndarray
        Displacement vectors.
    matched_idx : (N,) ndarray, optional
        Matched index in c for each c0 point, or -1 if unmatched.
    dists : (N,) ndarray, optional
        Nearest distance for each c0 point (np.inf if unmatched).
    """
    c0 = np.asarray(c0, dtype=float)
    c = np.asarray(c, dtype=float)

    if c0.ndim != 2 or c.ndim != 2:
        raise ValueError("c0 and c must both be 2D arrays")
    if c0.shape[1] != c.shape[1]:
        raise ValueError("c0 and c must have the same point dimension")

    N, D = c0.shape
    M = c.shape[0]

    disp = np.zeros((N, D), dtype=float)
    matched_idx = -np.ones(N, dtype=int)
    dists = np.full(N, np.inf, dtype=float)

    if N == 0 or M == 0:
        outputs = [disp]
        if return_indices:
            outputs.append(matched_idx)
        if return_distances:
            outputs.append(dists)
        return tuple(outputs) if len(outputs) > 1 else disp

    for j in range(N):
        best_i = -1
        best_d2 = np.inf

        for i in range(M):
            diff = c[i] - c0[j]
            d2 = np.dot(diff, diff)
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        if best_i >= 0:
            best_d = np.sqrt(best_d2)
            if (max_dist is None) or (best_d <= max_dist):
                matched_idx[j] = best_i
                dists[j] = best_d
                disp[j] = c[best_i] - c0[j]

    outputs = [disp]
    if return_indices:
        outputs.append(matched_idx)
    if return_distances:
        outputs.append(dists)

    return tuple(outputs) if len(outputs) > 1 else disp

def clamp_displacements_to_previous(disp_new, disp_prev, max_change):
    """
    Replace extreme displacement updates by previous values.

    Parameters
    ----------
    disp_new : (N, D) array
        Newly computed displacement vectors.
    disp_prev : (N, D) array or None
        Previous displacement vectors.
        If None, disp_new is returned unchanged.
    max_change : float
        Maximum allowed change in displacement magnitude between frames.

    Returns
    -------
    disp_out : (N, D) ndarray
        Filtered displacement vectors.
    outlier_mask : (N,) ndarray of bool
        True where the new vector was replaced by the previous one.
    """
    disp_new = np.asarray(disp_new, dtype=float)

    if disp_prev is None:
        return disp_new.copy(), np.zeros(disp_new.shape[0], dtype=bool)

    disp_prev = np.asarray(disp_prev, dtype=float)

    if disp_new.shape != disp_prev.shape:
        raise ValueError("disp_new and disp_prev must have the same shape")

    diff = disp_new - disp_prev
    diff_norm = np.linalg.norm(diff, axis=1)

    outlier_mask = diff_norm > max_change

    disp_out = disp_new.copy()
    disp_out[outlier_mask] = disp_prev[outlier_mask]

    return disp_out, outlier_mask

def match_and_simple_filter(c0, c, disp_prev=None, max_dist=None, max_change=5.0,
                            return_indices=False, return_mask=False):
    """
    1. Exhaustive independent nearest-neighbor matching
    2. If a displacement changes too much from previous value, keep previous value
    """
    disp_new, idx = match_displacements_silly(c0, c, max_dist=max_dist, return_indices=True)
    disp_out, mask = clamp_displacements_to_previous(disp_new, disp_prev, max_change=max_change)

    outputs = [disp_out]
    if return_indices:
        outputs.append(idx)
    if return_mask:
        outputs.append(mask)

    return tuple(outputs) if len(outputs) > 1 else disp_out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load cpd_net weights for displacement field prediction
    # cpd_net_predictor = displacement_predictor(CPD_WEIGHTS_PATH, device)
    # load centernet weights
    centernet_model = get_centernet_model(WEIGHTS_PATH, device)

    cap = cv2.VideoCapture(0 if VIDEO_SOURCE == 0 else VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {VIDEO_SOURCE}")

    ret, frame = cap.read()  # frame: (H, W, C)
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
            CONCAT_SIZE,
        )

    prev_time = time.time()
    frame_count = 0
    # grid sampling flow
    height, width = INPUT_SIZE[1], INPUT_SIZE[0]

    # this is something like a buffer
    # this is because the centernet infer model returns a tensor 1/4 of its original size.
    probmap_inferred_cpu_tensor = torch.empty(
        (height//4, width//4), pin_memory=True)
    flow_cpu_tensor = torch.empty(
        (2, height//4, width//4), pin_memory=True)  # [2, H, W] Tensor
    # matplotlib settings, grid data sampling
    # X,Y = np.meshgrid(np.linspace(0,height//4,height//4),np.linspace(0,width//4,width//4),indexing="ij")
    
    # plt.ion()
    # fig, ax = plt.subplots()
    # q = ax.quiver(X,Y,X,Y)
    

    while True:
        # resize x to INPUT_SIZE tensor, if input_size = None, it will do no resize on input.
        x = preprocess_frame(frame, input_size=(
            height, width))  # x: (N, C, H, W)
        frame_downsampled = cv2.resize(
            frame, (width, height), cv2.INTER_NEAREST)

        # get inference probability map
        # please be noted that the outputed probability map will be downsampled by 4x
        # that's why we have resize everywhere
        probmap_inferred = centernet_infer(centernet_model, x, device)
        # let's try to have Gaussian blur here
        # probmap_inferred = gaussian_blur(probmap_inferred.unsqueeze(0)*3,kernel_size=5).squeeze()
        probmap_inferred_cpu_tensor.copy_(probmap_inferred, non_blocking=True)
        probmap_inferred_cpu = probmap_inferred_cpu_tensor.numpy()
        # find the point of interest
        heat_raw = get_heatmap_raw(probmap_inferred_cpu, (height, width))
        # convert into grayscale
        heat_gray = np.uint8(heat_raw*255.0)

        # get cluster centroids
        if frame_count <= 0:
            c = get_pointset(heat_gray)
            n_clusters = c.shape[0]
        c = get_pointset_kmeans(n_clusters, heat_gray)
        if frame_count <= 0:
            c0 = c # c0 is the markers point set sampled at the first frame
            d = None 
            frame0_tensor = probmap_inferred
        d = match_and_simple_filter(c0, c, d, max_change=35.0)
        heat_color = render_heatmap(probmap_inferred_cpu, (height, width))
        overlay = cv2.addWeighted(
            frame_downsampled, OVERLAY_ALPHA, heat_color, OVERLAY_BETA, 0)
        overlay = draw_keypoints(overlay, c)
        overlay = draw_keypoints(overlay, c0, color=(0, 255, 0))
        if d is not None:
            overlay = draw_displacement_vectors(overlay, c0, d*DISP_AMP_COEFF)
        if SHOW_FPS:
            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now
            cv2.putText(overlay, f"FPS: {fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        displacement = draw_displacement_vectors(
            frame, c0, d*DISP_AMP_COEFF)
        frame_row_0 = np.concatenate((overlay, heat_color), axis=0)
        frame_row_1 = np.concatenate(
            (displacement, cv2.resize(frame, INPUT_SIZE)), axis=0)
        frame = np.concatenate((frame_row_0, frame_row_1), axis=1)
        cv2.imshow("concatenated_frame", frame)
        # write to video
        if writer is not None:
            writer.write(frame)
        # regularize the c0 and c1 and the displacement vectors
        c0r = np.stack((c0[:,0]/width,c0[:,1]/height),axis=1)
        c1r = np.stack((c[:,0]/width,c[:,1]/height),axis=1)
        dr = np.stack((d[:,0]/width,d[:,1]/width),axis=1) # notice here we divide all components by width to preserve the ratio information
        pass
        


        # write to json
        data_record = {
            "frame": frame_count,
            # "disp_x_sample_based_c0": flow_vector_c0[0].astype(float),
            # "disp_y_sample_based_c0": flow_vector_c0[1].astype(float),
            # "disp_x_sample_based_c1": flow_vector_c1[0].astype(float),
            # "disp_y_sample_based_c1": flow_vector_c1[1].astype(float),
            # "disp_x_harmonics": flow_vector_harmonics[0].astype(float),
            # "disp_y_harmonics": flow_vector_harmonics[1].astype(float),
            # "disp_x_avg": flow_vector_avg[0].astype(float),
            # "disp_y_avg": flow_vector_avg[1].astype(float),
            # "Phi_max_diff": Phi_max_diff.astype(float),
            "c1r": c1r.astype(float).tolist(),
            "c0r": c1r.astype(float).tolist(),
            "dr": dr.astype(float).tolist(),

        }
        with open(DISPLACEMENT_OUTPUT_JSON_PATH, 'a', encoding='utf-8') as f:
            displacement_json = json.dumps(data_record)
            f.write(displacement_json + '\n')
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
