import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from sklearn.cluster import DBSCAN

from centernet.centernet_model import CenterNetModel
from helmholtz import helmholtz_hodge_2d_fft
from voxelmorph.model import VoxelMorph2D


WEIGHTS_PATH = "centernet/checkpoints/centernet_resnet9_e35.pth"
WEIGHTS_PATH_VOXELMORPH = "voxelmorph/ckpt/voxelmorph2d_images_20_new_sensor.pt"
INPUT_SIZE = (600, 460)  # (W, H)
DEFAULT_OUTPUT_JSONL = "mlp_force_prediction/test.jsonl"
DEFAULT_FORCE_OUTPUT_JSONL = "mlp_force_prediction/test_force.jsonl"
DEFAULT_IMAGE_GLOB = "*.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pointset displacement JSONL from an image directory using an "
            "explicit reference image, and optionally export aligned force labels "
            "to a second JSONL."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing tactile images to process in sorted order.",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        required=True,
        help="Reference image used everywhere the old pipeline used frame 0.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(DEFAULT_OUTPUT_JSONL),
        help="Output pointset/displacement JSONL path.",
    )
    parser.add_argument(
        "--force-label-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing per-image force txt labels. If provided, the "
            "script also writes a force JSONL."
        ),
    )
    parser.add_argument(
        "--force-output-jsonl",
        type=Path,
        default=Path(DEFAULT_FORCE_OUTPUT_JSONL),
        help="Output force-label JSONL path.",
    )
    parser.add_argument(
        "--image-glob",
        type=str,
        default=DEFAULT_IMAGE_GLOB,
        help="Glob for sequence images inside --image-dir.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device.",
    )
    return parser.parse_args()


def select_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_centernet_model(weights_path: str, device: torch.device) -> CenterNetModel:
    state = torch.load(weights_path, map_location=device)
    if "head.2.bias" in state:
        num_classes = state["head.2.bias"].shape[0]
        model = CenterNetModel(num_classes=num_classes)
        model.load_state_dict(state, strict=True)
    else:
        model = state
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError("Unsupported CenterNet checkpoint format.")
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


def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, INPUT_SIZE, cv2.INTER_NEAREST)
    x = T.functional.to_tensor(frame_rgb)
    x = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(x)
    return x.unsqueeze(0)


@torch.no_grad()
def centernet_infer(model: CenterNetModel, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    logits = model(x.to(device))
    prob = torch.sigmoid(logits)[0]
    if prob.shape[0] > 1:
        return prob.max(dim=0).values
    return prob[0]


@torch.no_grad()
def voxelmorph_infer(model: VoxelMorph2D, moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    _, flow = model(moving, fixed)
    return flow


def get_heatmap_raw(heat: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    height, width = out_shape
    return cv2.resize(heat, (width, height), cv2.INTER_NEAREST)


def dbscan_extractor(dbscan_result, points: np.ndarray) -> list[np.ndarray]:
    labels = dbscan_result.labels_
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    return [points[labels == cluster_id] for cluster_id in unique_labels]


def centroids_calc(cluster_array: list[np.ndarray]) -> np.ndarray:
    if not cluster_array:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([np.mean(cluster, axis=0) for cluster in cluster_array], dtype=np.float32)


def get_pointset(heatmap_uint8: np.ndarray) -> np.ndarray:
    threshold = 50
    eps = 3
    min_samples = 2
    _, activate = cv2.threshold(heatmap_uint8, threshold, 255, cv2.THRESH_BINARY)
    pts = cv2.findNonZero(activate)
    if pts is None:
        return np.zeros((0, 2), dtype=np.float32)

    cluster_coordinates = pts.reshape(-1, 2)
    cluster_data = DBSCAN(eps=eps, min_samples=min_samples).fit(cluster_coordinates)
    return centroids_calc(dbscan_extractor(cluster_data, cluster_coordinates))


def blur_flow_field(flow: np.ndarray, ksize: int = 3) -> np.ndarray:
    u = cv2.blur(flow[0], (ksize, ksize))
    v = cv2.blur(flow[1], (ksize, ksize))
    return np.stack([u, v], axis=0)


def sample_flow_at_points(flow: np.ndarray, points: np.ndarray, radius: int = 5) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    height, width = flow.shape[1], flow.shape[2]
    pts = np.asarray(points, dtype=np.float32)
    xs = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, width - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, height - 1)

    if radius <= 0:
        u = flow[0, ys, xs]
        v = flow[1, ys, xs]
    else:
        ksize = 2 * radius + 1
        u_blur = cv2.blur(flow[0], (ksize, ksize), borderType=cv2.BORDER_REFLECT)
        v_blur = cv2.blur(flow[1], (ksize, ksize), borderType=cv2.BORDER_REFLECT)
        u = u_blur[ys, xs]
        v = v_blur[ys, xs]

    return np.stack([u, v], axis=1).astype(np.float32)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def list_images(image_dir: Path, image_glob: str) -> list[Path]:
    image_paths = sorted(image_dir.glob(image_glob))
    if not image_paths:
        raise RuntimeError(f"No images matched '{image_glob}' in {image_dir}")
    return image_paths


def make_record(
    frame_index: int,
    current_points: np.ndarray,
    reference_points: np.ndarray,
    displacement_at_reference: np.ndarray,
    flow: np.ndarray,
    flow_upsampled: np.ndarray,
) -> dict:
    height, width = INPUT_SIZE[1], INPUT_SIZE[0]
    _, _, flow_harmonic, phi, _ = helmholtz_hodge_2d_fft(flow, return_potentials=True)

    phi_max_diff = float(np.max(phi) - np.min(phi))
    flow_vector_avg = np.mean(flow, axis=(1, 2))
    flow_vector_c0 = (
        np.mean(displacement_at_reference, axis=0)
        if len(displacement_at_reference) > 0
        else np.zeros(2, dtype=np.float32)
    )
    flow_vector_c1 = (
        np.mean(sample_flow_at_points(flow_upsampled, current_points, radius=2), axis=0)
        if len(current_points) > 0
        else np.zeros(2, dtype=np.float32)
    )
    flow_vector_harmonics = np.mean(flow_harmonic, axis=(1, 2))

    c0r = (
        np.stack((reference_points[:, 0] / width, reference_points[:, 1] / height), axis=1)
        if len(reference_points) > 0
        else np.zeros((0, 2), dtype=np.float32)
    )
    c1r = (
        np.stack((current_points[:, 0] / width, current_points[:, 1] / height), axis=1)
        if len(current_points) > 0
        else np.zeros((0, 2), dtype=np.float32)
    )
    dr = (
        np.stack(
            (
                displacement_at_reference[:, 0] / width,
                displacement_at_reference[:, 1] / width,
            ),
            axis=1,
        )
        if len(displacement_at_reference) > 0
        else np.zeros((0, 2), dtype=np.float32)
    )

    return {
        "frame": frame_index,
        "disp_x_sample_based_c0": float(flow_vector_c0[0]),
        "disp_y_sample_based_c0": float(flow_vector_c0[1]),
        "disp_x_sample_based_c1": float(flow_vector_c1[0]),
        "disp_y_sample_based_c1": float(flow_vector_c1[1]),
        "disp_x_harmonics": float(flow_vector_harmonics[0]),
        "disp_y_harmonics": float(flow_vector_harmonics[1]),
        "disp_x_avg": float(flow_vector_avg[0]),
        "disp_y_avg": float(flow_vector_avg[1]),
        "Phi_max_diff": phi_max_diff,
        "c1r": c1r.astype(float).tolist(),
        "c0r": c0r.astype(float).tolist(),
        "dr": dr.astype(float).tolist(),
    }


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def resolve_force_label_path(image_path: Path, label_dir: Path) -> Path:
    candidates = [label_dir / f"{image_path.stem}.txt"]
    if image_path.stem.startswith("raw_"):
        candidates.insert(0, label_dir / f"data_{image_path.stem[len('raw_'):]}.txt")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_names = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(
        f"No matching force label txt found for image {image_path.name}. Tried: {candidate_names}"
    )


def parse_force_label(force_label_path: Path) -> list[float]:
    line = force_label_path.read_text(encoding="utf-8").strip()
    if not line:
        raise RuntimeError(f"Force label file is empty: {force_label_path}")

    values = [float(x.strip()) for x in line.split(",")]
    if len(values) < 3:
        raise RuntimeError(
            f"Force label file must contain at least 3 comma-separated values: {force_label_path}"
        )
    return values[:3]


def build_force_records(image_paths: list[Path], force_label_dir: Path) -> list[dict]:
    records = []
    for frame_index, image_path in enumerate(image_paths):
        force_label_path = resolve_force_label_path(image_path, force_label_dir)
        force = parse_force_label(force_label_path)
        records.append(
            {
                "frame": frame_index,
                "force": force,
                "status": "Holding_Shear",
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if not args.image_dir.exists():
        raise FileNotFoundError(args.image_dir)
    if not args.reference_image.exists():
        raise FileNotFoundError(args.reference_image)
    if args.force_label_dir is not None and not args.force_label_dir.exists():
        raise FileNotFoundError(args.force_label_dir)

    device = select_device(args.device)
    print(f"Using device: {device}")

    centernet_model = get_centernet_model(WEIGHTS_PATH, device)
    voxelmorph_model = get_voxelmorph_model(WEIGHTS_PATH_VOXELMORPH, device)

    image_paths = list_images(args.image_dir, args.image_glob)
    reference_image = read_image(args.reference_image)
    height, width = INPUT_SIZE[1], INPUT_SIZE[0]

    reference_tensor = preprocess_frame(reference_image)
    reference_probmap = centernet_infer(centernet_model, reference_tensor, device)
    reference_probmap_cpu = reference_probmap.cpu().numpy()
    reference_heat = get_heatmap_raw(reference_probmap_cpu, (height, width))
    reference_points = get_pointset(np.uint8(reference_heat * 255.0))
    if len(reference_points) == 0:
        raise RuntimeError("No reference points were detected in the reference image.")

    pointset_records = []
    for frame_index, image_path in enumerate(image_paths):
        frame = read_image(image_path)
        frame_tensor = preprocess_frame(frame)

        probmap = centernet_infer(centernet_model, frame_tensor, device)
        probmap_cpu = probmap.cpu().numpy()
        heat_raw = get_heatmap_raw(probmap_cpu, (height, width))
        current_points = get_pointset(np.uint8(heat_raw * 255.0))

        flow_gpu = voxelmorph_infer(
            voxelmorph_model,
            probmap[None, None],
            reference_probmap[None, None],
        ).squeeze(0)
        flow = flow_gpu.cpu().numpy()
        flow_blurred = blur_flow_field(flow, ksize=3)

        h_small, w_small = flow_blurred.shape[1], flow_blurred.shape[2]
        scale_x = width / w_small
        scale_y = height / h_small
        u = cv2.resize(flow_blurred[0], (width, height), interpolation=cv2.INTER_LINEAR) * scale_x
        v = cv2.resize(flow_blurred[1], (width, height), interpolation=cv2.INTER_LINEAR) * scale_y
        flow_upsampled = np.stack([u, v], axis=0).astype(np.float32)
        displacement_at_reference = sample_flow_at_points(flow_upsampled, reference_points, radius=2)

        pointset_records.append(
            make_record(
                frame_index=frame_index,
                current_points=current_points,
                reference_points=reference_points,
                displacement_at_reference=displacement_at_reference,
                flow=flow,
                flow_upsampled=flow_upsampled,
            )
        )

    write_jsonl(args.output_jsonl, pointset_records)

    force_records = None
    if args.force_label_dir is not None:
        force_records = build_force_records(image_paths, args.force_label_dir)
        write_jsonl(args.force_output_jsonl, force_records)

    print(f"Reference image: {args.reference_image}")
    print(f"Sequence images: {len(image_paths)}")
    print(f"Saved pointset JSONL to: {args.output_jsonl}")
    if force_records is not None:
        print(f"Saved force JSONL to: {args.force_output_jsonl}")


if __name__ == "__main__":
    main()
