import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single-session MLP dataset from calibration force JSONL and "
            "frontend displacement JSONL."
        )
    )
    parser.add_argument(
        "--force-jsonl",
        type=Path,
        required=True,
        help="Path to calibration force log JSONL.",
    )
    parser.add_argument(
        "--displacement-jsonl",
        type=Path,
        required=True,
        help="Path to frontend displacement log JSONL (contains c0r and dr).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .pt dataset path.",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="Holding_Shear",
        help='Target status label to keep (default: "Holding_Shear").',
    )
    parser.add_argument(
        "--status-match",
        type=str,
        choices=["exact", "contains"],
        default="exact",
        help="How to match status field.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50,
        help="Fixed number of vectors m per frame (default: 50).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio in (0,1). Default: 0.2",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size metadata to store in dataset (default: 64).",
    )
    parser.add_argument(
        "--force-frame-offset",
        type=int,
        default=0,
        help=(
            "Offset added to force frame when matching displacement frame. "
            "Example: -1 if force starts at 1 while displacement starts at 0."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    parser.add_argument(
        "--randomize-slots",
        action="store_true",
        help=(
            "Scatter valid vectors into random positions within the fixed "
            "max-points slots instead of always using the first n slots."
        ),
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_no}: {exc}"
                ) from exc


def status_matches(status: Any, target: str, mode: str) -> bool:
    if not isinstance(status, str):
        return False
    if mode == "exact":
        return status == target
    return target in status


def encode_vectors(
    c0r: np.ndarray,
    dr: np.ndarray,
    max_points: int,
    slot_rng: random.Random | None = None,
) -> np.ndarray:
    if c0r.ndim != 2 or dr.ndim != 2 or c0r.shape[1] != 2 or dr.shape[1] != 2:
        raise ValueError(f"Expected c0r and dr to be shaped (n,2), got {c0r.shape} and {dr.shape}")

    num_points = min(len(c0r), len(dr), max_points)
    features = np.zeros((max_points, 4), dtype=np.float32)
    if num_points == 0:
        return features

    if slot_rng is None:
        slot_indices = list(range(num_points))
    else:
        slot_indices = sorted(slot_rng.sample(range(max_points), num_points))

    features[slot_indices, 0:2] = c0r[:num_points]
    features[slot_indices, 2:4] = dr[:num_points]
    # features[slot_indices, 4] = 1.0
    return features


def make_batches(size: int, batch_size: int) -> List[List[int]]:
    return [list(range(i, min(i + batch_size, size))) for i in range(0, size, batch_size)]


def main() -> None:
    args = parse_args()
    if args.max_points <= 0:
        raise ValueError("--max-points must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0,1)")

    if not args.force_jsonl.exists():
        raise FileNotFoundError(args.force_jsonl)
    if not args.displacement_jsonl.exists():
        raise FileNotFoundError(args.displacement_jsonl)

    disp_by_frame: Dict[int, Dict[str, Any]] = {}
    for rec in iter_jsonl(args.displacement_jsonl):
        frame = rec.get("frame")
        if not isinstance(frame, int):
            continue
        disp_by_frame[frame] = rec

    slot_rng = random.Random(args.seed)
    samples: List[Tuple[np.ndarray, np.ndarray, int, int]] = []
    available_statuses: Dict[str, int] = {}
    filtered_total = 0
    matched_total = 0
    clipped_total = 0

    for rec in iter_jsonl(args.force_jsonl):
        status = rec.get("status")
        if isinstance(status, str):
            available_statuses[status] = available_statuses.get(status, 0) + 1

        if not status_matches(status, args.status, args.status_match):
            continue
        filtered_total += 1

        force_frame = rec.get("frame")
        pose = rec.get("pose_rel")
        force = rec.get("force")

        if not isinstance(force_frame, int):
            continue
        if not (isinstance(force, list) and len(force) >= 3):
            continue

        disp_frame = force_frame + args.force_frame_offset
        disp_rec = disp_by_frame.get(disp_frame)
        if disp_rec is None:
            continue

        c0r = np.asarray(disp_rec.get("c0r", []), dtype=np.float32)
        dr = np.asarray(disp_rec.get("dr", []), dtype=np.float32)

        if c0r.ndim != 2 or dr.ndim != 2 or c0r.shape[1] != 2 or dr.shape[1] != 2:
            continue

        n_raw = min(len(c0r), len(dr))
        if n_raw > args.max_points:
            clipped_total += 1

        x = encode_vectors(
            c0r,
            dr,
            args.max_points,
            slot_rng=slot_rng if args.randomize_slots else None,
        )
        # rot_matrix = Rotation.from_euler('xyz', pose[3:], degrees=True).as_matrix()
        # force_base = rot_matrix @ force
        force_base = force
        y = np.asarray(force_base[:3], dtype=np.float32)
        samples.append((x, y, force_frame, disp_frame))
        matched_total += 1

    if not samples:
        status_preview = ", ".join(sorted(available_statuses.keys())[:8])
        raise RuntimeError(
            "No matched samples were built. "
            f"Requested status='{args.status}' (mode={args.status_match}). "
            f"Available statuses include: {status_preview}"
        )

    rng = random.Random(args.seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    val_size = max(1, int(len(indices) * args.val_ratio))
    train_size = len(indices) - val_size
    if train_size <= 0:
        raise RuntimeError("Not enough samples for train split. Reduce --val-ratio.")

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    def pack(split_idx: List[int]) -> Dict[str, Any]:
        x = torch.tensor(np.stack([samples[i][0] for i in split_idx]), dtype=torch.float32)
        y = torch.tensor(np.stack([samples[i][1] for i in split_idx]), dtype=torch.float32)
        force_frames = torch.tensor([samples[i][2] for i in split_idx], dtype=torch.int64)
        disp_frames = torch.tensor([samples[i][3] for i in split_idx], dtype=torch.int64)
        return {
            "x": x,
            "y": y,
            "force_frame": force_frames,
            "displacement_frame": disp_frames,
            "batch_indices": make_batches(len(split_idx), args.batch_size),
        }

    dataset = {
        "meta": {
            "force_jsonl": str(args.force_jsonl),
            "displacement_jsonl": str(args.displacement_jsonl),
            "status": args.status,
            "status_match": args.status_match,
            "max_points": args.max_points,
            "feature_dim": 4,
            "batch_size": args.batch_size,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "randomize_slots": args.randomize_slots,
            "force_frame_offset": args.force_frame_offset,
            "filtered_status_count": filtered_total,
            "matched_count": matched_total,
            "clipped_count": clipped_total,
            "available_status_counts": available_statuses,
        },
        "train": pack(train_idx),
        "val": pack(val_idx),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.output)

    train_n = dataset["train"]["x"].shape[0]
    val_n = dataset["val"]["x"].shape[0]
    print(f"Saved dataset to: {args.output}")
    print(f"Samples: total={len(samples)}, train={train_n}, val={val_n}")
    print(f"Status filtered count={filtered_total}, matched count={matched_total}, clipped count={clipped_total}")


if __name__ == "__main__":
    main()
