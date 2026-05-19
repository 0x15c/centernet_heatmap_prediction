"""Datasets for self-supervised tactile registration.

Expected pair dataset layout:

    data_root/
        fixed/
            000001.png
            000002.png
        moving/
            000001.png
            000002.png

or sequence-like layout:

    data_root/
        seq_001/
            ref.png
            frame_000.png
            frame_001.png
        seq_002/
            ref.png
            frame_000.png

The dataset returns grayscale tensors in [0, 1]: fixed, moving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _image_files(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def read_gray_tensor(path: Path, size: tuple[int, int]) -> torch.Tensor:
    """Read image as 1xHxW grayscale float tensor in [0, 1].

    size is (height, width).
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
    return tensor


class TactilePairDataset(Dataset):
    """Load fixed/moving image pairs for QCRegNet training."""

    def __init__(self, root: str | Path, size: tuple[int, int] = (224, 224), *, fixed_name: str = "fixed", moving_name: str = "moving"):
        self.root = Path(root)
        self.size = size
        self.samples: list[tuple[Path, Path]] = []

        fixed_dir = self.root / fixed_name
        moving_dir = self.root / moving_name
        if fixed_dir.exists() and moving_dir.exists():
            fixed_files = _image_files(fixed_dir)
            moving_files = _image_files(moving_dir)
            moving_by_name = {p.name: p for p in moving_files}
            for f in fixed_files:
                if f.name in moving_by_name:
                    self.samples.append((f, moving_by_name[f.name]))
            if not self.samples:
                raise RuntimeError(f"No matching file names found under {fixed_dir} and {moving_dir}")
            return

        # Sequence layout fallback: every frame is paired with seq/ref.png.
        for seq_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            ref = seq_dir / "ref.png"
            if not ref.exists():
                ref = seq_dir / "reference.png"
            if not ref.exists():
                continue
            for frame in _image_files(seq_dir):
                if frame.name in {"ref.png", "reference.png"}:
                    continue
                self.samples.append((ref, frame))

        if not self.samples:
            raise RuntimeError(
                f"No samples found in {self.root}. Expected fixed/moving folders or seq_x/ref.png + frame images."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        fixed_path, moving_path = self.samples[idx]
        fixed = read_gray_tensor(fixed_path, self.size)
        moving = read_gray_tensor(moving_path, self.size)
        return {
            "fixed": fixed,
            "moving": moving,
            "fixed_path": str(fixed_path),
            "moving_path": str(moving_path),
        }
