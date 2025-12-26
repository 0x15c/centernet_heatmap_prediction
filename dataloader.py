# dataloader.py
from __future__ import annotations
import os, json, math, random
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.7) -> float:
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(0.0, b1 * b1 - 4 * a1 * c1))
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(0.0, b2 * b2 - 4 * a2 * c2))
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(0.0, b3 * b3 - 4 * a3 * c3))
    r3 = (b3 + sq3) / (2 * a3)

    return min(r1, r2, r3)


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int) -> None:
    diameter = 2 * radius + 1
    if radius <= 0:
        x, y = center
        if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
            heatmap[y, x] = max(heatmap[y, x], 1.0)
        return

    sigma = diameter / 6
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    ys = xs[:, None]
    gauss = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))

    x, y = center
    h, w = heatmap.shape

    left = min(x, radius)
    right = min(w - x, radius + 1)
    top = min(y, radius)
    bottom = min(h - y, radius + 1)

    sub_hm = heatmap[y - top : y + bottom, x - left : x + right]
    sub_g = gauss[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(sub_hm, sub_g, out=sub_hm)


class CenterNetDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        annotations: str,
        fmt: str,
        input_size: Tuple[int, int] = (640, 480),
        output_stride: int = 4,
        augment: bool = True,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.annotations_src = annotations
        self.fmt = fmt.lower()
        self.input_h, self.input_w = input_size
        self.stride = output_stride
        self.out_h, self.out_w = self.input_h // self.stride, self.input_w // self.stride
        self.augment = augment

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)

        self.image_paths: List[str] = []
        self.boxes_by_img: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
        self.num_classes: int = 0

        if self.fmt == "yolo":
            self._load_yolo()
        elif self.fmt == "coco":
            self._load_coco()
        else:
            raise ValueError("fmt must be 'yolo' or 'coco'")

    def _load_yolo(self) -> None:
        label_dir = self.annotations_src
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        for fn in os.listdir(self.images_dir):
            if not fn.lower().endswith(exts):
                continue
            img_path = os.path.join(self.images_dir, fn)
            lbl_path = os.path.join(label_dir, os.path.splitext(fn)[0] + ".txt")
            if not os.path.isfile(lbl_path):
                continue

            objs = []
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                    objs.append((cls, xc, yc, bw, bh))  # relative
                    self.num_classes = max(self.num_classes, cls + 1)

            self.image_paths.append(img_path)
            self.boxes_by_img[img_path] = objs

        if not self.image_paths:
            raise RuntimeError("No YOLO images+labels found (must have matching .txt).")

    def _load_coco(self) -> None:
        with open(self.annotations_src, "r") as f:
            coco = json.load(f)

        cat_ids = sorted([c["id"] for c in coco.get("categories", [])])
        cat2idx = {cid: i for i, cid in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)

        id2file = {img["id"]: img["file_name"] for img in coco.get("images", [])}

        ann_by_imgid: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            x, y, w, h = ann["bbox"]
            cx, cy = x + w / 2, y + h / 2
            cls = cat2idx.get(ann["category_id"], 0)
            ann_by_imgid.setdefault(img_id, []).append((cls, cx, cy, w, h))  # absolute px

        for img_id, file_name in id2file.items():
            img_path = os.path.join(self.images_dir, file_name)
            if not os.path.isfile(img_path):
                continue
            self.image_paths.append(img_path)
            self.boxes_by_img[img_path] = ann_by_imgid.get(img_id, [])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        ow, oh = img.size

        # boxes to absolute (cls, cx, cy, w, h) in pixels
        objs = self.boxes_by_img.get(img_path, [])
        abs_boxes: List[Tuple[int, float, float, float, float]] = []

        if self.fmt == "yolo":
            for cls, xc, yc, bw, bh in objs:
                abs_boxes.append((cls, xc * ow, yc * oh, bw * ow, bh * oh))
        else:
            abs_boxes = list(objs)

        # augment
        if self.augment:
            # flip
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                abs_boxes = [(cls, ow - cx, cy, bw, bh) for (cls, cx, cy, bw, bh) in abs_boxes]

            img = self.color_jitter(img)

        # resize to (512,512)
        img = img.resize((self.input_w, self.input_h), Image.BILINEAR)
        sx, sy = self.input_w / ow, self.input_h / oh
        abs_boxes = [(cls, cx * sx, cy * sy, bw * sx, bh * sy) for (cls, cx, cy, bw, bh) in abs_boxes]

        # build heatmap
        hm = np.zeros((self.num_classes, self.out_h, self.out_w), dtype=np.float32)
        centers = []
        for cls, cx, cy, bw, bh in abs_boxes:
            cx_out, cy_out = cx / self.stride, cy / self.stride
            x, y = int(cx_out), int(cy_out)
            w_out, h_out = bw / self.stride, bh / self.stride
            r = int(max(0.0, gaussian_radius((math.ceil(h_out), math.ceil(w_out)), 0.7)))
            draw_gaussian(hm[cls], (x, y), r)
            centers.append((cls, x, y, r))

        img_t = T.functional.to_tensor(img)
        img_t = self.normalize(img_t)
        hm_t = torch.from_numpy(hm)

        meta = {"path": img_path, "centers": centers}
        return img_t, hm_t, meta
