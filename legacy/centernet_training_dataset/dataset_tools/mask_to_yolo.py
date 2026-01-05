import os
import shutil
import random
from pathlib import Path

import cv2
import numpy as np

# =========================
# EDIT THESE VARIABLES
# =========================

# Input folders (paired by filename stem)
IMAGES_DIR = r"marker_dataset2/data_train/imgs"   # e.g. contains 0001.png, 0002.png ...
MASKS_DIR  = r"marker_dataset2/data_train/masks"    # e.g. contains 0001.png, 0002.png ... (0/255)

# Output YOLO dataset root
OUT_DIR = r"yolo_dataset"

# Train/val split
VAL_RATIO = 0.1
SEED = 42

# Mask processing
MASK_THRESHOLD = 127          # > threshold => foreground
MIN_AREA_PIXELS = 20          # discard tiny blobs
MERGE_ALL_AS_ONE_INSTANCE = False  # True: treat the whole mask as one object bbox

# Allowed extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# YOLO class id (single class)
CLASS_ID = 0

# =========================
# IMPLEMENTATION
# =========================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_matching_mask(mask_dir: Path, stem: str) -> Path | None:
    # try common extensions
    for ext in MASK_EXTS:
        cand = mask_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None

def bbox_from_mask(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    # binary: 0/255
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2

def yolo_line_from_bbox(x1, y1, x2, y2, w, h, class_id=0) -> str:
    # convert [x1,y1,x2,y2] to YOLO [cls, cx, cy, bw, bh] normalized
    bw = (x2 - x1 + 1)
    bh = (y2 - y1 + 1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"{class_id} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"

def extract_bboxes_from_mask(mask: np.ndarray) -> list[tuple[int,int,int,int]]:
    """
    Returns list of bboxes (x1,y1,x2,y2) for each connected component.
    """
    # connected components expects 0/255 or 0/1; we use 0/1
    bin01 = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    bboxes = []
    for i in range(1, num):  # skip background 0
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA_PIXELS:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        x1, y1 = x, y
        x2, y2 = x + w - 1, y + h - 1
        bboxes.append((x1,y1,x2,y2))
    return bboxes

def main():
    random.seed(SEED)

    images_dir = Path(IMAGES_DIR)
    masks_dir  = Path(MASKS_DIR)
    out_dir    = Path(OUT_DIR)

    # YOLO structure
    train_img_out = out_dir / "images" / "train"
    val_img_out   = out_dir / "images" / "val"
    train_lbl_out = out_dir / "labels" / "train"
    val_lbl_out   = out_dir / "labels" / "val"
    ensure_dir(train_img_out); ensure_dir(val_img_out)
    ensure_dir(train_lbl_out); ensure_dir(val_lbl_out)

    # gather image files
    image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    image_files.sort()

    pairs = []
    for img_path in image_files:
        stem = img_path.stem
        mask_path = find_matching_mask(masks_dir, stem)
        if mask_path is None:
            print(f"[WARN] No mask for image: {img_path.name} (skipping)")
            continue
        pairs.append((img_path, mask_path))

    if not pairs:
        raise RuntimeError("No image-mask pairs found. Check IMAGES_DIR/MASKS_DIR and filenames.")

    # split
    random.shuffle(pairs)
    n = len(pairs)
    n_val = int(round(n * VAL_RATIO))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    def process_split(split_pairs, img_out_dir, lbl_out_dir, split_name: str):
        kept = 0
        for img_path, mask_path in split_pairs:
            # read image (to get size)
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] Cannot read image {img_path} (skipping)")
                continue
            h, w = img.shape[:2]

            # read mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[WARN] Cannot read mask {mask_path} (skipping)")
                continue
            if mask.shape[:2] != (h, w):
                print(f"[WARN] Size mismatch {img_path.name} vs {mask_path.name} (skipping)")
                continue

            # binarize
            binary = (mask > MASK_THRESHOLD).astype(np.uint8) * 255

            # get bboxes
            if MERGE_ALL_AS_ONE_INSTANCE:
                bb = bbox_from_mask(binary)
                bboxes = [bb] if bb is not None else []
            else:
                bboxes = extract_bboxes_from_mask(binary)

            # write label file
            lbl_lines = []
            for bb in bboxes:
                x1,y1,x2,y2 = bb
                # sanity clamp
                x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
                y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
                if x2 <= x1 or y2 <= y1:
                    continue
                lbl_lines.append(yolo_line_from_bbox(x1,y1,x2,y2,w,h,CLASS_ID))

            # copy image
            out_img_path = img_out_dir / img_path.name
            shutil.copy2(img_path, out_img_path)

            # save labels (empty file allowed -> “no objects” image)
            out_lbl_path = lbl_out_dir / f"{img_path.stem}.txt"
            with open(out_lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lbl_lines) + ("\n" if lbl_lines else ""))

            kept += 1

        print(f"[{split_name}] processed {kept}/{len(split_pairs)} items.")

    process_split(train_pairs, train_img_out, train_lbl_out, "train")
    process_split(val_pairs, val_img_out, val_lbl_out, "val")

    print("\nDone.")
    print(f"YOLO dataset written to: {out_dir.resolve()}")
    print("Structure:")
    print("  images/train, images/val")
    print("  labels/train, labels/val")
    print("\nSingle-class dataset: class id = 0")

if __name__ == "__main__":
    main()
