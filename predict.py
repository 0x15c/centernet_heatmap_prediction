# predict.py
from __future__ import annotations
import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from model import CenterNetModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="image file or directory")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output_dir", default="predictions")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.weights, map_location=device)
    num_classes = state["head.2.bias"].shape[0]  # from model.py head[-1]

    model = CenterNetModel(num_classes=num_classes).to(device)
    model.load_state_dict(state)
    model.eval()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[
                            0.229, 0.224, 0.225])

    paths = []
    if os.path.isdir(args.images):
        for fn in os.listdir(args.images):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                paths.append(os.path.join(args.images, fn))
    else:
        paths.append(args.images)

    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)  # .resize((512, 512))
        x = T.functional.to_tensor(pil)
        x = normalize(x).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)[0]          # (C,128,128)
            # sigmoid is used here to generate probability-like output
            prob = torch.sigmoid(logits)  # (C,128,128)
            hm = prob.max(dim=0).values.cpu().numpy()

        hm_up = cv2.resize(hm, (W, H))
        hm_u8 = np.uint8(np.clip(hm_up * 255, 0, 255))
        color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(bgr, 0.65, color, 0.35, 0.0)

        out_path = os.path.join(args.output_dir, os.path.basename(p))
        cv2.imwrite(out_path, overlay)
        print("saved:", out_path)


if __name__ == "__main__":
    main()
