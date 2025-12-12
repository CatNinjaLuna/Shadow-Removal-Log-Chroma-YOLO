#!/usr/bin/env python3
"""
compare_yolo_side_by_side.py

This script loads YOLO detection result images from:
  - Baseline sRGB model predictions
  - Log-chroma shadow-invariant predictions

It aligns images by filename and produces a side-by-side comparison:
       [ baseline | log-chroma ]

Title:
    'YOLO Result on Original vs Shadow-Invariant Log-Chroma Image'

Usage:
    python compare_yolo_side_by_side.py
"""

import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt


# ================================
# CONFIG PATHS
# ================================
BASELINE_DIR = Path(
    "/projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics_baseline/runs/detect/baseline_predict"
)

LOGCHROMA_DIR = Path(
    "/projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics_baseline/runs/detect/log_chroma_predict"
)

OUTPUT_DIR = Path(
    "/projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics_baseline/runs/detect/compare"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================
# Helper: load & convert RGB
# ================================
def load_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ================================
# MAIN
# ================================
def main():
    baseline_files = sorted([f for f in BASELINE_DIR.iterdir() if f.suffix.lower() in {'.jpg', '.png'}])
    logchroma_files = sorted([f for f in LOGCHROMA_DIR.iterdir() if f.suffix.lower() in {'.jpg', '.png'}])

    # match by filename
    logchroma_dict = {f.name: f for f in logchroma_files}

    print(f"Found {len(baseline_files)} baseline predictions.")
    print(f"Found {len(logchroma_files)} log-chroma predictions.")
    print("Generating side-by-side comparisons...")

    for baseline_path in baseline_files:
        fname = baseline_path.name

        if fname not in logchroma_dict:
            print(f"[WARNING] No matching log-chroma file for {fname}")
            continue

        log_path = logchroma_dict[fname]

        img_left = load_image(baseline_path)
        img_right = load_image(log_path)

        if img_left is None or img_right is None:
            print(f"[ERROR] Unable to load {fname}")
            continue

        # create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle("YOLO Result on Original vs Shadow-Invariant Log-Chroma Image",
                     fontsize=14, fontweight='bold')

        axes[0].imshow(img_left)
        axes[0].set_title("Baseline (sRGB)")
        axes[0].axis('off')

        axes[1].imshow(img_right)
        axes[1].set_title("Log-Chroma (Shadow Removed)")
        axes[1].axis('off')

        out_path = OUTPUT_DIR / f"{fname.split('.')[0]}_compare.jpg"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {out_path}")

    print("Done! All comparisons saved.")


if __name__ == "__main__":
    main()
