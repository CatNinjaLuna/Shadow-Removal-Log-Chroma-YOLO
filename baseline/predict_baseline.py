"""
predict_baseline.py

Use the baseline YOLO model (trained on sRGB JPGs) to generate predictions
on a folder of sRGB JPG images. Results (images with bounding boxes) will be
saved under runs/predict_baseline/exp_baseline3_predict/.
"""

from ultralytics import YOLO
from pathlib import Path


# --------------------------------------------------------------------
# CONFIG: UPDATE THESE PATHS IF NEEDED
# --------------------------------------------------------------------

# Path to trained baseline weights
BASELINE_WEIGHTS = Path(
    "/projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics-main/runs/train_baseline/exp_baseline3/weights/best.pt"
)

# Folder containing sRGB for prediction
SRGB_DIR = Path(
    "/projects/SuperResolutionData/carolinali-shadowRemoval/baseline/srgb"
)
# --------------------------------------------------------------------


def main():
    if not BASELINE_WEIGHTS.is_file():
        raise FileNotFoundError(f"Baseline weights not found: {BASELINE_WEIGHTS}")

    if not SRGB_DIR.is_dir():
        raise FileNotFoundError(f"sRGB directory not found: {SRGB_DIR}")

    print(f"Loading baseline model from: {BASELINE_WEIGHTS}")
    model = YOLO(str(BASELINE_WEIGHTS))

    print(f"Running predictions on sRGB in: {SRGB_DIR}")
    results = model.predict(
        source=str(SRGB_DIR),
        imgsz=1280,
        save=True,           # save images with bounding boxes
        save_txt=False,      # set True for YOLO txt outputs
        project="runs/predict_baseline",
        name="exp_baseline3_predict",
        exist_ok=True,
        verbose=True,
        conf=0.5, # added since initial results has too many boxes, init default=0.25
        iou=0.6,  # slightly stricter NMS, init default=0.45
    )

    print("\nDone! Baseline predictions saved in:")
    print("  runs/predict_baseline/exp_baseline3_predict/")


if __name__ == "__main__":
    main()
