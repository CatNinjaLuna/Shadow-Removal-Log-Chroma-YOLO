"""
predict_logchroma.py

Use the log-chroma YOLO model (trained on log-chromaticity TIFFs) to generate
predictions on a folder of log-chroma TIFF images.

Results (images with bounding boxes) will be saved under:
    runs/predict_log_chroma/exp_log_chroma_predict/

# (optional) activate virtual env
conda init bash
source ~/.bashrc
conda activate SR-shadow-removal
"""

from ultralytics import YOLO
from pathlib import Path


# --------------------------------------------------------------------
# CONFIG: UPDATE THESE PATHS IF NEEDED
# --------------------------------------------------------------------

# Path to trained log-chroma weights
LOGC_WEIGHTS = Path(
    "/projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics-main/runs/train_log_chroma/exp_log_chroma/weights/best.pt"
)

# Folder containing log-chroma TIFFs for prediction 
LOGCHROMA_TIFF_DIR = Path(
    "/projects/SuperResolutionData/carolinali-shadowRemoval/output_yolo_all/log_chroma"
)
# --------------------------------------------------------------------


def main():
    if not LOGC_WEIGHTS.is_file():
        raise FileNotFoundError(f"Log-chroma weights not found: {LOGC_WEIGHTS}")

    if not LOGCHROMA_TIFF_DIR.is_dir():
        raise FileNotFoundError(f"Log-chroma TIFF directory not found: {LOGCHROMA_TIFF_DIR}")

    print(f"Loading log-chroma model from: {LOGC_WEIGHTS}")
    model = YOLO(str(LOGC_WEIGHTS))

    print(f"Running predictions on log-chroma TIFFs in: {LOGCHROMA_TIFF_DIR}")
    results = model.predict(
        source=str(LOGCHROMA_TIFF_DIR),
        imgsz=1280,
        save=True,           # save images with bounding boxes
        save_txt=False,      # set True for YOLO txt outputs
        project="runs/predict_log_chroma",
        name="exp_log_chroma_predict",
        exist_ok=True,
        verbose=True,
        conf=0.5,      # (0.25 is default)
        iou=0.6,       # slightly stricter NMS

    )

    print("\nDone! Log-chroma predictions saved in:")
    print("  runs/predict_log_chroma/exp_log_chroma_predict/")


if __name__ == "__main__":
    main()
