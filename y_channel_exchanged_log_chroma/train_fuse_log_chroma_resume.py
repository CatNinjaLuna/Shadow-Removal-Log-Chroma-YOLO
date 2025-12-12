"""
Resume (or start) YOLO training on the fused log-chroma dataset.

This script is consistent with train_fuse_log_chroma.py:
- Same data YAML
- Same training hyperparameters
- Same project/name layout (runs/train/exp)

Usage examples:
    python train_fuse_log_chroma_resume.py
    python train_fuse_log_chroma_resume.py --epochs 300 --expdir ./runs/train/exp
"""

import os
from pathlib import Path
from ultralytics import YOLO
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Resume or start YOLO training on fused log-chroma data.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Total target epochs for training (YOLO resumes from the last epoch if possible).",
    )
    parser.add_argument(
        "--expdir",
        type=str,
        default="./runs/train/exp",
        help="Experiment directory containing weights/last.pt (default matches train_fuse_log_chroma.py).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./yolo11s.pt",
        help="Path to the YOLO model definition or pretrained weights when starting from scratch.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/projects/SuperResolutionData/carolinali-shadowRemoval/Y_Channel_Exchange/fuse_log_chroma_data/fuse_log_chroma_data.yaml",
        help="Path to the dataset YAML file (same as train_fuse_log_chroma.py).",
    )
    return parser.parse_args()


def main():
    # Match environment tweak used in train_fuse_log_chroma.py
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    args = parse_args()

    expdir = Path(args.expdir)
    last_ckpt = expdir / "weights" / "last.pt"

    if last_ckpt.exists():
        # -----------------------------
        # Resume from last checkpoint
        # -----------------------------
        print(f"[INFO] Resuming from checkpoint: {last_ckpt}")
        model = YOLO(str(last_ckpt))

        # resume=True tells Ultralytics to reuse original training settings from the checkpoint
        model.train(
            resume=True,
            epochs=args.epochs,  # total target epochs (e.g., 300); training continues from last epoch
        )
    else:
        # -----------------------------
        # No checkpoint found: start from scratch
        # -----------------------------
        print(f"[WARN] No last.pt found at {last_ckpt}, training from scratch using {args.model}")

        model = YOLO(args.model)

        # These hyperparameters are aligned with train_fuse_log_chroma.py
        model.train(
            data=args.data,
            cache=True,
            imgsz=1280,
            epochs=args.epochs,
            single_cls=False,
            batch=16,
            close_mosaic=10,
            workers=0,
            device="0",
            optimizer="SGD",
            amp=False,
            project="runs/train",
            name="exp",
        )


if __name__ == "__main__":
    main()
