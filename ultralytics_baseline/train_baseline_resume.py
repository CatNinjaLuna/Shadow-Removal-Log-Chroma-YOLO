import os
from pathlib import Path
from ultralytics import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300,
                        help="Target total epochs (YOLO will know where to resume)")
    parser.add_argument("--expdir", type=str,
                        default="runs/train",  #  exp where it stopped last run (eg. exp3)
                        help="Experiment directory containing weights/last.pt")
    args = parser.parse_args()

    data_yaml = "/projects/SuperResolutionData/carolinali-shadowRemoval/data/baseline_splitted_data/data.yaml"

    expdir = Path(args.expdir)
    last_ckpt = expdir / "weights" / "last.pt"

    if last_ckpt.exists():
        print(f"[INFO] Resuming from checkpoint: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        resume_flag = True
    else:
        print("[INFO] No last.pt found, starting from scratch with yolo11s.pt")
        model = YOLO("./yolo11s.pt")   # or absolute path if you prefer
        resume_flag = False

    model.train(
        data=data_yaml,
        epochs=args.epochs,       # total target; YOLO uses internal epoch counter
        resume=resume_flag,
        cache=True,
        imgsz=1280,
        single_cls=False,
        batch=16,
        close_mosaic=0,
        workers=8,
        device="0",
        optimizer="SGD",
        amp=False,
        project="runs/train",
        name="exp_baseline",    #  exp where it stopped last run (eg. exp3)
        val=False,                # 🔴 TURN OFF PER-EPOCH VALIDATION
    )


if __name__ == "__main__":
    main()
