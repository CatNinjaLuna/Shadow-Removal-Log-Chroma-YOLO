import os
from pathlib import Path
from ultralytics import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Target total epochs (YOLO will know where to resume)",
    )
    parser.add_argument(
        "--expdir",
        type=str,
        # ✅ 用绝对路径，指向真正的 exp2
        default="/projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics-src-new/runs/train/exp2",
        help="Experiment directory containing weights/last.pt",
    )
    args = parser.parse_args()

    data_yaml = "/projects/SuperResolutionData/carolinali-shadowRemoval/data/log_chroma_splitted_data/data.yaml"

    expdir = Path(args.expdir)
    last_ckpt = expdir / "weights" / "last.pt"

    if last_ckpt.exists():
        print(f"[INFO] Resuming from checkpoint: {last_ckpt}")
        model = YOLO(str(last_ckpt))

        # ✅ 这里用 resume=True，让它沿用原来的 project/name/epochs 等配置
        model.train(
            resume=True,
            epochs=args.epochs,  # 目标总 epoch=300，Ultralytics 会从上次的 epoch 继续算
        )
    else:
        print(f"[WARN] No last.pt found at {last_ckpt}, starting from scratch with yolo11s.pt")

        model = YOLO("/projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics-src-new/yolo11s.pt")

        # ✅ 从头训的时候才指定 data / project / name 等
        model.train(
            data=data_yaml,
            epochs=args.epochs,
            cache=False,
            imgsz=1280,
            single_cls=False,
            batch=16,
            close_mosaic=0,
            workers=8,
            device="0",
            optimizer="SGD",
            amp=False,
            project="runs/train",
            name="exp2",
            val=False,
        )


if __name__ == "__main__":
    main()
