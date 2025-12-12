#!/usr/bin/env python3
"""
tiff_to_png_8bit_and_batch.py

1. Convert 16-bit TIFF log-chroma images to 8-bit PNGs.
2. Split the original TIFF files into <1 GB batches.
3. Split the generated 8-bit PNG files into <1 GB batches.

Input directory (TIFFs, ~4k files):
    /projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_tiff_4k

PNG output directory (8-bit PNGs, ~4k files):
    /projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_8_bit_png

Batch roots (batches stored "under my_training_data"):
    TIFF batches:
        /projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_tiff_batches
    PNG batches:
        /projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_png_batches
"""

import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

TIFF_SRC_DEFAULT = Path(
    "/projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_tiff_4k"
)

PNG_DST_DEFAULT = Path(
    "/projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_8_bit_png"
)

TIFF_BATCH_ROOT_DEFAULT = Path(
    "/projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_tiff_batches"
)

PNG_BATCH_ROOT_DEFAULT = Path(
    "/projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_png_batches"
)

# Keep each batch BELOW 1 GB (1,000,000,000 bytes)
BATCH_SIZE_LIMIT = 1_000_000_000   # ~0.93 GiB

TIFF_EXTS = {".tif", ".tiff"}
PNG_EXTS = {".png"}


# -------------------------------------------------------------------
# Conversion: TIFF -> 8-bit PNG
# -------------------------------------------------------------------

def convert_tiff_to_png_8bit(src_dir: Path, dst_dir: Path, overwrite: bool = False):
    dst_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(
        [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in TIFF_EXTS],
        key=lambda x: x.name
    )

    if not tiff_files:
        print(f"[INFO] No TIFF files found in {src_dir}")
        return

    print(f"[INFO] Found {len(tiff_files)} TIFF files in {src_dir}")
    print(f"[INFO] Saving 8-bit PNGs to: {dst_dir}")

    for i, tiff_path in enumerate(tiff_files, start=1):
        img = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Could not read {tiff_path}, skipping.")
            continue

        # Convert to 8-bit
        if img.dtype == np.uint16:
            # Scale 0..65535 -> 0..255
            img_8 = (img.astype(np.float32) / 257.0).round().astype(np.uint8)
        else:
            img_8 = img.astype(np.uint8)

        out_name = tiff_path.stem + ".png"
        out_path = dst_dir / out_name

        if out_path.exists() and not overwrite:
            print(f"[SKIP] {out_path} already exists (use --overwrite to replace).")
            continue

        success = cv2.imwrite(str(out_path), img_8)
        if not success:
            print(f"[WARN] Failed to write PNG for {tiff_path}")
        else:
            if i % 100 == 0 or i == len(tiff_files):
                print(f"[INFO] Converted {i}/{len(tiff_files)}: {tiff_path.name} -> {out_name}")

    print("[DONE] TIFF -> PNG conversion finished.")


# -------------------------------------------------------------------
# Generic batch splitter (<1GB per batch)
# -------------------------------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def get_next_batch_dir(root: Path, prefix: str, batch_index: int) -> Path:
    """
    Return the directory path for a given batch index, e.g.
    root / 'tiff_batch_01' or root / 'png_batch_02'.
    """
    batch_name = f"{prefix}_batch_{batch_index:02d}"
    batch_dir = root / batch_name
    ensure_dir(batch_dir)
    return batch_dir


def split_into_batches(src_dir: Path, exts, batch_root: Path, prefix: str):
    """
    Split files from src_dir (matching exts) into batch directories under batch_root.
    Each batch directory will stay below BATCH_SIZE_LIMIT bytes.

    Example:
        split_into_batches(TIFF_SRC_DEFAULT, TIFF_EXTS, TIFF_BATCH_ROOT_DEFAULT, "tiff")
    """
    if not src_dir.exists():
        print(f"[WARN] Source directory does not exist: {src_dir}")
        return

    batch_root.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda x: x.name
    )

    if not files:
        print(f"[INFO] No {prefix.upper()} files found in {src_dir} to batch.")
        return

    print(f"[INFO] Found {len(files)} {prefix.upper()} files in {src_dir}")
    print(f"[INFO] Batch size limit: {BATCH_SIZE_LIMIT} bytes "
          f"({BATCH_SIZE_LIMIT / (1024 ** 3):.3f} GiB)")
    print(f"[INFO] Batch root: {batch_root}")

    batch_index = 1
    batch_size_bytes = 0
    batch_dir = get_next_batch_dir(batch_root, prefix, batch_index)

    count_moved = 0

    for f in files:
        file_size = f.stat().st_size

        # If adding this file would exceed the limit, start a new batch
        if batch_size_bytes + file_size > BATCH_SIZE_LIMIT and batch_size_bytes > 0:
            print(
                f"[INFO] {prefix}_batch_{batch_index:02d} reached "
                f"{batch_size_bytes / (1024 ** 3):.3f} GiB -> starting new batch."
            )
            batch_index += 1
            batch_size_bytes = 0
            batch_dir = get_next_batch_dir(batch_root, prefix, batch_index)

        dst_path = batch_dir / f.name
        shutil.move(str(f), dst_path)
        batch_size_bytes += file_size
        count_moved += 1

    print(
        f"[DONE] Moved {count_moved} {prefix.upper()} files "
        f"into {batch_index} {prefix}_batch_XX directories under {batch_root}."
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert 16-bit TIFF log-chroma images to 8-bit PNGs and split both TIFFs and PNGs into <1GB batches."
    )
    parser.add_argument(
        "--tiff_src",
        type=str,
        default=str(TIFF_SRC_DEFAULT),
        help="Source directory containing TIFF files.",
    )
    parser.add_argument(
        "--png_dst",
        type=str,
        default=str(PNG_DST_DEFAULT),
        help="Destination directory for 8-bit PNG files.",
    )
    parser.add_argument(
        "--tiff_batch_root",
        type=str,
        default=str(TIFF_BATCH_ROOT_DEFAULT),
        help="Directory under which TIFF batches (<1GB) will be created.",
    )
    parser.add_argument(
        "--png_batch_root",
        type=str,
        default=str(PNG_BATCH_ROOT_DEFAULT),
        help="Directory under which PNG batches (<1GB) will be created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files if they already exist.",
    )

    args = parser.parse_args()

    tiff_src_dir = Path(args.tiff_src)
    png_dst_dir = Path(args.png_dst)
    tiff_batch_root = Path(args.tiff_batch_root)
    png_batch_root = Path(args.png_batch_root)

    if not tiff_src_dir.exists():
        raise FileNotFoundError(f"TIFF source directory does not exist: {tiff_src_dir}")

    # 1. Convert TIFF -> PNG
    convert_tiff_to_png_8bit(tiff_src_dir, png_dst_dir, overwrite=args.overwrite)

    # 2. Split TIFF files into <1GB batches
    split_into_batches(tiff_src_dir, TIFF_EXTS, tiff_batch_root, prefix="tiff")

    # 3. Split PNG files into <1GB batches
    split_into_batches(png_dst_dir, PNG_EXTS, png_batch_root, prefix="png")


if __name__ == "__main__":
    main()

"""
# (optional) activate virtual env
# conda init bash
# source ~/.bashrc
# conda activate SR-shadow-removal
#
# Run:
# python tiff_to_png_8bit_and_batch.py
"""
