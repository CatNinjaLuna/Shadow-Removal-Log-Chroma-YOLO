# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Carolina Li 2025:
#   Baseline: convert linear 16-bit TIFFs to sRGB and save both
#   16-bit sRGB TIFFs and 8-bit sRGB JPGs.

'''
# (optional) activate virtual env
conda init bash
source ~/.bashrc
conda activate SR-shadow-removal
'''

"""
Baseline script:

Input  : directory of *linear* 16-bit TIFF images
         (default: /projects/SuperResolutionData/driving/ROD_dataset/Dataset/LinearFiles/dataset)
Output : 
    - 16-bit sRGB TIFFs
      /projects/SuperResolutionData/carolinali-shadowRemoval/baseline/srgb
    - 8-bit sRGB JPGs
      /projects/SuperResolutionData/carolinali-shadowRemoval/baseline/jpgs

Example (you can omit --in_dir if you want to use the default):

python convert_linear_tiff_to_srgb_jpg.py \
    --in_dir /projects/SuperResolutionData/driving/ROD_dataset/Dataset/LinearFiles/dataset \
    --srgb_dir /projects/SuperResolutionData/carolinali-shadowRemoval/baseline/srgb \
    --jpg_dir  /projects/SuperResolutionData/carolinali-shadowRemoval/baseline/jpgs
"""

import os
import argparse
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

BIT8 = 2 ** 8   # 256
BIT16 = 2 ** 16 # 65536

# If you need to resume, set this to the last successfully processed file name.
# The script will then process only files with names > LAST_PROCESSED (lexicographically).
# For example, after finishing day-04236.tif, set:
LAST_PROCESSED = "day-04236.tif"   # change or set to None / "" to disable resume


# ------------------------ sRGB helper ------------------------ #

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """
    Convert linear RGB in [0,1] to sRGB (gamma encoded) in [0,1].
    Applies the standard sRGB EOTF.
    """
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    threshold = 0.0031308

    low = x * 12.92
    high = (1 + a) * np.power(x, 1 / 2.4) - a
    return np.where(x <= threshold, low, high)


# ------------------------ core function ------------------------ #

def process_tiff_file(in_path: str, srgb_tiff_dir: str, jpg_dir: str) -> None:
    """
    Read a *linear* 16-bit TIFF, convert to sRGB, save as:
      - 16-bit sRGB TIFF
      - 8-bit sRGB JPG
    """
    # Read as-is (16-bit, 1 or 3 channels)
    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"[WARN] Failed to read: {in_path}")
        return

    # Ensure 3-channel
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # Convert to float32 linear [0,1]
    img = img.astype(np.float32) / (BIT16 - 1)

    # Convert from linear to sRGB
    srgb = linear_to_srgb(img)

    # --- Save 16-bit sRGB TIFF ---
    srgb_tiff_path = os.path.join(
        srgb_tiff_dir,
        os.path.basename(in_path)
    )
    srgb_16 = np.clip(srgb * (BIT16 - 1), 0, BIT16 - 1).astype(np.uint16)
    cv2.imwrite(srgb_tiff_path, srgb_16)

    # --- Save 8-bit sRGB JPG ---
    base_name = os.path.splitext(os.path.basename(in_path))[0]
    jpg_path = os.path.join(jpg_dir, base_name + ".jpg")
    srgb_8 = np.clip(srgb * (BIT8 - 1), 0, BIT8 - 1).astype(np.uint8)
    cv2.imwrite(jpg_path, srgb_8, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ------------------------ main ------------------------ #

def main(args):
    in_dir = os.path.realpath(args.in_dir)
    srgb_tiff_dir = os.path.realpath(args.srgb_dir)
    jpg_dir = os.path.realpath(args.jpg_dir)

    assert os.path.isdir(in_dir), f"Input directory does not exist: {in_dir}"

    os.makedirs(srgb_tiff_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)

    print(f"Input (linear TIFF) dir : {in_dir}")
    print(f"Output 16-bit sRGB dir  : {srgb_tiff_dir}")
    print(f"Output 8-bit JPG dir    : {jpg_dir}")

    # Collect TIFF files whose names start with "day-"
    all_tiff_files = sorted(
        glob(os.path.join(in_dir, "day-*.tif")) +
        glob(os.path.join(in_dir, "day-*.tiff"))
    )

    print(f"Found {len(all_tiff_files)} TIFF files matching 'day-*'")

    # ---------- Resume logic ----------
    if LAST_PROCESSED:
        remaining_files = []
        for fp in all_tiff_files:
            base = os.path.basename(fp)
            # Process only files with names > LAST_PROCESSED (lexicographic compare)
            if base > LAST_PROCESSED:
                remaining_files.append(fp)

        tiff_files = remaining_files
        print(f"Resuming after {LAST_PROCESSED}, remaining files to process: {len(tiff_files)}")
    else:
        tiff_files = all_tiff_files
        print(f"No resume point set, processing all files.")
    # -----------------------------------

    for fp in tqdm(tiff_files):
        process_tiff_file(fp, srgb_tiff_dir, jpg_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline: convert linear 16-bit TIFFs to sRGB TIFF + JPG"
    )

    parser.add_argument(
        "--in_dir",
        type=str,
        default="/projects/SuperResolutionData/driving/ROD_dataset/Dataset/LinearFiles/dataset",
        help="Directory containing *linear* 16-bit TIFF images."
    )
    parser.add_argument(
        "--srgb_dir",
        type=str,
        default="/projects/SuperResolutionData/carolinali-shadowRemoval/baseline/srgb",
        help="Directory to save 16-bit sRGB TIFFs."
    )
    parser.add_argument(
        "--jpg_dir",
        type=str,
        default="/projects/SuperResolutionData/carolinali-shadowRemoval/baseline/jpgs",
        help="Directory to save 8-bit sRGB JPGs."
    )

    args = parser.parse_args()
    main(args)
