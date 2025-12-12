# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Carolina Li 2025 to apply 99.5th percentile scaling for 16-bit TIFF output

'''
# (optional) activate virtual env
conda init bash
source ~/.bashrc
conda activate SR-shadow-removal

python /projects/SuperResolutionData/carolinali-shadowRemoval/preprocess_raw_to_tiff.py \
  -p /projects/SuperResolutionData/driving/ROD_dataset/Dataset/RawFiles/dataset \
  -t 1 \
  -o /projects/SuperResolutionData/carolinali-shadowRemoval/raw_to_tiff_output
'''



import os
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
import cv2
import gzip
import shutil
import argparse
import numpy as np
import multiprocessing
import imageio
import re

from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F

# device selection: prefer CUDA, else MPS (Apple silicon), else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Use plain Python ints (not uint8) so they can represent 256 and 65536 safely
BIT8  = 2 ** 8   # 256
BIT16 = 2 ** 16  # 65536
BIT24 = 2 ** 24  # 16777216  (not used for scaling anymore, just for reference)


class Debayer3x3(nn.Module):
    # This code is adjusted from the following url
    # https://github.com/cheind/pytorch-debayer/blob/master/debayer/modules.py

    def __init__(self):
        super(Debayer3x3, self).__init__()

        self.kernels = nn.Parameter(
            torch.tensor([
                [0,0,0],
                [0,1,0],
                [0,0,0],
                
                [0, 0.25, 0],
                [0.25, 0, 0.25],
                [0, 0.25, 0],
                
                [0.25, 0, 0.25],
                [0, 0, 0],
                [0.25, 0, 0.25],
                
                [0, 0, 0],
                [0.5, 0, 0.5],
                [0, 0, 0],
                
                [0, 0.5, 0],
                [0, 0, 0],
                [0, 0.5, 0],
            ]).view(5,1,3,3), requires_grad=False
        )
        
        self.index = nn.Parameter(
            torch.tensor([
                # dest channel r
                [0, 3], # pixel is R,G1
                [4, 2], # pixel is G2,B
                # dest channel g
                [1, 0], # pixel is R,G1
                [0, 1], # pixel is G2,B
                # dest channel b
                [2, 4], # pixel is R,G1
                [3, 0], # pixel is G2,B
            ]).view(1,3,2,2), requires_grad=False
        )
        
    def forward(self, x):
        
        B,C,H,W = x.shape

        x = F.pad(x, (1,1,1,1), mode='replicate')
        c = F.conv2d(x, self.kernels, stride=1)
        rgb = torch.gather(c, 1, self.index.repeat(B,1,H//2,W//2))
        return rgb


def read_raw_24b(file_path, img_shape=(1, 1, 1856, 2880), read_type=np.uint8):
    """
    Read Huawei 24-bit packed Bayer RAW:
    3 bytes -> 1 uint32 pixel value.
    """
    # Read as uint8, then promote to uint32 before doing 24-bit packing
    raw_data = np.fromfile(file_path, dtype=read_type).astype(np.uint32)

    # Pack 3 bytes into one 24-bit value: L + M*256 + H*65536
    raw_data = (
        raw_data[0::3] +
        raw_data[1::3] * BIT8 +
        raw_data[2::3] * BIT16
    )

    raw_data = raw_data.reshape(img_shape).astype(np.float32)
    return raw_data


def func(filename, debayer, out_path):
    im = read_raw_24b(filename)
    im = torch.from_numpy(im).to(device).float()

    with torch.no_grad():
        im = debayer(im).detach().cpu().numpy()
    
    # Bx3xHxW -> HxWx3
    im = im.squeeze(0).transpose(1, 2, 0)
    im = cv2.resize(im, (1280, 1280), interpolation=cv2.INTER_LINEAR)

    # Simple gray-world white balance
    mean_r = im[:, :, 0].mean()
    mean_g = im[:, :, 1].mean()
    mean_b = im[:, :, 2].mean()
    im[:, :, 0] *= mean_g / (mean_r + 1e-8)
    im[:, :, 2] *= mean_g / (mean_b + 1e-8)

    # ---- NEW: 99.5th-percentile scaling into 16-bit range ----
    arr = im.astype(np.float32)
    p = np.nanpercentile(arr, 99.5)
    print(f"{os.path.basename(filename)}: p99.5 = {p}, min = {arr.min()}, max = {arr.max()}")

    if p <= 0:
        # fall back to [0,1] assumption if something is weird
        im16 = np.clip(arr, 0, 1) * (BIT16 - 1)
    else:
        im16 = np.clip(arr, 0, p) / p * (BIT16 - 1)

    im16 = np.clip(im16, 0, BIT16 - 1).astype(np.uint16)
    # ----------------------------------------------------------

    save_path = os.path.join(out_path, os.path.basename(filename))
    tiff_save_path = save_path.replace('.raw', '.tif')
    imageio.imsave(tiff_save_path, im16)


def main(args):
    in_path  = os.path.realpath(args['path']).rstrip('/') 
    out_path = os.path.realpath(args.get('out', '')).rstrip('/') if args.get('out') else None
    if not out_path:
        raise RuntimeError('Output path must be provided via --out')
    assert os.path.isdir(in_path), f'Invalid path < {args["path"]} >!'
    assert in_path != out_path, f'in_path should NOT be the same as out_path!'

    shutil.rmtree(out_path, ignore_errors=True); os.makedirs(out_path)
    print(f'input  path: {in_path}')
    print(f'output path: {out_path}')

    all_raws = sorted(glob(os.path.join(in_path, '*.raw')))
    # Filter only 'day-XXXXX.raw' files in the desired inclusive range 02000..06052
    day_pattern = re.compile(r"day-(\d+)", re.IGNORECASE)
    lines = []
    for p in all_raws:
        fn = os.path.basename(p)
        m = day_pattern.search(fn)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        if 2000 <= idx <= 6052:
            lines.append(p)

    print(f'Total raw files found: {len(all_raws)}; Day files in range 02000-06052: {len(lines)}')
    debayer = Debayer3x3().to(device)

    if args['threads'] in [0, 1]:
        print('Single thread')
        for fn in tqdm(lines):
            func(fn, debayer, out_path)
    
    else:
        if args['threads'] == -1:
            threads = multiprocessing.cpu_count() // 4
        else:
            threads = args['threads']
        print(f'{threads} threads')

        para = Parallel(n_jobs=threads, backend='threading')
        para(delayed(func)(filename, debayer, out_path) for filename in tqdm(lines))

    files_out = glob(os.path.join(out_path, '*.tif'))
    print(f'output number: {len(files_out)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path',    type=str, required=True,
                        help="Input directory containing .raw files")
    parser.add_argument('-t', '--threads', type=int, default=-1,
                        help="Number of threads (1 for single-thread, -1 for auto)")
    parser.add_argument('-o', '--out',     type=str, required=True,
                        help="Output directory for .tif files")

    args = parser.parse_args()
    main(vars(args))




