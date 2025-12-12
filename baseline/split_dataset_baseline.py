import os
import argparse
import shutil
from pathlib import Path
import random

IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}

def list_images(d):
    out = {}
    for n in os.listdir(d):
        p = os.path.join(d, n)
        if not os.path.isfile(p):
            continue
        ext = Path(n).suffix.lower()
        if ext in IMAGE_EXTS:
            out[Path(n).stem] = p
    return out

def list_labels(d):
    out = {}
    for n in os.listdir(d):
        p = os.path.join(d, n)
        if not os.path.isfile(p):
            continue
        if Path(n).suffix.lower() == '.txt':
            out[Path(n).stem] = p
    return out

def ensure_dirs(root):
    for sub in ['train/images', 'train/labels', 'val/images', 'val/labels']:
        Path(root, sub).mkdir(parents=True, exist_ok=True)

def split_pairs(images_dir, labels_dir, out_dir, ratio, seed):
    imap = list_images(images_dir)
    lmap = list_labels(labels_dir)
    base = sorted(set(imap.keys()) & set(lmap.keys()))
    if not base:
        print('no matched pairs')
        return
    random.seed(seed)
    random.shuffle(base)
    n = int(len(base) * ratio)
    train = base[:n]
    val = base[n:]
    ensure_dirs(out_dir)
    for b in train:
        shutil.copy2(imap[b], Path(out_dir, 'train/images', Path(imap[b]).name))
        shutil.copy2(lmap[b], Path(out_dir, 'train/labels', Path(lmap[b]).name))
    for b in val:
        shutil.copy2(imap[b], Path(out_dir, 'val/images', Path(imap[b]).name))
        shutil.copy2(lmap[b], Path(out_dir, 'val/labels', Path(lmap[b]).name))
    print(f'split done: total {len(base)} train {len(train)} val {len(val)}')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', type=str, default="/projects/SuperResolutionData/carolinali-shadowRemoval/baseline/jpgs")
    p.add_argument('--labels', type=str, default=r"/projects/SuperResolutionData/carolinali-shadowRemoval/data/log_chroma_txt") # corrected label -> (json to correct image size [L, W, H]) txt)
    p.add_argument('--out', type=str, default="/projects/SuperResolutionData/carolinali-shadowRemoval/data/baseline_splitted_data")
    p.add_argument('--ratio', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    split_pairs(a.images, a.labels, a.out, a.ratio, a.seed)

if __name__ == '__main__':
    main()