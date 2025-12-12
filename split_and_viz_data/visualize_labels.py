'''
# (optional) activate virtual env
conda init bash
source ~/.bashrc
conda activate SR-shadow-removal
'''

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

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

def to_uint8_image(img):
    arr = np.array(img)
    if arr.dtype == np.uint16:
        if arr.ndim == 2:
            arr8 = (arr / 257.0).astype(np.uint8)
            return Image.fromarray(arr8, mode='L').convert('RGB')
        else:
            arr8 = (arr / 257.0).astype(np.uint8)
            return Image.fromarray(arr8)
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img

def draw_boxes(img_path, label_path, out_path, names=None):
    im = Image.open(img_path)
    im = to_uint8_image(im)
    w, h = im.size
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    dr = ImageDraw.Draw(im)
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            continue
        ci = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])
        x1 = int((cx - bw/2.0) * w)
        y1 = int((cy - bh/2.0) * h)
        x2 = int((cx + bw/2.0) * w)
        y2 = int((cy + bh/2.0) * h)
        color = colors[ci % len(colors)]
        dr.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if names and 0 <= ci < len(names):
            dr.text((x1+2, y1+2), names[ci], fill=color)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', type=str, default="/projects/SuperResolutionData/carolinali-shadowRemoval/output_yolo_all/log_chroma")
    p.add_argument('--labels', type=str,default="/projects/SuperResolutionData/carolinali-shadowRemoval/data/log_chroma_txt")
    p.add_argument('--out', type=str, default="/projects/SuperResolutionData/carolinali-shadowRemoval/data/log_chroma_vis")
    p.add_argument('--names', type=str, nargs='*', default=['Car','Cyclist','Pedestrian','Tram','Truck'])
    args = p.parse_args()
    imap = list_images(args.images)
    lmap = list_labels(args.labels)
    base = sorted(set(imap.keys()) & set(lmap.keys()))
    for b in base:
        outp = os.path.join(args.out, b + '.png')
        draw_boxes(imap[b], lmap[b], outp, args.names)

if __name__ == '__main__':
    main()