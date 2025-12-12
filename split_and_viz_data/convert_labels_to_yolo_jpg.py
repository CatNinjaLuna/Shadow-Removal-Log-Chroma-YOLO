

import os
import json
import argparse
from pathlib import Path
import cv2


def list_jsons(d):
    return sorted([p for p in os.listdir(d) if p.lower().endswith('.json')])


def read_labels(label_dir):
    names = set()
    files = list_jsons(label_dir)
    for f in files:
        with open(os.path.join(label_dir, f), 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        for s in data.get('shapes', []):
            lab = str(s.get('label', '')).strip()
            if lab:
                names.add(lab)
    return files, sorted(names)


def build_map(names):
    return {n: i for i, n in enumerate(names)}


def find_image(image_dir, base, exts):
    for e in exts:
        p = os.path.join(image_dir, base + e)
        if os.path.isfile(p):
            return p
    return None


def get_image_size(img_path):
    if img_path and os.path.isfile(img_path):
        im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError(f"ERROR: cv2.imread failed for {img_path}")
        h, w = im.shape[:2]
        return h, w
    else:
        raise FileNotFoundError(f"ERROR: image file not found: {img_path}")


def rect_to_yolo(x1, y1, x2, y2, w, h):
    x1, x2 = float(min(x1, x2)), float(max(x1, x2))
    y1, y2 = float(min(y1, y2)), float(max(y1, y2))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / float(w), cy / float(h), bw / float(w), bh / float(h)


def convert(label_dir, image_dir, out_dir, image_exts):
    os.makedirs(out_dir, exist_ok=True)
    files, names = read_labels(label_dir)
    cls_map = build_map(names)
    print('classes', names)

    for i, f in enumerate(files):
        fp = os.path.join(label_dir, f)
        with open(fp, 'r', encoding='utf-8') as fi:
            data = json.load(fi)

        h_json = int(data.get('imageHeight', 0) or 0)
        w_json = int(data.get('imageWidth', 0) or 0)

        base = Path(f).stem
        imgp = find_image(image_dir, base, image_exts)
        h_img, w_img = get_image_size(imgp)

        # DEBUG / SAFETY CHECK
        if h_json > 0 and w_json > 0 and (h_json != h_img or w_json != w_img):
            raise ValueError(
                f"Size mismatch for {base}: "
                f"json=({w_json}x{h_json}), image=({w_img}x{h_img}).\n"
                f"Labels were drawn on a different resolution than the JPGs."
            )

        # Use the actual image size for YOLO normalization
        h, w = h_img, w_img

        lines = []
        for s in data.get('shapes', []):
            if str(s.get('shape_type', '')).lower() != 'rectangle':
                continue
            pts = s.get('points', [])
            if not isinstance(pts, list) or len(pts) != 2:
                continue

            x1, y1 = pts[0]
            x2, y2 = pts[1]
            cx, cy, bw, bh = rect_to_yolo(x1, y1, x2, y2, w, h)

            lab = str(s.get('label', '')).strip()
            if lab not in cls_map:
                continue
            ci = cls_map[lab]

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))

            lines.append(f"{ci} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        outp = os.path.join(out_dir, base + '.txt')
        with open(outp, 'w', encoding='utf-8') as fo:
            fo.write('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--label_dir',
        type=str,
        default='/projects/SuperResolutionData/carolinali-shadowRemoval/data/label'
    )
    p.add_argument(
        '--image_dir',
        type=str,
        default='/projects/SuperResolutionData/carolinali-shadowRemoval/data/log_chroma_unsplitted_jpg'
    )
    p.add_argument(
        '--out_dir',
        type=str,
        default='/projects/SuperResolutionData/carolinali-shadowRemoval/data/log_chroma_jpg_txt'
    )
    p.add_argument(
        '--exts',
        type=str,
        nargs='*',
        default=['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    )
    args = p.parse_args()
    convert(args.label_dir, args.image_dir, args.out_dir, args.exts)


if __name__ == '__main__':
    main()
