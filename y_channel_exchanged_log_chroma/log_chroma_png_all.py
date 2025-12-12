from pathlib import Path
import shutil
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

SRC_DIR = Path(r"/projects/SuperResolutionData/carolinali-shadowRemoval/Y_Channel_Exchange/fuse_output")
DST_DIR = Path(r"/projects/SuperResolutionData/carolinali-shadowRemoval/Y_Channel_Exchange/fuse_log_chroma_png_8bit")

def convert_tiff_to_png_8bit(src_dir: Path, dst_dir: Path) -> int:
    """
    Convert all .tif/.tiff images in src_dir to 8-bit PNGs in dst_dir.
    If images are already uint8, they are just saved as PNG.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    exts = {".tif", ".tiff"}
    for f in src_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in exts:
            continue

        # Output name: same stem, .png extension
        out_path = dst_dir / (f.stem + ".png")
        if out_path.exists():
            # skip if already converted (resume behavior)
            print(f"Skipping existing {out_path.name}")
            continue

        if cv2 is None:
            raise RuntimeError("cv2 is required for TIFF -> PNG conversion")

        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: failed to read {f}")
            continue

        # If 16-bit, scale to 8-bit
        if img.dtype == np.uint16:
            img8 = (img / 257.0).astype("uint8")
        elif img.dtype == np.float32 or img.dtype == np.float64:
            img_clipped = np.clip(img, 0.0, 1.0)
            img8 = (img_clipped * 255.0 + 0.5).astype("uint8")
        else:
            # already uint8 or something compatible
            img8 = img

        cv2.imwrite(str(out_path), img8)
        count += 1
        print(f"Converted {f.name} -> {out_path.name}")

    return count


if __name__ == "__main__":
    n = convert_tiff_to_png_8bit(SRC_DIR, DST_DIR)
    print(f"Total converted: {n}")
