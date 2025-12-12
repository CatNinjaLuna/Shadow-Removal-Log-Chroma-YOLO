from pathlib import Path
import shutil


def move_pngs(src_dir, dst_dir):
    s = Path(src_dir)
    d = Path(dst_dir)
    if not s.is_dir():
        return 0
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in s.rglob("*.png"):
        if not f.is_file():
            continue
        t = d / f.name
        if t.exists():
            stem = t.stem
            suf = t.suffix
            i = 1
            while True:
                alt = d / f"{stem}_{i}{suf}"
                if not alt.exists():
                    t = alt
                    break
                i += 1
        shutil.move(str(f), str(t))
        n += 1
    return n


if __name__ == "__main__":
    src = r"/projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_png_batches"
    dst = r"/projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_8_bit_png"
    c = move_pngs(src, dst)
    print(c)
