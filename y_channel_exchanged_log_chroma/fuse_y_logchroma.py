import os
import argparse
from pathlib import Path
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from PIL import Image
try:
    import tifffile as tiff
except Exception:
    tiff = None

IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}

def list_images(d, allowed=None):
    out = {}
    for n in os.listdir(d):
        p = os.path.join(d, n)
        if not os.path.isfile(p):
            continue
        ext = Path(n).suffix.lower()
        if (allowed is None and ext in IMAGE_EXTS) or (allowed is not None and ext in allowed):
            out[Path(n).stem] = p
    return out

def read_image(path):
    if cv2 is not None:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError('read failed')
        if arr.ndim == 2:
            rgb = np.stack([arr, arr, arr], axis=-1)
        else:
            c = arr.shape[2]
            if c == 1:
                rgb = np.repeat(arr, 3, axis=2)
            elif c == 3:
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            elif c >= 4:
                bgr = arr[:, :, :3]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                rgb = arr[:, :, :3]
        return rgb
    im = Image.open(path)
    if im.mode in ('I;16', 'I'):
        arr = np.array(im)
        rgb = np.stack([arr, arr, arr], axis=-1)
    else:
        im = im.convert('RGB')
        rgb = np.array(im)
    return rgb

def read_log(path):
    if cv2 is not None:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError('read failed')
        if arr.ndim == 2:
            arr = np.stack([arr, arr], axis=-1)
        return arr
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 2:
        arr = np.stack([arr, arr], axis=-1)
    return arr

def to_float_rgb(rgb):
    if rgb.dtype == np.uint16:
        return rgb.astype(np.float32) / 65535.0, 16
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0, 8
    arr = rgb.astype(np.float32)
    mx = float(arr.max()) if arr.size else 1.0
    if mx > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr, 32

def srgb_to_linear(c):
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1.0 + a)) ** 2.4)

def compute_y_auto(srgb_path, rgbf):
    ym, isp = detect_modes(srgb_path)
    if ym == 'luminance':
        lin = rgbf if isp == 'linear_srgb' else srgb_to_linear(rgbf)
        y = lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722
        return y, ym, isp
    if ym == 'bt601':
        y = rgbf[..., 0] * 0.299 + rgbf[..., 1] * 0.587 + rgbf[..., 2] * 0.114
        return y, ym, isp
    y = rgbf[..., 0] * 0.2126 + rgbf[..., 1] * 0.7152 + rgbf[..., 2] * 0.0722
    return y, ym, isp

def detect_modes(srgb_path):
    dtype = None
    srgb_flag = False
    gamma_val = None
    gamma_tag = None
    icc_srgb = False
    try:
        im = Image.open(srgb_path)
        info = getattr(im, 'info', {}) or {}
        g = info.get('gamma', None)
        if g is not None:
            gamma_val = float(g)
        if 'srgb' in info:
            srgb_flag = True
        icc = info.get('icc_profile', None)
        if icc is not None:
            try:
                s = icc.decode('latin-1', errors='ignore') if isinstance(icc, (bytes, bytearray)) else str(icc)
            except Exception:
                s = str(icc)
            if ('sRGB' in s) or ('IEC' in s):
                icc_srgb = True
    except Exception:
        pass
    if cv2 is not None:
        try:
            arr = cv2.imread(srgb_path, cv2.IMREAD_UNCHANGED)
            if arr is not None:
                dtype = arr.dtype
        except Exception:
            pass
    if tiff is not None:
        try:
            with tiff.TiffFile(srgb_path) as tf:
                tags = tf.pages[0].tags
                gt = tags.get('Gamma')
                if gt is not None:
                    val = gt.value
                    if isinstance(val, (list, tuple)):
                        gamma_tag = float(val[0])
                    else:
                        gamma_tag = float(val)
                icct = tags.get('ICCProfile')
                if icct is not None:
                    icb = icct.value
                    try:
                        s2 = icb.decode('latin-1', errors='ignore') if isinstance(icb, (bytes, bytearray)) else str(icb)
                    except Exception:
                        s2 = str(icb)
                    if ('sRGB' in s2) or ('IEC' in s2):
                        icc_srgb = True
                sf = tags.get('SampleFormat')
                if sf is not None and sf.value == 3:
                    return 'luminance', 'linear_srgb'
        except Exception:
            pass
    if srgb_flag or icc_srgb:
        return 'bt709', 'srgb_gamma'
    if gamma_val is not None:
        if abs(gamma_val - 1.0) < 1e-2:
            return 'luminance', 'linear_srgb'
        return 'bt709', 'srgb_gamma'
    if gamma_tag is not None:
        if abs(gamma_tag - 1.0) < 1e-2:
            return 'luminance', 'linear_srgb'
        return 'bt709', 'srgb_gamma'
    if dtype in (np.float32, np.float64, np.uint16, np.uint32):
        return 'luminance', 'linear_srgb'
    return 'bt709', 'srgb_gamma'

def select_uv(log_arr, y_norm):
    h, w = log_arr.shape[:2]
    if log_arr.ndim == 2:
        return np.stack([log_arr, log_arr], axis=-1), (0, 0), 0
    c = log_arr.shape[2]
    if c <= 2:
        return log_arr[..., :2], (0, 1), 0
    den = 65535.0 if log_arr.dtype == np.uint16 else (255.0 if log_arr.dtype == np.uint8 else float(np.max(log_arr)))
    yn = y_norm.reshape(-1)
    corrs = []
    for i in range(c):
        lc = log_arr[..., i].astype(np.float64) / den
        a = lc.reshape(-1)
        a = a - a.mean()
        b = yn - yn.mean()
        d = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
        r = float((a * b).sum() / d) if d > 0 else 0.0
        corrs.append((i, r, abs(r)))
    corrs_sorted = sorted(corrs, key=lambda x: x[2])
    y_idx = max(corrs, key=lambda x: x[2])[0]
    uv_candidates = [t for t in corrs_sorted if t[0] != y_idx]
    if len(uv_candidates) == 0:
        i1 = i2 = y_idx
    elif len(uv_candidates) == 1:
        i1 = i2 = uv_candidates[0][0]
    else:
        i1, i2 = uv_candidates[0][0], uv_candidates[1][0]
    return np.stack([log_arr[..., i1], log_arr[..., i2]], axis=-1), (i1, i2), y_idx

def select_yuv_independent(log_arr):
    h, w = log_arr.shape[:2]
    if log_arr.ndim == 2:
        return 0, (0, 0), {'score': [1.0], 'grad': [0.0], 'std': [float(np.std(log_arr))]}
    c = log_arr.shape[2]
    den = 65535.0 if log_arr.dtype == np.uint16 else (255.0 if log_arr.dtype == np.uint8 else float(np.max(log_arr)))
    grads = []
    stds = []
    norm = []
    for i in range(c):
        a = log_arr[..., i].astype(np.float64) / (den if den > 0 else 1.0)
        dx = np.diff(a, axis=1)
        dy = np.diff(a, axis=0)
        ge = float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))
        s = float(np.std(a))
        grads.append(ge)
        stds.append(s)
        norm.append(a)
    scores = [0.6 * grads[i] + 0.4 * stds[i] for i in range(c)]
    y_idx = int(np.argmax(scores))
    uv = [i for i in range(c) if i != y_idx]
    if len(uv) >= 2:
        yv = norm[y_idx] - float(np.mean(norm[y_idx]))
        corrs = []
        for j in uv:
            b = norm[j] - float(np.mean(norm[j]))
            d = np.sqrt((yv * yv).sum()) * np.sqrt((b * b).sum())
            r = float((yv * b).sum() / d) if d > 0 else 0.0
            corrs.append((j, abs(r)))
        corrs_sorted = sorted(corrs, key=lambda x: x[1])
        i1, i2 = corrs_sorted[0][0], corrs_sorted[1][0] if len(corrs_sorted) > 1 else corrs_sorted[0][0]
    elif len(uv) == 1:
        i1 = i2 = uv[0]
    else:
        i1 = i2 = y_idx
    return y_idx, (i1, i2), {'score': scores, 'grad': grads, 'std': stds}

def save_tiff(arr, out_path):
    if cv2 is not None:
        cv2.imwrite(str(out_path), arr)
        return
    if tiff is not None:
        tiff.imwrite(str(out_path), arr)
        return
    raise RuntimeError('Saving multi-channel 16-bit TIFF requires OpenCV or tifffile')

def process_pair(srgb_path, log_path, out_path, verbose=False):
    rgb = read_image(srgb_path)
    rgbf, hint = to_float_rgb(rgb)
    y, ym, isp = compute_y_auto(srgb_path, rgbf)
    log = read_log(log_path)
    h, w = log.shape[:2]
    if y.shape[:2] != (h, w):
        if cv2 is not None:
            y = cv2.resize(y, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            y = np.array(Image.fromarray((y * 65535.0).astype(np.uint16)).resize((w, h))) / 65535.0
    if log.ndim == 3 and log.shape[2] >= 3:
        log2 = np.stack([log[..., 1], log[..., 2]], axis=-1)
        uv_idx = (1, 2)
    elif log.ndim == 3 and log.shape[2] == 2:
        log2 = log
        uv_idx = (0, 1)
    elif log.ndim == 2:
        log2 = np.stack([log, log], axis=-1)
        uv_idx = (0, 0)
    else:
        log2 = np.stack([log[..., 1], log[..., 2]], axis=-1)
        uv_idx = (1, 2)
    y_idx = 0
    if log2.dtype == np.uint8:
        lc = log2.astype(np.uint16) * 257
    elif log2.dtype == np.uint16:
        lc = log2
    else:
        lc = np.clip(log2, 0.0, 1.0)
        lc = (lc * 65535.0 + 0.5).astype(np.uint16)
    y16 = (np.clip(y, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    fused = np.stack([y16, lc[..., 0], lc[..., 1]], axis=-1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_tiff(fused, out_path)
    if verbose:
        bn = Path(srgb_path).name
        print(f'{bn} y_mode={ym} input_space={isp} y_idx={y_idx} uv_idx={uv_idx[0]},{uv_idx[1]}')

def process_dirs(srgb_dir, log_dir, out_dir, verbose=False):
    # list inputs
    sim = list_images(srgb_dir, allowed=IMAGE_EXTS)
    lim = list_images(log_dir, allowed={'.tif', '.tiff', '.png'})
    base = sorted(set(sim.keys()) & set(lim.keys()))

    # NEW: detect already-processed outputs to allow resume
    out_existing = list_images(out_dir, allowed={'.tif', '.tiff'}) if os.path.isdir(out_dir) else {}

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for b in base:
        op = os.path.join(out_dir, b + '.tif')
        # If output already exists, skip (resume behavior)
        if b in out_existing or os.path.exists(op):
            if verbose:
                print(f'skipping {b}.tif (already exists)')
            skipped += 1
            continue

        sp = sim[b]
        lp = lim[b]
        process_pair(sp, lp, op, verbose=verbose)
        processed += 1

    print(f'processed {processed} new pairs, skipped {skipped} existing')

def analyze_log_only(log_dir, verbose=True):
    lim = list_images(log_dir, allowed={'.tif', '.tiff', '.png'})
    base = sorted(lim.keys())
    for b in base:
        lp = lim[b]
        log = read_log(lp)
        y_idx, uv_idx, metrics = select_yuv_independent(log)
        if verbose:
            print(f'{Path(lp).name} y_idx_ind={y_idx} uv_idx_ind={uv_idx[0]},{uv_idx[1]} '
                  f'score={metrics["score"]} grad={metrics["grad"]} std={metrics["std"]}')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--srgb', type=str, default=r"/projects/SuperResolutionData/carolinali-shadowRemoval/raw_to_tiff_output")
    p.add_argument('--log_chroma', type=str, default=r"/projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_tiff_4k/log_chroma")
    p.add_argument('--out', type=str, default=r"/projects/SuperResolutionData/carolinali-shadowRemoval/Y_Channel_Exchange/fuse_output")
    p.add_argument('--verbose', type=bool, default=True)
    p.add_argument('--analyze_log_only', action='store_true')
    a = p.parse_args()
    if a.analyze_log_only:
        analyze_log_only(a.log_chroma, verbose=a.verbose)
        return
    process_dirs(a.srgb, a.log_chroma, a.out, verbose=a.verbose)

if __name__ == '__main__':
    main()
