"""
Batch conversion:
All 16-bit TIFFs in raw_to_tiff_output/ -> log-chroma images

No GUI. Works on cluster.
"""

import os
import cv2
import numpy as np
import logging
import torch
from pathlib import Path

from model.unet_models2 import ResNet50UNet


# -----------------------------------------------------------------------------------------------
# PATH CONFIG (update if needed)
# -----------------------------------------------------------------------------------------------
INPUT_DIR = "/projects/SuperResolutionData/carolinali-shadowRemoval/raw_to_tiff_output"
OUTPUT_DIR = "/projects/SuperResolutionData/carolinali-shadowRemoval/log_chroma_output"
MODEL_PATH = "/projects/SuperResolutionData/carolinali-shadowRemoval/model/UNET_run_x10_01_last_model.pth"

DEVICE = "cpu"      # or "cuda" if GPU available
ANCHOR = 10.4
SAVE_PREVIEW = True
FACTOR = 32         # UNet requires dims divisible by 32


# -----------------------------------------------------------------------------------------------
# PROCESSOR CLASS
# -----------------------------------------------------------------------------------------------

class imgProcessor:
    def __init__(self, image: np.ndarray, sr_map: np.ndarray, filename: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.image = image
        self.sr_map = sr_map
        self.filename = filename

    def convert_img_to_log_space(self, linear_img):
        log_img = np.zeros_like(linear_img, dtype=np.float32)
        mask = linear_img > 0
        log_img[mask] = np.log(linear_img[mask])
        return log_img

    def log_to_linear(self, log_img):
        return np.exp(log_img).astype(np.float32)

    def compute_log_chroma(self, anchor=10.4):
        log_img = self.convert_img_to_log_space(self.image)

        shifted = log_img - anchor
        dot = np.einsum("ijk,ijk->ij", shifted, self.sr_map)
        projection = dot[:, :, None] * self.sr_map
        plane = shifted - projection
        plane += anchor

        lin = self.log_to_linear(plane)
        lin = np.clip(lin, 0, 65535).astype(np.uint16)
        return lin

    @staticmethod
    def to_8bit_preview(img):
        img = np.clip(img.astype(np.float32), 0, 65535)
        return (img / 256.0).astype(np.uint8)


# -----------------------------------------------------------------------------------------------
# ISD ESTIMATOR
# -----------------------------------------------------------------------------------------------

class ISDMapEstimator:
    def __init__(self, model, model_path, device):
        self.device = torch.device(device)
        self.model = model

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, img):
        log_img = np.zeros_like(img, dtype=np.float32)
        mask = img > 0
        log_img[mask] = np.log(img[mask])
        log_img = log_img / 11.1
        t = torch.from_numpy(log_img.transpose(2,0,1)).float().unsqueeze(0).to(self.device)
        return t

    def predict(self, img):
        x = self._preprocess(img)
        with torch.no_grad():
            out = self.model(x)
        out = out.squeeze(0).permute(1,2,0).cpu().numpy()

        norm = np.linalg.norm(out, axis=2, keepdims=True)
        norm[norm == 0] = 1.0
        return out / norm


# -----------------------------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------------------------

def crop_even(img):
    h, w = img.shape[:2]
    return img[:h - (h % 2), :w - (w % 2)]

def center_crop(img, factor):
    h, w = img.shape[:2]
    nh = (h // factor) * factor
    nw = (w // factor) * factor
    sh = (h - nh) // 2
    sw = (w - nw) // 2
    return img[sh:sh+nh, sw:sw+nw]


# -----------------------------------------------------------------------------------------------
# MAIN BATCH PIPELINE
# -----------------------------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    logger = logging.getLogger("batch")

    # Create output folder
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load model
    model = ResNet50UNet(in_channels=3, out_channels=3, pretrained=False, checkpoint=None, se_block=True)
    estimator = ISDMapEstimator(model, MODEL_PATH, DEVICE)

    # Find all TIFF files
    input_dir = Path(INPUT_DIR)
    tif_files = sorted(list(input_dir.glob("day-*.tif")))

    logger.info(f"Found {len(tif_files)} TIFFs to process.")

    for tif_path in tif_files:
        img_name = tif_path.stem
        logger.info(f"Processing {img_name} ...")

        img = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error(f"Failed to load image: {tif_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_even(img)
        img = center_crop(img, FACTOR)

        # Predict ISD
        sr_map = estimator.predict(img)

        # Compute log chroma
        proc = imgProcessor(img, sr_map, img_name)
        chroma16 = proc.compute_log_chroma(anchor=ANCHOR)

        # Save 16-bit output
        out_path_tif = Path(OUTPUT_DIR) / f"{img_name}_log_chroma.tif"
        cv2.imwrite(str(out_path_tif), cv2.cvtColor(chroma16, cv2.COLOR_RGB2BGR))

        if SAVE_PREVIEW:
            preview = imgProcessor.to_8bit_preview(chroma16)
            out_path_png = Path(OUTPUT_DIR) / f"{img_name}_log_chroma_preview.png"
            cv2.imwrite(str(out_path_png), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

        logger.info(f"Saved: {out_path_tif}")

    logger.info("Batch processing complete.")


if __name__ == "__main__":
    main()
