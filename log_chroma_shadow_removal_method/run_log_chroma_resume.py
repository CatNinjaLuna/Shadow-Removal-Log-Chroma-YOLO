"""
Batch, non-interactive script:
All 16-bit TIFFs in INPUT_DIR -> ISD prediction (UNet) -> 2D log-chromaticity images.

Resumes from a chosen filename (START_FROM_NAME), skipping earlier files.

Runs on cluster, no cv2.imshow, no GUI.

# (optional) activate virtual env
conda init bash
source ~/.bashrc
conda activate SR-shadow-removal

python run_log_chroma_resume.py

"""

import os
import cv2
import numpy as np
import logging
import torch
from pathlib import Path

from model.unet_models2 import ResNet50UNet


####################################################################################################
# CONFIG: update these if needed
####################################################################################################

INPUT_DIR = "/projects/SuperResolutionData/carolinali-shadowRemoval/raw_to_tiff_output"
OUTPUT_DIR = "/projects/SuperResolutionData/carolinali-shadowRemoval/log_chroma_output"

# Name of the file (including extension) from which to resume, inclusive
START_FROM_NAME = "day-05039.tif"

MODEL_PATH = "/projects/SuperResolutionData/carolinali-shadowRemoval/model/UNET_run_x10_01_last_model.pth"

DEVICE = "cpu"        # or "cuda" on the cluster if GPU is available
ANCHOR = 10.4         # log-space anchor for projection
SAVE_PREVIEW = True   # also save an 8-bit PNG for quick visual check
FACTOR = 32           # UNet-friendly dimension factor


####################################################################################################
# CLASSES
####################################################################################################

class imgProcessor:
    """
    Image Processor class for computing log-chromaticity using an ISD map.
    """

    def __init__(self, image: np.ndarray, sr_map: np.ndarray, filename: str = "None"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.filename = filename

        self.image = image
        if self.image is None:
            raise ValueError("Failed to set image")

        self.sr_map = sr_map
        if self.sr_map is None:
            raise ValueError("Failed to set spectral ratio map")

        self.logger.info(
            f"Image: {filename}\n"
            f"  Image loaded | Shape: {self.image.shape} | Max: {self.image.max()} | Min: {self.image.min()}\n"
            f"  Map loaded   | Shape: {self.sr_map.shape} | Max: {self.sr_map.max()} | Min: {self.sr_map.min()}"
        )

    def convert_img_to_log_space(self, linear_img: np.ndarray) -> np.ndarray:
        """
        Converts a 16-bit linear image to log-RGB space.
        Pixels with value 0 remain 0 in log space.
        """
        log_img = np.zeros_like(linear_img, dtype=np.float32)
        mask = linear_img > 0
        log_img[mask] = np.log(linear_img[mask])
        # sanity check: ln(65535) ~ 11.09
        assert np.min(log_img) >= 0 and np.max(log_img) <= 11.1
        return log_img

    def log_to_linear(self, log_img: np.ndarray) -> np.ndarray:
        """
        Converts a log-RGB image back to linear space (float32).
        """
        return np.exp(log_img).astype(np.float32)

    def compute_log_chroma(self, anchor: float = 10.4) -> np.ndarray:
        """
        Project each pixel into the plane orthogonal to the ISD for that pixel
        in log-RGB space, then convert back to linear intensities.

        Returns:
            np.ndarray: 16-bit image (H, W, 3) representing log-chromaticity in linear domain.
        """
        log_img = self.convert_img_to_log_space(self.image)

        # Shift by anchor
        shifted_log_rgb = log_img - anchor  # (H, W, 3)

        # Dot product with ISD (per pixel)
        dot_product_map = np.einsum("ijk,ijk->ij", shifted_log_rgb, self.sr_map)  # (H, W)

        # Expand for broadcasting
        dot_product_reshaped = dot_product_map[:, :, np.newaxis]  # (H, W, 1)

        # Projection of shifted_log_rgb onto ISD
        projection = dot_product_reshaped * self.sr_map  # (H, W, 3)

        # Remove the projected component: now in 2D plane orthogonal to ISD
        projected_rgb = shifted_log_rgb - projection

        # Shift back by anchor
        projected_rgb += anchor

        # Back to linear
        linear_chroma = self.log_to_linear(projected_rgb)

        # Clip & cast to 16-bit
        linear_chroma = np.clip(linear_chroma, 0, 65535).astype(np.uint16)
        return linear_chroma

    @staticmethod
    def to_8bit_preview(img16: np.ndarray) -> np.ndarray:
        """
        Simple 16-bit -> 8-bit for preview.
        """
        img16 = img16.astype(np.float32)
        img16 = np.clip(img16, 0, 65535)
        img8 = (img16 / 256.0).astype(np.uint8)
        return img8


class ISDMapEstimator:
    """
    Wraps the UNet model to predict ISD maps from 16-bit linear RGB images.
    """

    def __init__(self, model: object, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Device = {self.device}")
        self.sr_map = None

        self.model = model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {model_path}: {e}")

        if "model_state_dict" not in checkpoint:
            raise KeyError(f"'model_state_dict' key not found in checkpoint at: {model_path}")

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load model state_dict: {e}")

        self.model.to(self.device)
        self.model.eval()

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert 16-bit linear RGB to normalized log-space tensor (1, 3, H, W).
        """
        log_img = np.zeros_like(image, dtype=np.float32)
        mask = image > 0
        log_img[mask] = np.log(image[mask])
        assert np.min(log_img) >= 0 and np.max(log_img) <= 11.1

        log_img = log_img / 11.1  # normalize to [0, 1]
        input_tensor = (
            torch.from_numpy(log_img.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )
        self.logger.info(
            f"Input Tensor | Shape: {input_tensor.shape} | Dtype: {input_tensor.dtype}"
        )
        return input_tensor

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Runs inference on the input image and returns a normalized ISD map (H, W, 3).
        """
        input_tensor = self._preprocess_image(image)

        with torch.no_grad():
            output = self.model(input_tensor)

        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        norm = np.linalg.norm(output_np, axis=2, keepdims=True).astype(np.float32)
        norm[norm == 0] = 1.0
        self.sr_map = output_np / norm

        self.logger.info(
            f"ISD Map | Shape: {self.sr_map.shape} | Dtype: {self.sr_map.dtype}"
        )
        return self.sr_map


####################################################################################################
# HELPERS
####################################################################################################

def crop_to_even_dims(image: np.ndarray) -> np.ndarray:
    """
    Crop image so H and W are even.
    """
    h, w = image.shape[:2]
    new_h = h - (h % 2)
    new_w = w - (w % 2)
    return image[:new_h, :new_w]


def center_crop_to_divisible(image: np.ndarray, factor: int = 32) -> np.ndarray:
    """
    Center crop image so H and W are divisible by 'factor' (for UNet).
    """
    h, w = image.shape[:2]
    new_h = (h // factor) * factor
    new_w = (w // factor) * factor
    start_h = (h - new_h) // 2
    start_w = (w - new_w) // 2
    return image[start_h:start_h + new_h, start_w:start_w + new_w]


####################################################################################################
# MAIN
####################################################################################################

def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    logger = logging.getLogger("main")

    input_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Model checkpoint: {MODEL_PATH}")
    logger.info(f"Resuming from (inclusive): {START_FROM_NAME}")

    # -----------------------------------------------------------------------------------------
    # Collect all TIFFs & sort
    # -----------------------------------------------------------------------------------------
    tif_files = sorted(input_dir.glob("day-*.tif"))
    if not tif_files:
        logger.error("No TIFF files found matching 'day-*.tif' in INPUT_DIR.")
        return

    # Flag to control resume behavior
    resume = False
    for img_path in tif_files:
        img_name = img_path.name  # includes extension
        stem = img_path.stem

        # Skip until we reach START_FROM_NAME
        if not resume:
            if img_name == START_FROM_NAME:
                resume = True
            else:
                logger.info(f"Skipping {img_name} (before START_FROM_NAME)")
                continue

        logger.info(f"Processing image: {img_path}")

        # -------------------------------------------------------------------------------------
        # Load 16-bit TIFF
        # -------------------------------------------------------------------------------------
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            logger.error(f"Failed to load image from {img_path}, skipping.")
            continue

        if image.ndim == 2:
            logger.error(f"Expected 3-channel RGB image, got single channel at {img_path}, skipping.")
            continue

        # OpenCV loads as BGR; convert to RGB to stay consistent
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crop to be UNet-friendly
        image = crop_to_even_dims(image)
        image = center_crop_to_divisible(image, factor=FACTOR)
        logger.info(f"Image | Shape: {image.shape} | Dtype: {image.dtype}")

        # -------------------------------------------------------------------------------------
        # Model + ISD prediction (model created once outside loop)
        # -------------------------------------------------------------------------------------
        # To avoid re-loading the model every image, we create the model/estimator once
        # outside the loop; so we need to break out of this loop after setting them up
        # if we haven't yet. But simpler: we instantiate them once before the loop.
        # => we'll move model init out of the loop and just call estimator.predict here.

        # We'll just break here & restructure below if necessary.
        # But to keep your code clean, we actually instantiate before loop (see below).
        # For now, assume 'estimator' is already defined.
        # -------------------------------------------------------------------------------------

        # Predict ISD
        sr_map = estimator.predict(image)

        # -------------------------------------------------------------------------------------
        # Log-chroma computation
        # -------------------------------------------------------------------------------------
        processor = imgProcessor(image, sr_map, filename=stem)
        chroma16 = processor.compute_log_chroma(anchor=ANCHOR)

        # Save 16-bit log-chroma TIFF
        out_tif_path = out_dir / f"{stem}_log_chroma.tif"
        chroma16_bgr = cv2.cvtColor(chroma16, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(out_tif_path), chroma16_bgr)
        if not ok:
            logger.error(f"Failed to save 16-bit log-chroma TIFF to {out_tif_path}")
        else:
            logger.info(f"Saved 16-bit log-chroma TIFF to {out_tif_path}")

        # Optional 8-bit preview
        if SAVE_PREVIEW:
            chroma8 = imgProcessor.to_8bit_preview(chroma16)
            chroma8_bgr = cv2.cvtColor(chroma8, cv2.COLOR_RGB2BGR)
            out_png_path = out_dir / f"{stem}_log_chroma_preview.png"
            ok2 = cv2.imwrite(str(out_png_path), chroma8_bgr)
            if not ok2:
                logger.error(f"Failed to save 8-bit preview to {out_png_path}")
            else:
                logger.info(f"Saved 8-bit preview PNG to {out_png_path}")


if __name__ == "__main__":
    # Instantiate the model & estimator ONCE, then call main()
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    logger_root = logging.getLogger("root")

    logger_root.info("Initializing model + estimator...")

    model = ResNet50UNet(
        in_channels=3,
        out_channels=3,
        pretrained=False,
        checkpoint=None,
        se_block=True,
    )

    estimator = ISDMapEstimator(
        model=model,
        model_path=MODEL_PATH,
        device=DEVICE,
    )

    main()
