#!/usr/bin/env python3
# This file is for testing models for the spectral ratio project
# Written by Michael Massone
# Updated by Bruce A. Maxwell 11/2025
# Modified to run one or more images from the command line and to generate log-space chromaticity visualizations
# Further modified (Carolina cluster version): disable GUI, save outputs to disk, and support batch directory processing.

#############################################################################################
# PACKAGES
import os
import sys
import logging
import argparse
from pathlib import Path

import torch
import cv2
import numpy as np
# from tqdm import tqdm
import torch.nn                 as nn
import torch.nn.functional      as F
import torchvision.transforms   as transforms

#############################################################################################
# MODULES

from srvizlib              import BasicDataset_16BitTIFF, ToTensor, ToLogRGB, CenterCropToSize, ProjectLogChrom
from models_unet           import ResNet50UNet
from models_vit            import ViT_Patch2Patch_ver2
# need to add the mambavision models

#############################################################################################
# FUNCTIONS

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Use a trained model to estimate the ISD map for one or more images"
    )

    parser.add_argument(
        "-m",
        "--model",
        nargs=1,
        type=str,
        default="unet",
        choices=("unet", "vit", "mamba"),
        help="Model to use for estimation: {unet (default), vit, mamba}",
    )

    parser.add_argument(
        "-r",
        "--representation",
        nargs=1,
        type=str,
        default="log",
        choices=("linear", "log"),
        help="Input space to use for the image: {log default), linear}",
    )

    parser.add_argument(
        "images",
        nargs="+",
        type=str,
        help="<image 1> ... or <directory 1> ...",
    )

    return parser.parse_args()


def expand_inputs(image_args):
    """
    支持：
    - 直接传入若干 tiff 文件路径
    - 传入一个或多个目录路径，自动在目录中查找 *.tif / *.tiff
    """
    all_files = []
    for p in image_args:
        p_path = Path(p)
        if p_path.is_dir():
            # 扫描该目录下所有 tif / tiff
            tifs = sorted(list(p_path.glob("*.tif"))) + sorted(list(p_path.glob("*.tiff")))
            all_files.extend([str(x) for x in tifs])
        else:
            # 直接是文件
            all_files.append(str(p_path))

    # 去重并保持顺序
    seen = set()
    unique_files = []
    for f in all_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return unique_files

#############################################################################################
# MAIN

def main():

    # parse the arguments
    args = parse_args()

    # args.model can be a list if passed with nargs=1
    test_name = args.model[0] if isinstance(args.model, (list, tuple)) else args.model

    # get the model path
    model_weight_path = f"weights/{test_name}/model_states/best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # useful constants
    eps = 1e-8
    size = 512
    print("Setting crop size to 512 x 512")

    # ------- 输出目录（绝对路径）-------
    base_output_dir = Path("/projects/SuperResolutionData/carolinali-shadowRemoval/new_log_chroma_Bruce/log_chroma_outputs")
    png_output_dir = base_output_dir / "png"
    tiff_output_dir = base_output_dir / "tiff"
    png_output_dir.mkdir(parents=True, exist_ok=True)
    tiff_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"PNG  输出目录: {png_output_dir.resolve()}")
    print(f"TIFF 输出目录: {tiff_output_dir.resolve()}")

    try:
        # 展开输入（既支持传目录也支持传单个文件）
        input_files = expand_inputs(args.images)
        if not input_files:
            print("No input .tif/.tiff files found. Please check the path.")
            return

        print("Preloading images and ISD/segment maps into memory...")
        print("Input files:")
        for f in input_files:
            print("  ", f)

        # Set transforms
        transform_list = [
            ToTensor(),
            # CenterCropToSize(target_size=(size, size)),
            ToLogRGB(),
        ]
        composed_transforms = transforms.Compose(transform_list)

        # Init Model
        print(f"Initializing model {test_name}")
        if test_name == "unet":
            model = ResNet50UNet(
                in_channels=3,
                out_channels=3,
                pretrained=False,
                se_block=True,
                dropout=0.0,
            ).to(device)
        else:
            model = ViT_Patch2Patch_ver2(
                img_size=512,
                patch_size=16,
                in_ch=3,
                out_ch=3,
                embed_dim=768,
                depth=6,
                heads=8,
                dropout=0.1,
            ).to(device)

        # Read the images from the command line and do the necessary conversions
        test_ds = BasicDataset_16BitTIFF(
            input_files,
            ds_device=device,
            image_transforms=composed_transforms,
        )

        # make a data loader
        test_loader = torch.utils.data.DataLoader(test_ds)

        # load model weights
        model_weights = torch.load(model_weight_path, map_location=device)
        model.load_state_dict(model_weights["model_state_dict"])

        # put the model in evaluation mode
        model.eval()

        results = []

        with torch.no_grad():
            # pbar = tqdm(test_loader, desc=f"Testing:")
            # for i, batch in enumerate(pbar):

            for i, batch in enumerate(test_loader):
                print(f"\nprocessing index {i}")

                # get the data and image name
                images = batch["image"].to(device)
                image_name = batch.get("img_name", None)

                # call the model
                prediction = model(images)

                # normalize the output (per-pixel direction)
                norm_prediction = torch.norm(prediction, dim=1, keepdim=True)
                predicted_norm = torch.where(
                    norm_prediction != 0, prediction / norm_prediction, prediction
                )

                # compute the global ISD
                isd_mean = predicted_norm.mean(dim=(2, 3))
                isd_norm = F.normalize(isd_mean, dim=1, eps=eps)

                # store results
                results.append(
                    {
                        "index": i,
                        "image": images[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .transpose(1, 2, 0),
                        "pred": predicted_norm[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .transpose(1, 2, 0),
                        "gbl_isd": isd_norm.detach().cpu().numpy()[0],
                        "filename": image_name,
                    }
                )

        # Process and save outputs
        for item in results:
            raw_name = item["filename"]

            # img_name might be a list/tuple/ndarray from the dataset → normalize it
            if isinstance(raw_name, (list, tuple, np.ndarray)):
                fname = str(raw_name[0]) if len(raw_name) > 0 else "unknown"
            else:
                fname = str(raw_name) if raw_name is not None else "unknown"

            print("\n", fname)
            print(
                "isd_norm: %.2f %.2f %.2f"
                % (
                    item["gbl_isd"][0],
                    item["gbl_isd"][1],
                    item["gbl_isd"][2],
                )
            )

            # project each image onto the log chrom plane and generate a visualization
            print("computing projection")
            logchrom_src, logchrom_viz = ProjectLogChrom(
                item["image"], isd_vector=item["gbl_isd"]
            )
            print("computing log chrom")

            # ---- CONSISTENT NAMING: use *input filename* stem ----
            input_path = Path(fname)
            stem = input_path.stem

            # fallback if stem is empty
            if stem.strip() == "":
                stem = f"img_{item['index']:05d}"

            # 1) Save an 8-bit visualization PNG
            viz_path = png_output_dir / f"{stem}.png"
            cv2.imwrite(str(viz_path), logchrom_viz)

            # 2) Save a 16-bit TIFF log-chroma image derived from logchrom_src
            log_min = np.min(logchrom_src)
            log_max = np.max(logchrom_src)
            if log_max > log_min:
                log_norm = (logchrom_src - log_min) / (log_max - log_min)
            else:
                log_norm = np.zeros_like(logchrom_src, dtype=np.float32)

            log_uint16 = (log_norm * 65535.0).astype(np.uint16)
            tiff_path = tiff_output_dir / f"{stem}.tif"
            cv2.imwrite(str(tiff_path), log_uint16)

            print(f"Saved PNG : {viz_path}")
            print(f"Saved TIFF: {tiff_path}")

    except Exception as e:
        print(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()
