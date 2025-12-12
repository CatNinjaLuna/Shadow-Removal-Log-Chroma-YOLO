# Spectral Ratio Estimation and Application: Illuminant-Invariant Imaging for Robotic Perception

**Author:** Carolina (Yuhan) Li  
**Advisors:** Prof. Bruce A. Maxwell, Prof. Yifan Hu  
**Affiliation:** Khoury College of Computer Sciences, Northeastern University  
**Project Type:** Research Capstone / Thesis-Oriented Project  
**Status:** Ongoing Research (Results subject to publication)

---

## Table of Contents

-  [Overview](#overview)
-  [Research Motivation](#research-motivation)
-  [Pipeline Summary](#pipeline-summary)
-  [Repository Structure](#repository-structure)
-  [Models & Checkpoints](#models--checkpoints)
-  [Training Environment](#training-environment)
-  [Experiments Conducted](#experiments-conducted)
-  [Publication & Usage Notice](#publication--usage-notice)
-  [Contact](#contact)

---

## Overview

This repository contains the codebase and experiment scaffolding for a research project investigating **shadow-invariant** and **illumination-robust object detection** using log-chromaticity representations.

The project explores whether log-space chromaticity transforms, derived from physical image formation models, can improve the robustness of modern object detectors (YOLO-based) under challenging lighting conditions compared to standard sRGB inputs.

### The work integrates:

-  Physics-inspired color representations
-  Large-scale image preprocessing pipelines
-  Controlled ablation studies on object detection models
-  High-performance cluster-based training workflows

---

## Research Motivation

Object detection models trained on standard sRGB imagery are highly sensitive to:

-  **Shadows**
-  **Illumination changes**
-  **Exposure variations**
-  **Sensor nonlinearities**

Inspired by the **Illuminant Spectral Direction (ISD)** and **log-chromaticity plane models**, this project investigates whether transforming images into log-chromaticity space can:

-  Suppress illumination intensity effects
-  Preserve object-relevant chromatic structure
-  Improve detection stability for small or shadowed objects (e.g., pedestrians, cyclists)

---

## Pipeline Summary

The full experimental pipeline is as follows:

```
RAW Images
   ↓
16-bit Linear TIFF (Debayered)
   ↓
Log-RGB / Log-Chromaticity Transform
   ↓
2D Log-Chroma Image Generation
   ↓
YOLO Training & Evaluation
   ↓
Ablation Studies & Analysis
```

### Key Variants Evaluated

| Variant              | Description                                    |
| -------------------- | ---------------------------------------------- |
| **Baseline**         | YOLO on sRGB images                            |
| **Log-Chroma**       | YOLO on log-chromaticity images                |
| **Y-Channel Fusion** | YOLO with fused luminance (Y-channel exchange) |
| **Partial-Freeze**   | Ablations freezing early backbone layers       |

### Ablation Studies

1. **Freezing first few layers** - Evaluating transfer learning effectiveness
2. **Training from scratch** - Full model training without pretraining
3. **Y-channel exchange with log-chroma** - Luminance reintegration strategy
4. **Local ISD-generated log-chroma** - Spatially-adaptive illumination estimation

---

## Repository Structure

```
Shadow-Removal-Log-Chroma-YOLO/
├── README.md
│
├── baseline/
│   ├── Data processing and baseline YOLO training scripts
│   ├── Training plots, confusion matrices, and validation results
│   └── Configuration files for baseline experiments
│
├── split_and_viz_data/
│   ├── Data conversion, visualization, and splitting scripts
│   ├── Label conversion utilities (YOLO format)
│   └── Dataset splitting and visualization tools
│
├── data_src_rename_tif_labels/
│   └── Alternative data processing and labeling scripts
│
├── raw_to_tiff_output/
│   ├── RAW to TIFF conversion pipeline
│   └── Preprocessing scripts for raw sensor data
│
├── my_training_data/
│   └── Custom training data preparation utilities
│
├── log_chroma_shadow_removal_method/
│   ├── Core shadow removal implementation using log-chroma method
│   ├── Batch processing and prediction scripts
│   └── UNet models for shadow removal
│
├── new_log_chroma_Bruce/
│   ├── Alternative log-chroma implementation with U-Net and ViT
│   ├── Models and training utilities
│   └── Experiment results and visualizations
│
├── train_freeze_first_few_layers/
│   ├── Ablation studies with layer freezing experiments
│   ├── Training scripts for frozen backbone layers
│   └── Results for freezing layers 1-3, 1-4, 1-5, 1-6
│
├── train_from_scratch_use_yaml/
│   ├── Training from scratch with YAML configuration
│   ├── Ultralytics library integration
│   ├── baseline_yaml/ - Baseline training results
│   └── log_chroma_yaml/ - Log-chroma training results
│
├── ultralytics_src_new_log_chroma/
│   ├── Modified Ultralytics source code for log-chroma
│   ├── Docker configurations
│   ├── Comprehensive documentation
│   ├── Training examples and use cases
│   ├── tests/ - Test suite
│   └── ultralytics/ - Core library with custom modifications
│
├── ultralytics_baseline/
│   ├── Baseline Ultralytics training scripts
│   ├── runs/ - Training and validation outputs
│   └── ultralytics/ - Standard Ultralytics library
│
└── y_channel_exchanged_log_chroma/
    ├── Y-channel fusion experiments
    ├── Fused log-chroma with luminance reintegration
    ├── Training scripts and configuration
    └── Validation results and performance graphs
```

> **Note:** Large datasets, model weights, and intermediate results are stored on the computing cluster and not included in this repository.

---

## Models & Checkpoints

Due to size constraints and cluster storage policies, the following artifacts are **not included** in this repository:

-  Trained YOLO model weights (`.pt`)
-  Intermediate UNet / ViT checkpoints
-  Large generated image datasets (`.png`, `.tiff`)
-  Raw sensor data

### Access to Artifacts

-  Model weights and large datasets are stored on the **Northeastern University computing cluster**
-  Selected results are shared via batched compressed archives or object storage links
-  Access can be granted upon request for academic or research purposes

### Design Rationale

This design choice ensures:

-  Compliance with storage policies
-  Clean version control
-  Reproducible experiment definitions without bloated repositories

---

## Training Environment

### Software Stack

| Component            | Version/Tool     |
| -------------------- | ---------------- |
| **Framework**        | Ultralytics YOLO |
| **Language**         | Python 3.8+      |
| **Deep Learning**    | PyTorch 2.0+     |
| **Image Processing** | OpenCV, NumPy    |
| **Computer Vision**  | torchvision      |

### Hardware & Infrastructure

-  **Hardware:** NVIDIA GPUs (cluster-based)
-  **OS:** Linux (HPC environment)
-  **Job Scheduling:** SLURM (`sbatch`)

### Cluster Job Scripts

Cluster job scripts are included for:

-  Training
-  Validation
-  Resuming interrupted runs
-  Multi-experiment ablation sweeps

<details>
<summary>Example SLURM Script</summary>
```bash
#!/bin/bash
#SBATCH --job-name=yolo_logchroma
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB

module load python/3.8
module load cuda/11.8

python training/train.py --config models/yolo_config.yaml

````

</details>

---

## Experiments Conducted

### Training Experiments

- Baseline sRGB YOLO training
- Log-chromaticity YOLO training
- Y-channel fusion experiments
- Backbone layer freezing ablations (freeze layers 1–N)

### Evaluation Metrics

- Precision-Recall curves
- mAP@0.5 analysis
- Small-object performance inspection
- Shadow region robustness testing

### Key Findings

> Results indicate that while log-chromaticity representations offer **interpretability** and **illumination suppression**, performance gains depend strongly on how **luminance information is reintegrated**.

---

## Publication & Usage Notice

**This codebase supports ongoing academic research.**

- Results are subject to revision
- Figures and metrics may appear in future publications
- Please do not redistribute or commercialize derived artifacts without permission

### Citation

If you use or reference this work, please cite appropriately once a formal publication is available.
```bibtex
@mastersthesis{li2025shadow,
  author = {Li, Carolina (Yuhan)},
  title = {Shadow-Invariant Object Detection via Log-Chromaticity Representations},
  school = {Northeastern University},
  year = {2025},
  type = {Research Capstone}
}
````

---

## Contact

**Carolina (Yuhan) Li**  
MS in Computer Science, Northeastern University  
Research Focus: Computer Vision, Robotics Perception, Illumination-Invariant Vision

For questions, collaboration, or access requests, please reach out directly.

**Email:** [li.yuhan5@northeastern.edu](mailto:li.yuhan5@northeastern.edu)  
**LinkedIn:** [linkedin.com/in/carolina-li](https://www.linkedin.com/in/carolina-li/)

---

<div align="center">

**Made with care at Khoury College of Computer Sciences**

</div>
