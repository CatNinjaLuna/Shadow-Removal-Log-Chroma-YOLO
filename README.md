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
shadow-invariant-detection/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── preprocessing/
│   │   ├── raw_to_tiff.py
│   │   ├── log_chroma_transform.py
│   │   └── isd_estimation.py
│   └── loaders/
│       └── dataset.py
├── models/
│   ├── yolo_config.yaml
│   └── custom_layers.py
├── training/
│   ├── train.py
│   ├── validate.py
│   └── slurm_scripts/
│       ├── train_baseline.sh
│       ├── train_logchroma.sh
│       └── ablation_sweep.sh
├── evaluation/
│   ├── metrics.py
│   ├── visualize.py
│   └── analyze_results.py
└── docs/
    ├── methodology.md
    └── results.md
```

> **Note:** Directory structure is subject to updates as research progresses.

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
