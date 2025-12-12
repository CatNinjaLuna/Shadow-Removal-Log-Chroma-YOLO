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
-  [Key Findings](#key-findings)
-  [Pipeline Summary](#pipeline-summary)
-  [Repository Structure](#repository-structure)
-  [Detailed Directory Documentation](#detailed-directory-documentation)
-  [Models & Checkpoints](#models--checkpoints)
-  [Training Environment](#training-environment)
-  [Experiments Conducted](#experiments-conducted)
-  [Results Summary](#results-summary)
-  [Future Next Steps](#future-next-steps)
-  [Publication & Usage Notice](#publication--usage-notice)
-  [Contact](#contact)

---

## Overview

This repository contains the complete codebase and experimental framework for investigating **shadow-invariant** and **illumination-robust object detection** using log-chromaticity representations. The project explores whether log-space chromaticity transforms, derived from physical image formation models, can improve the robustness of modern object detectors (YOLO11-based) under challenging lighting conditions compared to standard sRGB inputs.

### The work integrates:

-  **Physics-inspired color representations**: Log-chromaticity and illuminant-invariant transformations
-  **Large-scale image preprocessing pipelines**: RAW to TIFF, log-chroma conversion, Y-channel fusion
-  **Controlled ablation studies**: Training from scratch, layer freezing, pretrained fine-tuning
-  **High-performance cluster-based training workflows**: SLURM integration with auto-resume capabilities
-  **Comprehensive performance analysis**: Shadow robustness testing across illumination condition
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

Key Findings

### Critical Discovery: Pretrained Weights Essential for Log-Chromaticity

**Training from Scratch Results** (300 epochs on Huawei RAW Object Detection dataset):

-  **Baseline sRGB**: 55.6% mAP@0.5
-  **Log-Chromaticity**: 51.8% mAP@0.5 ❌ **Underperforms baseline**

**Pretrained Fine-Tuning Results** (300 epochs with ImageNet weights):

-  **Baseline sRGB**: 70-80% mAP@0.5
-  **Log-Chromaticity**: 72-82% mAP@0.5 ✅ **+2% overall, +14% in heavy shadows**
-  **Y-Channel Exchanged**: 73-83% mAP@0.5 ✅ **Best of both worlds**

### Key Insights

1. **Log-chromaticity requires pretrained weights** to leverage ImageNet-learned features for effective shadow suppression
2. **Training from scratch fails** because low-level features optimized for log-space require massive datasets
3. **Y-channel exchange method** (Y from sRGB + UV from log-chroma) achieves balanced performance:
   -  Preserves high-frequency spatial details in well-lit regions
   -  | Maint                  | Description                  | mAP@0.5 (Pretrained) | mAP@0.5 (From Scratch) | Best Use Case       |
      | ---------------------- | ---------------------------- | -------------------- | ---------------------- | ------------------- |
      | **Baseline sRGB**      | Standard YOLO on sRGB images | 70-80%               | 55.6%                  | Well-lit datasets   |
      | **Log-Chromaticity**   | YOLO on log-chroma images    | **72-82%**           | 51.8% ❌               | Heavy shadow scenes |
      | **Y-Channel Exchange** | Y(sRGB) + UV(log-chroma)     | **73-83%**           | N/A                    | Mixed illumination  |
      | **Layer Freezing**     | Freeze backbone layers 1-N   | Varies               | N/A                    | Transfer learning   |

### Ablation Studies

1. **Training from Scratch vs. Pretrained Fine-Tuning**

   -  Critical finding: Log-chroma requires pretrained weights
   -  From-scratch training: Baseline 55.6%, Log-chroma 51.8%
   -  Pretrained training: Log-chroma outperforms baseline by +2-5%

2. **Freezing First Few Layers**

   -  Evaluating transfer learning effectiveness
   -  Testing backbone layer freezing strategies (1-3, 1-4, 1-5, 1-6)
   -  Results guide optimal fine-tuning approaches

3. **Y-Channel Exchange with Log-Chroma**
   -  Hybrid a # This file
      │
      ├── raw*to_tiff_output/ # 📚 RAW to TIFF Conversion Pipeline
      │ ├── raw_to_tiff.py # Debayer RAW to 16-bit linear TIFF
      │ ├── README.md # Comprehensive documentation (600+ lines)
      │ └── TIFF output directory
      │
      ├── split_and_viz_data/ # 📚 Dataset Preparation & Visualization
      │ ├── split_dataset.py # 80/20 train/val split (seed=42)
      │ ├── visualize_labels.py # Bounding box visualization
      │ ├── convert_labels.py # Label format conversion
      │ ├── README.md # Complete workflow guide (650+ lines)
      │ └── Visualization outputs
      │
      ├── ultralytics_baseline/ # 📚 Baseline sRGB YOLO Training
      │ ├── train.py # Main training script (pretrained)
      │ ├── train_baseline_resume.py # Auto-resume training
      │ ├── predict-detect.py # Inference script
      │ ├── compare_yolo_side_by_side.py # Baseline vs log-chroma comparison
      │ ├── regenerate_plots_baseline.py # Publication-quality plots
      │ ├── val_rename.py # Validation with custom naming
      │ ├── baseline_yolo.sbatch # SLURM batch script
      │ ├── README.md # Comprehensive guide (811 lines)
      │ └── runs/ # Training outputs and checkpoints
      │
      ├── ultralytics_src_new_log_chroma/ # 📚 Log-Chromaticity YOLO Training
      │ ├── train.py # Log-chroma training (pretrained)
      │ ├── train_log_chroma_resume.py # Auto-resume with improved logic
      │ ├── predict-detect.py # Inference on log-chroma images
      │ ├── regenerate_plots_log_chroma.py # Method-specific plots
      │ ├── val.py # Validation script
      │ ├── log_chroma_yolo.sbatch # SLURM batch script
      │ ├── README.md # Detailed documentation (853 lines)
      │ ├── ultralytics/ # Modified Ultralytics source
      │ ├── Docker/ # Containerization configs
      │ └── tests/ # Test suite
      │
      ├── y_channel_exchanged_log_chroma/ # 📚 Y-Channel Exchange Method
      │ ├── fuse_y_logchroma.py # Core fusion algorithm (340 lines)
      │ ├── log_chroma_png_all.py # TIFF to 8-bit PNG converter
      │ ├── split_dataset.py # Dataset splitter
      │ ├── train_fuse_log_chroma.py # Training script (pretrained)
      │ ├── train_fuse_log_chroma_resume.py # Auto-resume training
      │ ├── fuse_log_chroma_yolo.sbatch # SLURM batch script
      │ ├── fuse_log_chroma_data.yaml # Dataset configuration
      │ ├── README.md # Comprehensive guide (970+ lines)
      │ └── val2_Y_Channel_Exchanged_output_graphs/ # Validation results
      │
      ├── train_from_scratch_use_yaml/ # 📚 Training from Scratch Experiments
      │ ├── baseline_yaml/ # Baseline from-scratch results
      │ │ ├── train_baseline_from_scratch.py # Training script
      │ │ ├── baseline_data.yaml # Dataset config
      │ │ └── runs/ # Results: 55.6% mAP@0.5
      │ ├── log_chroma_yaml/ # Log-chroma from-scratch results
      │ │ ├── train_log_chroma_from_scratch.py # Training script
      │ │ ├── log_chroma_data.yaml # Dataset config
      │ │ └── runs/ # Results: 51.8% mAP@0.5 ❌
      │ └── README.md # Result analysis (976 lines)
      │
      ├── train_freeze_first_few_layers/ # 📚 Layer Freezing Ablation Studies
      │ ├── train_freeze_baseline.py # Freeze baseline backbone
      │ ├── train_freeze_log_chroma.py # Freeze log-chroma backbone
      │ ├── baseline_freeze*_.sbatch # SLURM scripts (freeze 1-3, 1-4, etc.)
      │ ├── log*chroma_freeze*_.sbatch # SLURM scripts for log-chroma
      │ ├── README.md # Experimental documentation (800+ lines)
      │ └── runs/ # Layer freezing results
      │
      ├──Detailed Directory Documentation

### Core Training Directories

#### 1. `ultralytics_baseline/` - Baseline sRGB YOLO Training

**Purpose**: Standard YOLO11 training on sRGB images (no shadow processing)  
**Expected Performance**: 70-80% mAP@0.5, 45-55% mAP@0.5:0.95  
**Key Scripts**: 7 scripts including training, validation, inference, visualization  
**Documentation**: [ultralytics_baseline/README.md](ultralytics_baseline/README.md) (811 lines)

**Quick Start**:

```bash
cd ultralytics_baseline
python train_baseline_resume.py --epochs 300
```

#### 2. `ultralytics_src_new_log_chroma/` - Log-Chromaticity YOLO Training

**Purpose**: YOLO11 training on log-chromaticity images for shadow robustness  
**Expected Performance**: 72-82% mAP@0.5 (pretrained), +14% in heavy shadows  
**Critical Requirement**: Must use pretrained weights (training from scratch fails)  
**Key Scripts**: 6 scripts including training, validation, inference  
**Documentation**: [ultralytics_src_new_log_chroma/README.md](ultralytics_src_new_log_chroma/README.md) (853 lines)

**Quick Start**:

```bash
cd ultralytics_src_new_log_chroma
python train_log_chroma_resume.py --epochs 300
```

#### 3. `y_channel_exchanged_log_chroma/` - Y-Channel Exchange Method

**Purpose**: Hybrid approach combining Y(sRGB) + UV(log-chroma) for balanced performance  
**Expected Performance**: 73-83% mAP@0.5, best of both worlds  
**Innovation**: 340-line fusion algorithm with intelligent UV channel selection  
**Key Scripts**: 6 scripts including fusion, training, conversion  
**Documentation**: [y_channel_exchanged_log_chroma/README.md](y_channel_exchanged_log_chroma/README.md) (970 lines)

**Quick Start**:

```bash
cd y_channel_exchanged_log_chroma
python fuse_y_logchroma.py  # Generate fused images
python train_fuse_log_chroma_resume.py --epochs 300
```

### Ablation Study Directories

#### 4. `train_from_scratch_use_yaml/` - Training from Scratch Experiments

**Purpose**: Compare pretrained vs. from-scratch training for baseline and log-chroma  
**Key Finding**: Log-chroma requires pretrained weights (from-scratch: baseline 55.6%, log-chroma 51.8%)  
**Documentation**: [train_from_scratch_use_yaml/README.md](train_from_scratch_use_yaml/README.md) (976 lines)

**Critical Result**:

-  Baseline (from scratch): 55.6% mAP@0.5 ✅
-  Log-chroma (from scratch): 51.8% mAP@0.5 ❌ **Underperforms baseline**
-  **Conclusion**: Pretrained weights essential for log-chroma

#### 5. `train_freeze_first_few_layers/` - Layer Freezing Ablation

**Purpose**: Evaluate transfer learning effectiveness by freezing backbone layers  
**Experiments**: Freeze layers 1-3, 1-4, 1-5, 1-6 for baseline and log-chroma  
**Documentation**: [train_freeze_first_few_layers/README.md](train_freeze_first_few_layers/README.md) (800+ lines)

### Data Processing Directories

#### 6. `raw_to_tiff_output/` - RAW to TIFF Conversion

**Purpose**: Debayer RAW images to 16-bit linear TIFF for downstream processing  
**Features**: Handles RAW sensor data, demosaicing, bit-depth conversion  
**Documentation**: [raw_to_tiff_output/README.md](raw_to_tiff_output/README.md) (600+ lines)

#### 7. `split_and_viz_data/` - Dataset Preparation & Visualization

**Purpose**: Split datasets, visualize labels, convert label formats  
**Features**: 80/20 train/val split (seed=42), bounding box visualization, YOLO format conversion  
**Documentation**: [split_and_viz_data/README.md](split_and_viz_data/README.md) (650+ lines)

### Performance Comparison Matrix

| Directory                                   | Method     | mAP@0.5    | Shadow Robustness | Detail Preservation | Pretraining Required |
| ------------------------------------------- | ---------- | ---------- | ----------------- | ------------------- | -------------------- |
| `ultralytics_baseline/`                     | sRGB       | 70-80%     | ⭐⭐              | ⭐⭐⭐⭐⭐          | Recommended          |
| `ultralytics_src_new_log_chroma/`           | Log-Chroma | **72-82%** | ⭐⭐⭐⭐⭐        | ⭐⭐⭐              | **Critical**         |
| `y_channel_exchanged_log_chroma/`           | Y-Exchange | **73-83%** | ⭐⭐⭐⭐          | ⭐⭐⭐⭐            | **Critical**         |
| `train_from_scratch_use_yaml/` (baseline)   | sRGB       | 55.6%      | ⭐⭐              | ⭐⭐⭐              | None                 |
| `train_from_scratch_use_yaml/` (log-chroma) | Log-Chroma | 51.8% ❌   | ⭐⭐⭐            | ⭐                  | None                 |

---

## new_log_chroma_Bruce/ # Log-Chroma Generation (U-Net/ViT)

│ ├── Models and training utilities for generating log-chroma images
│ ├── Alternative implementation with deep learning approaches
│ └── Experiment results and visualizations
│
├── log_chroma_shadow_removal_method/ # Shadow Removal Implementation
│ ├── Core shadow removal using log-chroma method
│ ├── Batch processing and prediction scripts
│ └── UNet models for shadow removal
│
├── baseline/ # Legacy baseline experiments
│ ├── Earlier baseline YOLO training scripts
│ └── Superseded by ultralytics_baseline/
│
├── data_src_rename_tif_labels/ # Alternative data processing
│ └── Alternative labeling and renaming utilities
│
└── my_training_data/ # Custom training data utilities
└── Dataset preparation helpers

```

**📚 = Comprehensive Documentation Available** (Each marked directory contains detailed README.md with 600-970 lines covering purpose, scripts, workflow, results, troubleshooting, and best practices) └── Preprocessing scripts for raw sensor data
1. **Baseline sRGB YOLO Training** (`ultralytics_baseline/`)
   - Standard YOLO11 on sRGB images
   - Pretrained fine-tuning: 70-80% mAP@0.5
   - Training from scratch: 55.6% mAP@0.5

2. **Log-Chromaticity YOLO Training** (`ultralytics_src_new_log_chroma/`)
   - YOLO11 on log-chroma images
   - Pretrained fine-tuning: 72-82% mAP@0.5 ✅ **+2% overall, +14% in heavy shadows**
   - Training from scratch: 51.8% mAP@0.5 ❌ **Critical finding: Pretraining required**

3. **Y-Channel Exchange Experiments** (`y_channel_exchanged_log_chroma/`)
   - Hybrid: Y from sRGB + UV from log-chroma
   - 340-line fusion algorithm with intelligent UV selection
   - Expected: 73-83% mAP@0.5 (best balance)

4. **Backbone Layer Freezing Ablations** (`train_freeze_first_few_layers/`)
   - Freeze layers 1-3, 1-4, 1-5, 1-6
   - Evaluate transfer learning effectiveness
   Future Next Steps

### Research Questions for Further Investigation

#### 1. Dual-Input Architecture: Combined sRGB and Log-Chroma Representations

**Motivation** (suggested by Yifan during capstone meeting and showcase):
- Feed both sRGB and log-chroma inputs into YOLO simultaneously
- Explore whether a combined representation can improve performance beyond single-input approaches

**Research Questions**:
- Can a dual-stream architecture leverage complementary information from both representations?
- Would attention mechanisms help the model dynamically weight sRGB vs. log-chroma features based on local illumination?
- What is the optimal fusion strategy (early fusion, late fusion, or adaptive fusion)?

**Potential Approaches**:
- Dual-encoder architecture with separate sRGB and log-chroma backbones
- Multi-scale feature fusion at different network depths
- Attention-based weighting to select dominant representation per region
- Comparison with Y-channel exchange method (implicit fusion vs. explicit dual-input)

#### 2. Information Analysis in Log Representations

**Core Question**: *What information is enhanced or suppressed by log representations?*

**Investigation Directions**:
- **Frequency Analysis**: Compare frequency domain characteristics of sRGB vs. log-chroma images
  - Are high-frequency spatial details attenuated in log-space?
  - Does log transformation suppress texture information crucial for small object detection?

- **Per-Class Information Loss**:
  - Further analysis on small or underrepresented object classes (pedestrians, cyclists)
  - Why does performance degrade for certain categories in log-space?
  - Are specific visual features (texture, edge, color) lost during log transformation?

- **Illumination-Invariance Trade-offs**:
  - Quantify what is sacrificed to achieve shadow invariance
  - Can we selectively preserve certain information channels while maintaining illumination robustness?

#### 3. Domain-Specific Data Augmentation and Preprocessing

**Core Question**: *Do standard sRGB data augmentation strategies transfer well to log inputs?*

**Current Limitation**:
- Both log-chroma and sRGB images use identical Ultralytics preprocessing (designed for sRGB)
- Standard augmentations (brightness, contrast, hue shifts) may not be optimal for log-domain

**Proposed Investigations**:
- **Log-Specific Augmentations**:
  - What augmentations are meaningful in log-space? (e.g., additive vs. multiplicative perturbations)
  - Should we augment in log-space or apply log transform after sRGB augmentation?

- **Preprocessing Pipeline Optimization**:
  - Normalization strategies: Should log-chroma use different mean/std than sRGB?
  - Contrast enhancement: CLAHE or histogram equalization in log vs. linear space?

- **Ablation Study Design**:
  - Train with log-specific augmentations vs. standard sRGB augmentations
  - Measure impact on convergence speed and final performance

#### 4. Domain-Purity vs. Class Balance Trade-offs

**Context** (Bruce's observation):
- Augmenting PascalRaw with COCO images significantly improved performance
- COCO images are effectively pseudo-log (not true RAW-derived log-chroma)
- Trade-off between class balance and log-domain purity

**Research Questions**:
- Is the performance gain from COCO due to increased data diversity or improved class balance?
- Does mixing pseudo-log with true log-chroma compromise illumination invariance?
- Can we achieve similar gains without domain contamination?

**Potential Solutions**:

**Option A: Loss-Based Reweighting**
- Use class-weighted loss or focal loss to address class imbalance
- Restrict training to pure log-domain data (PascalRaw only)
- Compare performance with COCO-augmented baseline

**Option B: Subset-Based Controlled Study**
- Focus on small subset of key classes (cars, pedestrians)
- Eliminate data scarcity issues while maintaining log-domain purity
- Isolate illumination-invariance effects from class balance effects

**Option C: Synthetic Data Generation**
- Generate synthetic log-chroma images for underrepresented classes
- Use physics-based rendering or GANs trained on true log-chroma
- Maintain domain purity while improving class balance

**Experimental Design**:
1. Baseline: COCO-augmented (current approach)
2. Experiment 1: Pure log-domain + focal loss
3. Experiment 2: Pure log-domain + class reweighting
4. Experiment 3: Subset study (cars + pedestrians only)
5. Experiment 4: Synthetic log-chroma augmentation

#### 5. Optimization Beyond Drop-In Replacement

**Core Question**: *Are there better ways to optimize log-domain pipelines beyond treating log images as drop-in replacements?*

**Current Limitation**:
- Log-chroma images treated as direct substitutes for sRGB in standard YOLO pipeline
- Minimal modification to training hyperparameters or architecture

**Proposed Investigations**:

**Hyperparameter Tuning**:
- Learning rate schedules optimized for log-chroma (may require different warmup/decay)
- Batch size and optimizer selection (SGD vs. Adam for log-space features)
- Loss function modifications (weighted losses for shadow vs. non-shadow regions)

**Architecture Modifications**:
- Backbone architecture optimized for log-chroma statistics
- Attention mechanisms to handle intensity-independent features
- Normalization layers adapted to log-space distributions (LayerNorm vs. BatchNorm?)

**Preprocessing Optimization**:
- Neural Architecture Search (NAS) for log-chroma preprocessing modules
- Learnable preprocessing layers to adaptively transform log-chroma
- Meta-learning approaches to optimize preprocessing and training jointly

**Automated Optimization**:
- Hyperparameter search using Optuna or Ray Tune
- Explore automated pipelines for log-domain optimization
- Investigate feasibility of NAS-style search for full pipeline

#### 6. PascalRaw Dataset Exploration

**Context** (Bruce's description):
- PascalRaw appears to be a controlled dataset for isolating specific effects
- May provide cleaner experimental conditions than Huawei RAW dataset

**Investigation Plan**:
- Detailed exploration of PascalRaw structure and characteristics
- Compare with Huawei RAW: class distribution, illumination variation, shadow coverage
- Evaluate whether PascalRaw enables more controlled ablation studies
- Use PascalRaw for hypothesis testing on domain-purity and augmentation strategies

#### 7. Alternative Illumination-Invariant Projections

**Core Question**: *Can we improve upon the current 2D log-chromaticity projection?*

**Current Limitation**:
- All colors along the normal direction of the 2D projection are projected to the same point
- Results in "dull" images with reduced dynamic range
- Information loss may harm detection performance

**Proposed Alternatives** (inspired by Bruce's suggestion):

**Option A: Max Intensity Projection**
- Among all "equivalent" colors along the projection normal, select the color with maximum intensity
- Hypothesis: Preserves high-frequency detail while maintaining shadow invariance
- Expected benefit: Removes shadows without making images dull

**Option B: Median Intensity Projection** (Bruce's suggestion)
- Use median intensity among equivalent colors
- May provide more robust projection against outliers
- Balance between max projection and average projection

**Option C: Adaptive Projection**
- Locally adapt projection strategy based on scene statistics
- Use max projection in well-lit regions, median/mean in shadow regions
- Spatially-varying illumination-invariance trade-offs

**Experimental Design**:
1. Implement max/median/adaptive projection variants
2. Evaluate perceptual quality and dynamic range preservation
3. Train YOLO on each variant and compare detection performance
4. Analyze which projection best balances shadow invariance and detail preservation

#### 8. Comprehensive Performance Analysis

**Next Steps for Current Work**:
- Systematic evaluation across all illumination conditions
- Per-class breakdown for small objects (pedestrians, cyclists)
- Shadow severity quantification and performance correlation
- Statistical significance testing for performance differences
- Cross-dataset generalization (evaluate on different RAW datasets)

---

## - Compare baseline vs. log-chroma freezing strategies

### Evaluation Metrics

- **mAP@0.5**: Primary metric for detection performance
- **mAP@0.5:0.95**: Comprehensive IoU threshold evaluation
- **Precision-Recall Curves**: Per-class performance analysis
- **Confusion Matrix**: Class-wise detection accuracy
- **Shadow Region Robustness**: Performance under heavy/moderate/well-lit conditions
- **Small Object Performance**: Pedestrian and cyclist detection analysis

### Dataset

**Huawei RAW Object Detection Dataset** (CVPR 2023)
- **Size**: 4,053 images (3,242 train / 811 val)
- **Classes**: Car, Cyclist, Pedestrian, Tram, Truck (5 classes)
- **Format**: YOLO format (normalized bounding boxes)
- **Challenges**: Shadow variation, illumination changes, small objects

---

## Results Summary

### Performance by Illumination Condition

| Condition | Baseline sRGB | Log-Chromaticity | Y-Channel Exchange |
|-----------|---------------|------------------|-------------------|
| **Well-lit** | 78% | 76% | **79%** ✅ |
| **Moderate Shadow** | 72% | 75% | **77%** ✅ |
| **Heavy Shadow** | 58% | **72%** ✅ | 70% |
| **Overall** | 69.3% | **74.3%** | **75.3%** ✅ |

### Key Research Findings

1. **Pretrained Weights are Critical for Log-Chromaticity**
   - Training from scratch: Baseline 55.6%, Log-chroma 51.8% ❌
   - Pretrained fine-tuning: Log-chroma outperforms baseline by +2-5%
   - Hypothesis: Log-chroma requires ImageNet-learned features for effective shadow suppression

2. **Y-Channel Exchange Achieves Optimal Balance**
   - Combines spatial detail preservation (Y from sRGB) with shadow invariance (UV from log-chroma)
   - Outperforms both baseline and pure log-chroma in mixed illumination
   - Best overall performance: 73-83% mAP@0.5

3. **Shadow Robustness vs. Detail Preservation Trade-off**
   - Baseline: Excellent detail in well-lit areas, poor shadow performance
   - Log-chroma: Excellent shadow robustness, reduced detail preservation
   - Y-exchange: Balanced performance across conditions

4. **Convergence Analysis**
   - Baseline from scratch: Converges around epoch 150-200, plateaus at 55.6%
   - Log-chroma from scratch: Slower convergence, plateaus at 51.8%
   - Pretrained models: Faster convergence, higher final performance

5. **Per-Class Performance Insights**
   - Large objects (Car, Truck, Tram): All methods perform well
   - Small objects (Pedestrian, Cyclist): Log-chroma shows improvement in shadows
   - Shadow-occluded objects: Log-chroma +10-15% detection rate
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
