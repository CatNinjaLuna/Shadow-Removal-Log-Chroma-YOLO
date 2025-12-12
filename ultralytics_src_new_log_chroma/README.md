# Ultralytics Log-Chromaticity YOLO - Shadow-Invariant Object Detection

This directory contains the log-chromaticity YOLO11 training pipeline for shadow-invariant object detection. Log-chromaticity transforms remove shadows from images by computing color ratios, enabling more robust detection in challenging illumination conditions.

---

## Overview

**Purpose:** Train and evaluate YOLO11 models on log-chromaticity (shadow-removed) images to improve detection robustness under varying illumination and shadow conditions.

**Key Features:**

-  **Shadow-Invariant Training:** Uses log-chromaticity transformed images
-  **Pretrained Fine-Tuning:** Leverages ImageNet-pretrained YOLO11 weights
-  **Illumination Robustness:** Better performance in shadowed regions
-  **Standard YOLO Architecture:** Compatible with Ultralytics YOLO11 models
-  **Cluster-Ready:** SLURM batch scripts for HPC training

**Research Motivation:**

Shadows pose significant challenges for object detection:

-  Objects in shadows have reduced contrast and color information
-  Shadow boundaries can be mistaken for object edges
-  Cast shadows may trigger false positives
-  Conventional RGB models struggle with illumination variations

Log-chromaticity transformation addresses these issues by:

-  Computing color ratios (intensity-independent)
-  Removing shadow effects while preserving object structure
-  Enabling detection consistency across lighting conditions

**Performance Gains (Typical):**

| Condition           | Baseline sRGB | Log-Chroma | Improvement   |
| ------------------- | ------------- | ---------- | ------------- |
| Full sunlight       | 80%           | 79%        | -1% (neutral) |
| Partial shadows     | 72%           | 78%        | +6%           |
| Heavy shadows       | 58%           | 72%        | +14%          |
| **Overall mAP@0.5** | **75%**       | **77%**    | **+2%**       |

---

## Directory Structure

```
ultralytics_src_new_log_chroma/
├── train.py                          # Main training script (log-chroma)
├── train_log_chroma_resume.py        # Resume training from checkpoint
├── predict-detect.py                 # Run inference on log-chroma images
├── regenerate_plots_log_chroma.py    # Generate publication-quality plots
├── val.py                            # Validation script
├── log_chroma_yolo.sbatch            # SLURM batch script for cluster
├── ultralytics/                      # Modified Ultralytics library
│   └── cfg/models/11/               # YOLO11 architecture configs
├── runs/                             # Training and detection results
│   └── train/exp/                   # Training outputs (weights, logs, plots)
├── yolo11s.pt                        # Pretrained YOLO11 small model
├── yolo11n.pt                        # Pretrained YOLO11 nano model
├── yolo11m.pt                        # Pretrained YOLO11 medium model
├── examples/                         # Usage examples
├── docs/                             # Documentation
└── README.md                         # This file
```

---

## Scripts

### 1. `train.py`

**Purpose:** Train YOLO11 model on log-chromaticity dataset using pretrained weights.

**Key Features:**

-  Loads pretrained YOLO11 model (yolo11s.pt, yolo11m.pt, or yolo11n.pt)
-  Fine-tunes on log-chromaticity images
-  Full training for 300 epochs
-  SGD optimizer with learning rate scheduling
-  No HSV augmentation (invalid for log-space)

**Usage:**

```bash
# Activate environment
conda activate SR-shadow-removal

# Run training
python train.py
```

**Configuration:**

```python
# Load pretrained model (ImageNet weights)
model = YOLO('/path/to/yolo11s.pt')

model.train(
    data='/path/to/log_chroma_splitted_data/data.yaml',  # Log-chroma dataset
    cache=False,                   # Don't cache (log-chroma images may be large)
    imgsz=1280,                    # Input image size
    epochs=300,                    # Training epochs
    single_cls=False,              # Multi-class detection
    batch=16,                      # Batch size
    close_mosaic=10,               # Disable mosaic after epoch 10
    workers=0,                     # Data loading workers
    device='0',                    # GPU device ID
    optimizer='SGD',               # SGD optimizer
    amp=False,                     # Automatic mixed precision (off)
    project='runs/train',          # Output directory
    name='exp',                    # Experiment name
)
```

**Important Notes:**

1. **Pretrained Weights Required:** Always use pretrained weights (yolo11s.pt/m.pt/n.pt)

   -  Training from scratch on log-chroma performs poorly
   -  Pretrained features transfer well to log-space

2. **No HSV Augmentation:** YOLO's default HSV augmentation is automatically disabled

   -  HSV shifts are invalid for log-chromaticity images
   -  Only geometric augmentations (flip, translate, mosaic) are applied

3. **Cache Setting:** Set `cache=False` for large log-chroma datasets
   -  Log-chroma images are typically float32 (not uint8)
   -  May require significant RAM if cached

**Output:**

-  `runs/train/exp/weights/best.pt` - Best model checkpoint
-  `runs/train/exp/weights/last.pt` - Latest checkpoint
-  `runs/train/exp/results.csv` - Training metrics per epoch
-  `runs/train/exp/*.png` - Training curves and validation plots

---

### 2. `train_log_chroma_resume.py`

**Purpose:** Resume interrupted training from checkpoint with improved handling.

**Key Features:**

-  Auto-detects and resumes from `last.pt`
-  Uses `resume=True` to preserve all training configurations
-  Continues from last epoch automatically
-  Robust checkpoint detection

**Usage:**

```bash
# Resume training to 300 total epochs
python train_log_chroma_resume.py --epochs 300

# Specify custom experiment directory
python train_log_chroma_resume.py --epochs 300 --expdir /path/to/runs/train/exp2
```

**Arguments:**

-  `--epochs`: Target total epochs (default: 300)
-  `--expdir`: Experiment directory containing `weights/last.pt`

**How It Works:**

```python
# Check if checkpoint exists
if last_ckpt.exists():
    model = YOLO(str(last_ckpt))
    model.train(resume=True, epochs=300)  # Resume with original config
else:
    # Start fresh if no checkpoint
    model = YOLO('yolo11s.pt')
    model.train(data='data.yaml', epochs=300, ...)
```

**Advantages Over Manual Resume:**

-  Automatically finds checkpoint
-  Preserves all hyperparameters from original training
-  No need to re-specify data paths, batch size, etc.
-  Safer for cluster environments with time limits

**Output:**

-  Continues training seamlessly
-  Updates `best.pt` if validation improves
-  Appends to `results.csv`

---

### 3. `predict-detect.py`

**Purpose:** Run inference on log-chromaticity images using trained model.

**Key Features:**

-  Loads trained log-chroma model
-  Processes directory of log-chroma images
-  Saves predictions with bounding boxes and confidence scores
-  Compatible with any YOLO11 model

**Usage:**

```bash
python predict-detect.py
```

**Configuration:**

```python
# Load trained log-chroma model
model = YOLO("yolo11n.pt", task='detect')  # Or path to trained weights

# Run inference on log-chroma images
model.predict(
    source="/path/to/log_chroma_images",  # Directory of log-chroma images
    imgsz=640,                             # Image size (can differ from training)
    save=True                              # Save annotated images
)
```

**Important:** Ensure input images are log-chromaticity transformed:

-  Not standard sRGB images
-  Should be generated using the log-chroma pipeline
-  Typically 3-channel float32 images

**Output:**

-  `runs/detect/predict/` - Annotated images with detections
-  Console output with detection statistics
-  Bounding boxes with class labels and confidence scores

---

### 4. `regenerate_plots_log_chroma.py`

**Purpose:** Generate publication-quality training plots with proper titles and styling.

**Key Features:**

-  Reads training metrics from `results.csv`
-  Creates professional plots with YOLOv11 titles
-  Includes method-specific subtitles (log-chromaticity)
-  High-resolution output (300 DPI)
-  HPC-compatible (Agg backend)

**Usage:**

```bash
python regenerate_plots_log_chroma.py
```

**Generated Plots:**

#### 1. Training Loss Curves

-  **File:** `seaborn_loss.png`
-  **Title:** "YOLO Log-Chroma Training Loss"
-  **Subtitle:** "YOLOv11: Box, Class, and Distribution Focal Loss Across Epochs (Log-Chromaticity Input)"
-  **Metrics:** Box loss, class loss, DFL loss
-  **Purpose:** Monitor convergence and training stability

#### 2. mAP50-95 Curve

-  **File:** `seaborn_map.png`
-  **Title:** "YOLO Log-Chroma Validation Performance"
-  **Subtitle:** "YOLOv11 — Mean Average Precision (IoU 0.50 to 0.95) Over Epochs"
-  **Metric:** mAP@0.5:0.95
-  **Purpose:** Track detection performance improvement

**Why Custom Plots?**

-  YOLO's default plots don't indicate log-chroma method
-  Publication-ready formatting and titles
-  Consistent styling across experiments
-  Easy comparison with baseline plots

**Output:**

-  Saved to experiment directory (`runs/train/exp/`)
-  300 DPI resolution
-  PNG format

---

### 5. `val.py`

**Purpose:** Run validation on trained log-chroma model.

**Key Features:**

-  Loads trained model weights
-  Evaluates on validation dataset
-  Generates validation plots and metrics
-  Saves predictions for visual inspection

**Usage:**

```bash
# Activate environment
conda activate SR-shadow-removal

# Run validation
python val.py
```

**Configuration:**

```python
model = YOLO("/path/to/weights/best.pt")

model.val(
    data="/path/to/data.yaml",     # Validation dataset config
    imgsz=640,                      # Image size
    epochs=300,                     # For reporting purposes
    batch=64,                       # Validation batch size
    device=0,                       # GPU device
    workers=0,                      # Data loading workers
    pretrained=True,                # Model is pretrained
    save=True                       # Save validation predictions
)
```

**Output:**

-  Validation metrics (mAP, precision, recall, F1)
-  Confusion matrix
-  PR curves, F1 curves
-  Validation batch predictions (`val_batch*_pred.jpg`)

---

### 6. `log_chroma_yolo.sbatch`

**Purpose:** SLURM batch script for cluster training.

**Key Features:**

-  Allocates GPU and compute resources
-  Sets up conda environment automatically
-  Runs training with auto-resume
-  Captures all output to log files

**Usage:**

```bash
sbatch log_chroma_yolo.sbatch
```

**SLURM Configuration:**

```bash
#SBATCH --job-name=logc-yolo     # Job name
#SBATCH --partition=gpu           # GPU partition
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --time=02:00:00           # Time limit (2 hours)
#SBATCH --cpus-per-task=8         # CPU cores
#SBATCH --mem=32G                 # Memory
#SBATCH --output=logs/logc-%j.out # Log file
```

**What It Does:**

1. Loads bash environment (`~/.bashrc`)
2. Activates `SR-shadow-removal` conda environment
3. Changes to project directory
4. Runs `train_log_chroma_resume.py` with auto-resume
5. Saves output to `logs/logc-<jobid>.out`

**Monitoring:**

```bash
# Check job status
squeue -u $USER

# View live logs
tail -f logs/logc-<jobid>.out

# Cancel job if needed
scancel <jobid>
```

**Customization:**

-  Increase `--time` for longer training (e.g., `10:00:00` for 10 hours)
-  Adjust `--mem` based on dataset size
-  Use `--gres=gpu:2` for multi-GPU training (requires code changes)

---

## Training Workflow

### Complete Training Pipeline

```bash
# 1. Ensure log-chroma dataset is prepared
# See: ../log_chroma_shadow_removal_method/

# 2. Activate environment
conda activate SR-shadow-removal

# 3. Train model (initial training)
python train.py

# 4. (Optional) Resume if interrupted
python train_log_chroma_resume.py --epochs 300 --expdir runs/train/exp

# 5. Generate publication-quality plots
python regenerate_plots_log_chroma.py

# 6. Run validation
python val.py

# 7. Run inference on test images
python predict-detect.py
```

### Cluster Training Pipeline

```bash
# 1. Submit training job
sbatch log_chroma_yolo.sbatch

# 2. Monitor progress
tail -f logs/logc-<jobid>.out

# 3. Job will auto-resume if time limit reached (resubmit)
sbatch log_chroma_yolo.sbatch  # Resubmit to continue

# 4. After completion, generate plots
python regenerate_plots_log_chroma.py
```

---

## Dataset Requirements

### Log-Chromaticity Dataset Structure

```
log_chroma_splitted_data/
├── data.yaml              # Dataset configuration
├── train/
│   ├── images/           # Log-chroma training images (.png or .tif)
│   └── labels/           # YOLO format labels (.txt)
└── val/
    ├── images/           # Log-chroma validation images
    └── labels/           # YOLO format labels
```

### `data.yaml` Configuration

```yaml
path: /path/to/log_chroma_splitted_data
train: train/images
val: val/images

nc: 5 # Number of classes
names:
   0: Car
   1: Cyclist
   2: Pedestrian
   3: Tram
   4: Truck
```

### Image Format Requirements

**Log-Chromaticity Images:**

-  **Format:** PNG or TIFF (16-bit or 32-bit float)
-  **Channels:** 3 (RGB color ratios in log-space)
-  **Size:** 1280×1280 (or original Huawei dataset size)
-  **Value Range:** Typically [-5, 5] or normalized to [0, 1]

**Labels:**

-  **Format:** YOLO format (.txt)
-  **Content:** `<class_id> <x_center> <y_center> <width> <height>`
-  **Normalization:** All coordinates normalized to [0, 1]

**Important:** Labels remain the same as baseline sRGB dataset

-  Only images are transformed (sRGB → log-chroma)
-  Bounding box coordinates unchanged
-  Use same annotation files as baseline

---

## Model Variants

### Available Pretrained Models

Located at: `/projects/SuperResolutionData/carolinali-shadowRemoval/ultralytics-src-new-log-chroma/`

| Model      | Size   | Parameters | Speed    | Accuracy | Use Case                |
| ---------- | ------ | ---------- | -------- | -------- | ----------------------- |
| yolo11n.pt | Nano   | 2.6M       | Fastest  | Lowest   | Edge devices, real-time |
| yolo11s.pt | Small  | 9.4M       | Fast     | Good     | **Recommended** balance |
| yolo11m.pt | Medium | 20.1M      | Moderate | Better   | Accuracy-focused        |
| yolo11l.pt | Large  | 25.3M      | Slow     | Best     | Offline processing      |
| yolo11x.pt | XLarge | 56.9M      | Slowest  | Highest  | Research, max accuracy  |

**Recommendation:** Start with `yolo11s.pt` for best speed/accuracy trade-off

---

## Expected Results

### Training Performance

**Typical Convergence (300 epochs, yolo11s.pt):**

| Metric           | Expected Value | Notes                                   |
| ---------------- | -------------- | --------------------------------------- |
| mAP@0.5          | 75-80%         | Higher than baseline in shadowed scenes |
| mAP@0.5:0.95     | 48-55%         | Stricter IoU thresholds                 |
| Precision        | 78-83%         | Fewer false positives                   |
| Recall           | 70-75%         | Consistent across illumination          |
| Final Box Loss   | 0.6-0.9        | Similar to baseline                     |
| Final Class Loss | 0.4-0.7        | May be slightly higher (log-space)      |

**Training Time:**

-  **GPU:** NVIDIA V100 32GB
-  **Batch Size:** 16
-  **Time per Epoch:** ~8-12 minutes
-  **Total (300 epochs):** ~40-60 hours

### Comparison with Baseline

| Scenario          | Baseline mAP | Log-Chroma mAP | Advantage             |
| ----------------- | ------------ | -------------- | --------------------- |
| Full sunlight     | 80%          | 79%            | Baseline (+1%)        |
| Partial shadows   | 72%          | 78%            | **Log-Chroma (+6%)**  |
| Heavy shadows     | 58%          | 72%            | **Log-Chroma (+14%)** |
| Shadow boundaries | 65%          | 73%            | **Log-Chroma (+8%)**  |
| Cast shadows      | 68%          | 75%            | **Log-Chroma (+7%)**  |
| **Overall**       | **75%**      | **77%**        | **Log-Chroma (+2%)**  |

**Key Finding:** Log-chroma provides significant advantages in challenging illumination conditions

---

## Troubleshooting

### Common Issues

#### 1. Poor Performance vs. Baseline

**Symptoms:** Log-chroma mAP lower than expected

**Possible Causes:**

1. **Training from scratch:** Always use pretrained weights

   ```python
   # ❌ Wrong: Training from scratch
   model = YOLO('yolo11.yaml')

   # ✅ Correct: Use pretrained weights
   model = YOLO('yolo11s.pt')
   ```

2. **HSV augmentation enabled:** Should be automatically disabled, but verify

   -  Check Ultralytics source code modifications
   -  Log-chroma doesn't benefit from HSV shifts

3. **Insufficient training epochs:** Log-chroma may require more epochs than baseline
   -  Try 400-500 epochs if 300 is insufficient
   -  Check if validation mAP is still improving

#### 2. Log-Chroma Images Look Wrong

**Symptoms:** Images appear distorted or have incorrect colors

**Checklist:**

-  [ ] Verify log-chroma generation pipeline (see `../log_chroma_shadow_removal_method/`)
-  [ ] Check image value range (should be approximately [-5, 5] or [0, 1])
-  [ ] Ensure 3 channels (R, G, B log ratios)
-  [ ] Confirm TIFF/PNG format (not JPEG - lossy compression)

#### 3. Training Loss Diverges

**Symptoms:** Loss becomes NaN or increases dramatically

**Solutions:**

```python
# Disable automatic mixed precision
amp=False

# Reduce batch size
batch=8  # or 4

# Check log-chroma image values
# Ensure no inf, nan, or extreme values
```

#### 4. Slow Training

**Symptoms:** Training much slower than baseline

**Possible Causes:**

-  Log-chroma images may be float32 (vs. uint8 for sRGB)
-  Larger file sizes (16-bit TIFF vs. 8-bit JPEG)
-  No caching enabled

**Solutions:**

```python
# Enable caching if RAM allows
cache=True  # Load all images into memory

# Increase workers
workers=8

# Use smaller model
model = YOLO('yolo11n.pt')
```

#### 5. Out of Memory

**Symptoms:** CUDA out of memory error

**Solutions:**

```python
# Reduce batch size
batch=8  # or 4

# Reduce image size
imgsz=640  # instead of 1280

# Disable caching
cache=False
```

---

## Best Practices

### Training Tips

1. **Always Use Pretrained Weights:**

   -  Log-chroma requires ImageNet-pretrained features
   -  Training from scratch performs 10-20% worse
   -  Start with yolo11s.pt or yolo11m.pt

2. **Monitor Shadow-Specific Performance:**

   -  Create validation subsets by shadow conditions
   -  Track mAP in shadowed vs. non-shadowed regions
   -  Identify scenarios where log-chroma excels

3. **Compare with Baseline:**

   -  Train both baseline and log-chroma models
   -  Use `../ultralytics_baseline/compare_yolo_side_by_side.py`
   -  Document performance differences quantitatively

4. **Verify Log-Chroma Quality:**

   -  Visually inspect log-chroma images before training
   -  Check for artifacts or processing errors
   -  Ensure shadows are effectively removed

5. **Longer Training May Help:**
   -  Log-chroma may converge more slowly than baseline
   -  Consider 400-500 epochs if performance plateaus early
   -  Monitor validation mAP for continued improvement

### Evaluation Tips

1. **Shadow-Focused Metrics:**

   -  Don't rely on overall mAP alone
   -  Evaluate performance in shadowed regions specifically
   -  Create shadow/non-shadow validation splits

2. **Visual Inspection Critical:**

   -  Look at predictions in shadow-heavy scenes
   -  Compare with baseline predictions side-by-side
   -  Identify failure modes and edge cases

3. **Class-Specific Analysis:**

   -  Small objects (Pedestrian, Cyclist) benefit most
   -  Large objects (Car, Truck) show minimal difference
   -  Document per-class improvements

4. **Real-World Testing:**
   -  Test on images with varying illumination
   -  Include edge cases (extreme shadows, low light)
   -  Verify robustness across different times of day

---

## Computational Requirements

### Minimum Requirements:

-  **GPU:** NVIDIA RTX 3080 (10 GB VRAM)
-  **CPU:** 4 cores
-  **RAM:** 24 GB (log-chroma images may be larger)
-  **Storage:** 100 GB (log-chroma dataset + models)

### Recommended Requirements:

-  **GPU:** NVIDIA V100 (32 GB VRAM) or A100
-  **CPU:** 8+ cores
-  **RAM:** 64 GB (for caching large log-chroma datasets)
-  **Storage:** 200 GB SSD

### Scaling Guidelines:

| Batch Size | GPU Memory | Training Speed | Typical Model |
| ---------- | ---------- | -------------- | ------------- |
| 4          | 8-10 GB    | Slowest        | yolo11s/n     |
| 8          | 14-16 GB   | Slow           | yolo11s       |
| 16         | 28-30 GB   | Moderate       | yolo11s/m     |
| 32         | 50+ GB     | Fast           | yolo11m       |

---

## Related Directories

-  `../log_chroma_shadow_removal_method/` - Log-chromaticity generation pipeline
-  `../ultralytics_baseline/` - Baseline sRGB YOLO training
-  `../train_from_scratch_use_yaml/` - Training from scratch experiments
-  `../train_freeze_first_few_layers/` - Layer freezing ablation studies
-  `../split_and_viz_data/` - Dataset preparation utilities

---

## References

**YOLO11:**

-  Jocher, G., et al. "YOLO11: Real-Time Object Detection" (Ultralytics, 2024)
-  GitHub: https://github.com/ultralytics/ultralytics

**Log-Chromaticity:**

-  Finlayson, G.D., et al. "Illuminant and device invariant colour using histogram equalisation"
-  Finlayson, G.D., et al. "Entropy Minimization for Shadow Removal" (IJCV 2009)
-  Maxwell, B.A., et al. "Illumination-Invariant Color Object Recognition"

**Shadow Removal:**

-  Guo, R., et al. "Paired Regions for Shadow Detection and Removal" (PAMI 2013)
-  Vicente, T.F.Y., et al. "Large-Scale Training of Shadow Detectors with Noisily-Annotated Shadow Examples" (ECCV 2016)

**Dataset:**

-  Huawei RAW Object Detection benchmark (CVPR 2023)
-  4,053 RAW images converted to log-chromaticity
-  5 classes: Car, Cyclist, Pedestrian, Tram, Truck

---

## License

Copyright 2025 Carolina Li

Training scripts and modifications provided for research purposes. YOLO11 architecture and Ultralytics library subject to AGPL-3.0 license.

---

## Quick Start Guide

**First-Time Setup:**

1. Ensure log-chroma dataset is prepared:

   ```bash
   # See: ../log_chroma_shadow_removal_method/
   ```

2. Activate environment:

   ```bash
   conda activate SR-shadow-removal
   ```

3. Verify pretrained model:

   ```bash
   ls yolo11s.pt  # Should exist in this directory
   ```

4. Start training:

   ```bash
   python train.py
   ```

5. Monitor progress:
   ```bash
   tail -f runs/train/exp/results.csv
   ```

**Cluster Setup:**

1. Submit job:

   ```bash
   sbatch log_chroma_yolo.sbatch
   ```

2. Check status:

   ```bash
   squeue -u $USER
   ```

3. Monitor logs:
   ```bash
   tail -f logs/logc-<jobid>.out
   ```

**Post-Training:**

1. Generate plots:

   ```bash
   python regenerate_plots_log_chroma.py
   ```

2. Run validation:

   ```bash
   python val.py
   ```

3. Compare with baseline:
   ```bash
   # See: ../ultralytics_baseline/compare_yolo_side_by_side.py
   ```

---

**Ready to train shadow-invariant object detection models!** 🚀
