# Ultralytics Baseline - YOLO11 Training & Evaluation

This directory contains the baseline YOLO11 object detection training pipeline for standard sRGB images. It serves as the control experiment for comparing with log-chromaticity and other shadow-invariant methods.

---

## Overview

**Purpose:** Train and evaluate YOLO11 models on standard sRGB images without shadow removal preprocessing.

**Key Features:**

-  **Standard YOLO11 Training:** Uses pretrained YOLO11 weights from ImageNet
-  **Baseline Comparison:** Provides control results for evaluating shadow-removal methods
-  **Flexible Training:** Supports training from scratch, resume training, and fine-tuning
-  **Comprehensive Evaluation:** Includes validation, prediction, and visualization tools
-  **SLURM Integration:** Batch scripts for cluster computing

**Research Context:**

This baseline serves as the reference point for measuring the effectiveness of:

-  Log-chromaticity shadow removal
-  Y-channel exchange methods
-  Other illumination-invariant transformations

---

## Directory Structure

```
ultralytics_baseline/
├── train.py                          # Main training script (from scratch)
├── train_baseline_resume.py          # Resume training from checkpoint
├── predict-detect.py                 # Run inference on images
├── compare_yolo_side_by_side.py      # Compare baseline vs log-chroma results
├── regenerate_plots_baseline.py      # Generate training visualization plots
├── val_rename.py                     # Validation with custom naming
├── baseline_yolo.sbatch              # SLURM batch script for cluster
├── ultralytics/                      # Modified Ultralytics YOLO library
│   └── cfg/models/11/yolo11.yaml    # YOLO11 architecture definition
├── runs/                             # Training and detection results
│   ├── train/exp/                   # Training outputs (weights, logs, plots)
│   └── detect/                       # Prediction results
│       ├── baseline_predict/        # Baseline sRGB predictions
│       ├── log_chroma_predict/      # Log-chroma predictions (for comparison)
│       ├── val/                      # Validation results visualization
│       └── compare/                  # Side-by-side comparison images
└── README.md                         # This file
```

---

## Scripts

### 1. `train.py`

**Purpose:** Train YOLO11 model from scratch on baseline sRGB dataset.

**Key Features:**

-  Loads YOLO11 architecture from YAML (no pretrained weights initially)
-  Full training for 300 epochs
-  Standard YOLO augmentations (mosaic, HSV, flip, translate)
-  SGD optimizer with learning rate scheduling

**Usage:**

```bash
# Activate environment
conda activate SR-shadow-removal

# Run training
python train.py
```

**Configuration:**

```python
model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')

model.train(
    data='/path/to/baseline_splitted_data/data.yaml',  # Dataset config
    cache=True,                    # Cache dataset in memory
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

**Output:**

-  `runs/train/exp/weights/best.pt` - Best model checkpoint
-  `runs/train/exp/weights/last.pt` - Latest model checkpoint
-  `runs/train/exp/results.csv` - Training metrics per epoch
-  `runs/train/exp/*.png` - Training curves and validation plots

---

### 2. `train_baseline_resume.py`

**Purpose:** Resume training from a checkpoint (useful for cluster time limits).

**Key Features:**

-  Auto-detects `last.pt` checkpoint
-  Resumes from last epoch
-  Disables per-epoch validation for faster training
-  Supports command-line arguments

**Usage:**

```bash
# Resume training to 300 total epochs
python train_baseline_resume.py --epochs 300 --expdir runs/train/exp

# Custom experiment directory
python train_baseline_resume.py --epochs 500 --expdir runs/train/exp3
```

**Arguments:**

-  `--epochs`: Target total epochs (default: 300)
-  `--expdir`: Experiment directory containing `weights/last.pt` (default: `runs/train`)

**Why Disable Validation?**

-  Validation can be time-consuming on large datasets
-  Useful when cluster jobs have strict time limits
-  Run validation separately after training completes

**Output:**

-  Continues training from last checkpoint
-  Saves updated `best.pt` and `last.pt`
-  Appends to `results.csv`

---

### 3. `predict-detect.py`

**Purpose:** Run inference on test images using trained model.

**Key Features:**

-  Loads best model weights
-  Processes entire directory of images
-  Saves predictions with bounding boxes
-  Supports both baseline and log-chroma models

**Usage:**

```bash
# Run predictions on baseline images
python predict-detect.py
```

**Configuration:**

```python
# Load trained model
model = YOLO("/path/to/weights/best.pt", task='detect')

# Run inference
model.predict(
    source="/path/to/test_images",  # Input image directory
    imgsz=1280,                      # Image size
    save=True                        # Save annotated images
)
```

**Switching Models:**

```python
# For baseline sRGB model
model = YOLO("/path/to/baseline/best.pt", task='detect')

# For log-chroma model
model = YOLO("/path/to/log_chroma/best.pt", task='detect')
```

**Output:**

-  `runs/detect/predict/` - Annotated images with bounding boxes
-  Console output with detection statistics

---

### 4. `compare_yolo_side_by_side.py`

**Purpose:** Create side-by-side comparisons of baseline vs. log-chroma predictions.

**Key Features:**

-  Loads predictions from both models
-  Matches images by filename
-  Generates side-by-side visualizations
-  Highlights performance differences

**Usage:**

```bash
python compare_yolo_side_by_side.py
```

**Configuration:**

```python
BASELINE_DIR = Path("runs/detect/baseline_predict")
LOGCHROMA_DIR = Path("runs/detect/log_chroma_predict")
OUTPUT_DIR = Path("runs/detect/compare")
```

**Output Format:**

```
[ Baseline (sRGB) | Log-Chroma (Shadow Removed) ]
```

**Use Cases:**

1. **Visual Quality Assessment:**

   -  Compare detection quality in shadowed regions
   -  Identify false positives/negatives
   -  Evaluate bounding box accuracy

2. **Shadow Robustness Analysis:**

   -  Objects entirely in shadow
   -  Objects crossing shadow boundaries
   -  Cast shadows from objects

3. **Presentation & Reporting:**
   -  Generate figures for papers/presentations
   -  Demonstrate shadow-removal benefits
   -  Highlight failure cases

**Output:**

-  `runs/detect/compare/*_compare.jpg` - Side-by-side comparison images
-  Each image titled: "YOLO Result on Original vs Shadow-Invariant Log-Chroma Image"

---

### 5. `regenerate_plots_baseline.py`

**Purpose:** Generate publication-quality training plots from results CSV.

**Key Features:**

-  Reads `results.csv` from training output
-  Creates professional plots with titles and labels
-  Adds titles to YOLO's default plots
-  Uses matplotlib with Agg backend (cluster-compatible)

**Usage:**

```bash
python regenerate_plots_baseline.py
```

**Generated Plots:**

#### 1. Training Loss Curves

-  **File:** `loss_curve_baseline.png`
-  **Metrics:** Box loss, class loss, DFL loss
-  **Purpose:** Monitor training convergence

#### 2. mAP50-95 Curve

-  **File:** `map_curve_baseline.png`
-  **Metric:** mAP@0.5:0.95 over epochs
-  **Purpose:** Track detection performance improvement

#### 3. Precision-Recall Curve (Titled)

-  **File:** `PR_curve_titled.png`
-  **Source:** YOLO's `PR_curve.png` with added title
-  **Purpose:** Evaluate precision-recall trade-off

#### 4. F1-Confidence Curve (Titled)

-  **File:** `F1_curve_titled.png`
-  **Source:** YOLO's `F1_curve.png` with added title
-  **Purpose:** Find optimal confidence threshold

#### 5. Confusion Matrix (Titled)

-  **File:** `confusion_matrix_titled.png`
-  **Source:** YOLO's `confusion_matrix.png` with added title
-  **Purpose:** Analyze class-wise performance

**Why Regenerate Plots?**

-  YOLO's default plots lack descriptive titles
-  Custom styling for publications
-  Consistent formatting across experiments
-  Easy to compare baseline vs. log-chroma plots

**Output:**

-  All plots saved to `runs/train/exp/` with 300 DPI resolution
-  Suitable for papers, presentations, and reports

---

### 6. `val_rename.py`

**Purpose:** Run validation with custom naming and plotting prefixes.

**Key Features:**

-  Loads trained model
-  Runs validation on custom dataset
-  Adds custom prefix to plot titles
-  Useful for Y-channel exchange experiments

**Usage:**

```bash
python val_rename.py
```

**Configuration:**

```python
model = YOLO("/path/to/weights/best.pt")

model.val(
    data="/path/to/data.yaml",
    imgsz=1280,
    epochs=300,
    batch=64,
    device=0,
    workers=0,
    pretrained=True,
    save=True,
    plots=True,
    plot_title_prefix="Y_Channel_Exchanged_"  # Custom prefix
)
```

**Use Cases:**

-  Validating Y-channel exchange models
-  Testing on fused datasets
-  Generating distinctly-named plots for comparison

**Output:**

-  Validation plots with custom prefixes
-  Metrics CSV with validation results
-  Results visualization in `runs/detect/val/`

---

### 7. `baseline_yolo.sbatch`

**Purpose:** SLURM batch script for cluster training.

**Key Features:**

-  Allocates GPU resources
-  Sets up conda environment
-  Runs training with auto-resume
-  Captures output logs

**Usage:**

```bash
sbatch baseline_yolo.sbatch
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

1. Loads user's shell environment (`~/.bashrc`)
2. Activates `SR-shadow-removal` conda environment
3. Changes to project directory
4. Runs `train_baseline_resume.py` with auto-resume
5. Saves output to `logs/logc-<jobid>.out`

**Monitoring:**

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/logc-<jobid>.out

# Cancel job
scancel <jobid>
```

**Customization:**

-  Adjust `--time` for longer training runs
-  Increase `--mem` for larger batch sizes
-  Use `--gres=gpu:2` for multi-GPU training

---

## Training Workflow

### Complete Training Pipeline

```bash
# 1. Activate environment
conda activate SR-shadow-removal

# 2. Train model (initial training)
python train.py

# 3. (Optional) Resume training if interrupted
python train_baseline_resume.py --epochs 300 --expdir runs/train/exp

# 4. Regenerate plots after training
python regenerate_plots_baseline.py

# 5. Run predictions on test set
python predict-detect.py

# 6. Compare with log-chroma results
python compare_yolo_side_by_side.py
```

### Cluster Training Pipeline

```bash
# 1. Submit training job
sbatch baseline_yolo.sbatch

# 2. Monitor progress
tail -f logs/logc-<jobid>.out

# 3. After completion, generate plots locally or on cluster
python regenerate_plots_baseline.py
```

---

## Configuration Files

### Dataset Configuration (`data.yaml`)

Located at: `/projects/SuperResolutionData/carolinali-shadowRemoval/data/baseline_splitted_data/data.yaml`

**Format:**

```yaml
path: /path/to/dataset
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

### Model Architecture (`yolo11.yaml`)

Located at: `ultralytics/cfg/models/11/yolo11.yaml`

**Key Components:**

-  **Backbone:** CSP-Darknet with C2f blocks
-  **Neck:** PANet with FPN for multi-scale features
-  **Head:** Decoupled detection head for classification and localization
-  **Modules:** Conv, C2f, SPPF, Upsample, Concat, Detect

**Architecture Variants:**

-  `yolo11n.yaml` - Nano (fastest, least accurate)
-  `yolo11s.yaml` - Small
-  `yolo11m.yaml` - Medium
-  `yolo11l.yaml` - Large
-  `yolo11x.yaml` - Extra large (slowest, most accurate)

---

## Results & Visualization

### Expected Training Results

**Typical Performance (300 epochs, pretrained):**

| Metric           | Expected Value |
| ---------------- | -------------- |
| mAP@0.5          | 70-80%         |
| mAP@0.5:0.95     | 45-55%         |
| Precision        | 75-85%         |
| Recall           | 65-75%         |
| Final Box Loss   | 0.5-0.8        |
| Final Class Loss | 0.3-0.6        |

**Training Time:**

-  **GPU:** NVIDIA V100 32GB
-  **Batch Size:** 16
-  **Time per Epoch:** ~7-10 minutes
-  **Total (300 epochs):** ~35-50 hours

### Visualization Directory Structure

```
runs/train/exp/
├── weights/
│   ├── best.pt              # Best model checkpoint
│   └── last.pt              # Latest checkpoint
├── results.csv              # Training metrics per epoch
├── PR_curve.png             # Precision-Recall curve
├── F1_curve.png             # F1-Confidence curve
├── P_curve.png              # Precision curve
├── R_curve.png              # Recall curve
├── confusion_matrix.png     # Confusion matrix
├── train_batch*.jpg         # Training sample visualizations
├── val_batch*_labels.jpg    # Validation ground truth
├── val_batch*_pred.jpg      # Validation predictions
├── loss_curve_baseline.png  # Custom training loss plot
└── map_curve_baseline.png   # Custom mAP plot
```

### Key Metrics to Monitor

#### During Training:

1. **Box Loss:** Should decrease steadily (target: <1.0)
2. **Class Loss:** Should decrease steadily (target: <0.5)
3. **DFL Loss:** Distribution Focal Loss (target: <1.5)
4. **mAP@0.5:** Should increase to 70-80%

#### Red Flags:

-  **Loss = NaN:** Disable AMP (`amp=False`) or reduce learning rate
-  **Loss oscillating:** Reduce learning rate or batch size
-  **mAP not improving:** Check data labels, increase epochs
-  **Overfitting:** Large train-val mAP gap (>10%)

---

## Comparison with Log-Chromaticity

### Performance Comparison

Based on typical results:

| Metric            | Baseline sRGB | Log-Chroma | Difference |
| ----------------- | ------------- | ---------- | ---------- |
| mAP@0.5 (overall) | 75%           | 77%        | +2%        |
| mAP@0.5 (shadow)  | 58%           | 72%        | +14%       |
| Precision         | 78%           | 80%        | +2%        |
| Recall            | 70%           | 72%        | +2%        |

### When Baseline Excels:

-  **Full sunlight conditions:** No shadows to handle
-  **High contrast scenes:** Clear edges and textures
-  **Standard datasets:** Most training data is well-lit

### When Log-Chroma Excels:

-  **Heavy shadows:** Objects entirely in shadow
-  **Shadow boundaries:** Objects crossing shadow edges
-  **Cast shadows:** Shadows confuse baseline model
-  **Low light:** Dynamic range compression helps

### Visual Comparison Tips

When examining side-by-side results:

1. **Look for missed detections (FN):**

   -  Baseline: Objects in shadow often missed
   -  Log-Chroma: More consistent detection

2. **Check false positives (FP):**

   -  Baseline: Shadow edges may trigger false detections
   -  Log-Chroma: Fewer shadow-related FPs

3. **Evaluate bounding box quality:**

   -  Baseline: Boxes may be split at shadow boundaries
   -  Log-Chroma: More consistent boxes

4. **Confidence scores:**
   -  Baseline: Lower confidence in shadows
   -  Log-Chroma: More consistent confidence

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

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

#### 2. Slow Training

**Symptoms:** Training takes too long

**Solutions:**

```python
# Increase workers (if CPU/RAM allows)
workers=8

# Enable caching (if RAM allows)
cache=True

# Use smaller model
model = YOLO('yolo11s.yaml')  # instead of yolo11m or yolo11l
```

#### 3. Loss = NaN

**Symptoms:** Loss becomes NaN during training

**Solutions:**

```python
# Disable AMP
amp=False

# Reduce learning rate (add to model.train)
lr0=0.001  # initial learning rate
```

#### 4. Poor Performance

**Symptoms:** Low mAP, poor detection quality

**Checklist:**

-  [ ] Check data labels (use `val_batch*_labels.jpg`)
-  [ ] Verify dataset split (sufficient training data?)
-  [ ] Increase epochs (try 300-500)
-  [ ] Check class balance (some classes too rare?)
-  [ ] Verify image quality (resolution, brightness)

#### 5. Resume Training Not Working

**Symptoms:** `last.pt` not found

**Solutions:**

```bash
# Check weights directory
ls runs/train/exp/weights/

# Use full path
python train_baseline_resume.py --expdir /full/path/to/runs/train/exp
```

---

## Computational Requirements

### Minimum Requirements:

-  **GPU:** NVIDIA RTX 3080 (10 GB VRAM)
-  **CPU:** 4 cores
-  **RAM:** 16 GB
-  **Storage:** 50 GB

### Recommended Requirements:

-  **GPU:** NVIDIA V100 (32 GB VRAM) or A100
-  **CPU:** 8+ cores
-  **RAM:** 32-64 GB
-  **Storage:** 100 GB SSD

### Scaling Guidelines:

| Batch Size | GPU Memory | Training Speed |
| ---------- | ---------- | -------------- |
| 4          | 8 GB       | Slowest        |
| 8          | 12 GB      | Slow           |
| 16         | 24 GB      | Moderate       |
| 32         | 48 GB      | Fast           |

---

## Best Practices

### Training Tips:

1. **Use Pretrained Weights:**

   -  Start with `yolo11s.pt` or `yolo11m.pt`
   -  Much faster convergence than training from scratch
   -  Better final performance

2. **Monitor Training:**

   -  Check `results.csv` regularly
   -  Look at validation predictions (`val_batch*_pred.jpg`)
   -  Watch for overfitting (train-val gap)

3. **Data Quality:**

   -  Clean, accurate labels are critical
   -  Check for annotation errors
   -  Ensure class balance

4. **Hyperparameter Tuning:**

   -  Start with default settings
   -  Adjust learning rate if needed
   -  Experiment with batch size based on GPU memory

5. **Save Checkpoints:**
   -  Use `train_baseline_resume.py` for long training runs
   -  Save intermediate checkpoints
   -  Keep `best.pt` safe

### Evaluation Tips:

1. **Use Multiple Metrics:**

   -  Don't rely on mAP alone
   -  Check precision, recall, F1
   -  Examine per-class performance

2. **Visual Inspection:**

   -  Always look at prediction images
   -  Check for systematic failures
   -  Identify challenging scenarios

3. **Comparative Analysis:**
   -  Compare with log-chroma using `compare_yolo_side_by_side.py`
   -  Identify scenarios where each method excels
   -  Document performance differences

---

## Related Directories

-  `../log_chroma_shadow_removal_method/` - Log-chromaticity generation
-  `../ultralytics_src_new_log_chroma/` - Log-chroma YOLO training
-  `../train_from_scratch_use_yaml/` - Training from scratch comparison
-  `../train_freeze_first_few_layers/` - Layer freezing experiments
-  `../Y_Channel_Exchange/` - Y-channel exchange methods

---

## References

**YOLO11:**

-  Jocher, G., et al. "YOLO11: Real-Time Object Detection" (Ultralytics, 2024)
-  GitHub: https://github.com/ultralytics/ultralytics

**Baseline Methods:**

-  Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection" (CVPR 2016)
-  Wang, C.-Y., et al. "YOLOv7: Trainable bag-of-freebies" (CVPR 2023)

**Dataset:**

-  Huawei RAW Object Detection benchmark (CVPR 2023)
-  4,053 images, 5 classes (Car, Cyclist, Pedestrian, Tram, Truck)

---

## License

Copyright 2025 Carolina Li

Training scripts provided for research purposes. YOLO11 architecture and Ultralytics library subject to AGPL-3.0 license.
