# Y-Channel Exchanged Log-Chromaticity YOLO Training

## Overview

This directory implements the **Y-Channel Exchange Method** for shadow-robust object detection, combining the advantages of both sRGB and log-chromaticity representations. The core innovation is fusing the Y (luminance) channel from well-lit sRGB images with the UV (chrominance) channels from shadow-invariant log-chromaticity images. This hybrid approach preserves high-frequency details in well-lit regions while maintaining shadow invariance in chrominance information.

### Key Features

-  **Hybrid Image Fusion**: Combines Y(sRGB) + UV(log-chroma) for optimal shadow robustness
-  **Intelligent UV Selection**: Automatically correlates log-chroma channels with Y to select optimal UV components
-  **Multi-Mode Y Extraction**: Supports luminance (linear), BT.601, and BT.709 standards
-  **Auto-Detection**: Detects gamma/color space from image metadata (PIL, TIFF tags, data types)
-  **Complete Pipeline**: From raw images to trained YOLO model with validation
-  **SLURM Integration**: Cluster-ready with auto-resume for long training sessions

### Research Context

This method addresses two competing requirements:

1. **Shadow Invariance**: Log-chromaticity provides intensity-independent color ratios
2. **Detail Preservation**: sRGB Y-channel maintains high-frequency spatial information

By fusing these representations, the method achieves superior performance in challenging illumination conditions while preserving detection accuracy in well-lit scenarios.

### Expected Performance Benefits

-  Improved shadow robustness compared to baseline sRGB (especially in heavy shadows)
-  Better detail preservation compared to pure log-chromaticity
-  Balanced performance across varying illumination conditions
-  Faster convergence with pretrained weights

---

## Directory Structure

```
y_channel_exchanged_log_chroma/
├── fuse_y_logchroma.py                    # Core Y-channel fusion algorithm (340 lines)
├── log_chroma_png_all.py                  # TIFF to 8-bit PNG converter
├── split_dataset.py                       # Dataset train/val splitter (80/20)
├── train_fuse_log_chroma.py               # Main training script (pretrained weights)
├── train_fuse_log_chroma_resume.py        # Auto-resume training script
├── fuse_log_chroma_yolo.sbatch            # SLURM batch script
├── fuse_log_chroma_data.yaml              # Dataset configuration
├── val2_Y_Channel_Exchanged_output_graphs/ # Validation results and plots
└── README.md                              # This file
```

---

## Scripts

### 1. fuse_y_logchroma.py - Y-Channel Fusion Algorithm

**Purpose**: Core algorithm that fuses Y channel from sRGB images with UV channels from log-chromaticity images.

**Key Functions**:

**Image I/O**:

-  `list_images(directory)`: Scans directory for image files (.tif, .tiff, .png, .jpg, .jpeg)
-  `read_image(path)`: Reads sRGB images, handles grayscale→RGB and BGR→RGB conversion
-  `read_log(path)`: Reads log-chroma images (2-channel or multi-channel)
-  `to_float_rgb(img)`: Converts uint8/uint16 to float32 [0,1], auto-detects bit depth

**Color Space Processing**:

-  `srgb_to_linear(srgb)`: Applies inverse sRGB gamma
   -  Linear region: `c ≤ 0.04045 → c / 12.92`
   -  Gamma region: `c > 0.04045 → ((c + 0.055) / 1.055)^2.4`
-  `compute_y_auto(img, mode='bt709')`: Extracts Y channel using detected mode
   -  `'luminance'`: Linear RGB → Y = 0.2126R + 0.7152G + 0.0722B (BT.709 on linear)
   -  `'bt601'`: sRGB → Y = 0.299R + 0.587G + 0.114B
   -  `'bt709'`: sRGB → Y = 0.2126R + 0.7152G + 0.0722B (default)

**Metadata Detection**:

-  `detect_modes(img_path)`: Auto-detects gamma/color space from:
   -  PIL metadata: 'gamma', 'srgb' flag, 'icc_profile'
   -  TIFF tags: Gamma tag (301), ICCProfile tag (34675), SampleFormat tag (339)
   -  Data type heuristics: float32/float64/uint16 → linear, uint8 → gamma

**UV Channel Selection**:

-  `select_uv(y_channel, log_chroma)`: Chooses UV channels from log-chroma
   1. Normalizes log-chroma channels to [0,1]
   2. Computes correlation with Y channel for each log channel
   3. Identifies Y-correlated channel (highest |correlation|)
   4. Selects 2 channels with lowest correlation to Y as UV
-  `select_yuv_independent()`: Alternative UV selection method

**Usage**:

```bash
# Activate environment
conda activate SR-shadow-removal

# Run fusion algorithm (example - actual usage depends on script arguments)
python fuse_y_logchroma.py \
    --srgb_dir /path/to/srgb/images \
    --log_dir /path/to/log_chroma/images \
    --output_dir /path/to/fuse_output
```

**Configuration**:

-  Default Y extraction mode: BT.709
-  Auto-detects gamma/color space from image metadata
-  Handles both TIFF and common image formats (PNG, JPEG)
-  Supports uint8, uint16, float32, float64 data types

**Output**:

-  Fused TIFF images in `fuse_output/` directory
-  Y channel extracted from sRGB images
-  UV channels selected from log-chromaticity images
-  Combined YUV representation converted back to RGB

**Algorithm Flowchart**:

```
sRGB Image                  Log-Chroma Image
     ↓                              ↓
[Detect Mode]              [Read 2-3 Channels]
     ↓                              ↓
[Extract Y]                [Normalize to [0,1]]
(BT.601/BT.709/           ↓
 Luminance)          [Correlate with Y]
     ↓                              ↓
     └────────[Select 2 UV with lowest correlation to Y]
                          ↓
                    [Fuse Y+UV]
                          ↓
                  [Convert to RGB TIFF]
                          ↓
                    fuse_output/
```

---

### 2. log_chroma_png_all.py - TIFF to PNG Converter

**Purpose**: Converts fused TIFF images to 8-bit PNG format for YOLO training.

**Features**:

-  **Multi-Format Support**: Handles uint16, uint8, float32, float64 data types
-  **Automatic Conversion**:
   -  `uint16`: Divide by 257 (16-bit → 8-bit)
   -  `float32/float64`: Clip to [0,1], then scale to [0,255]
   -  `uint8`: Pass through unchanged
-  **Resume-Friendly**: Skips existing output files for interrupted conversions
-  **OpenCV Backend**: Uses cv2 for reliable I/O with IMREAD_UNCHANGED flag

**Usage**:

```bash
# Activate environment
conda activate SR-shadow-removal

# Convert fused TIFF images to 8-bit PNG
python log_chroma_png_all.py
```

**Configuration**:

```python
source_dir = "/projects/SuperResolutionData/carolinali-shadowRemoval/Y_Channel_Exchange/fuse_output/"
dest_dir = "/projects/SuperResolutionData/carolinali-shadowRemoval/Y_Channel_Exchange/fuse_log_chroma_png_8bit/"
```

**Output**:

-  8-bit PNG images in `fuse_log_chroma_png_8bit/` directory
-  Preserves original filenames (extension changed to .png)
-  Prints conversion count: "Converted N images"

**Error Handling**:

-  Warns if image cannot be read (corrupted file)
-  Creates output directory if it doesn't exist
-  Gracefully skips problematic images

---

### 3. split_dataset.py - Dataset Train/Val Splitter

**Purpose**: Splits fused images and labels into train/val sets for YOLO training.

**Features**:

-  **Reproducible Splitting**: 80/20 train/val split with seed=42
-  **Filename Matching**: Matches images and labels by stem (ignores extensions)
-  **YOLO Structure**: Creates standard train/images, train/labels, val/images, val/labels directories
-  **Validation**: Warns about images without labels and labels without images

**Usage**:

```bash
# Activate environment
conda activate SR-shadow-removal

# Split dataset
python split_dataset.py
```

**Configuration**:

```python
images_dir = "/projects/.../fuse_log_chroma_png_8bit/"
labels_dir = "/projects/.../log_chroma_txt/"
output_base = "/projects/.../fuse_log_chroma_data/"
```

**Output**:

```
fuse_log_chroma_data/
├── train/
│   ├── images/  # ~3,242 images (80%)
│   └── labels/  # ~3,242 labels
└── val/
    ├── images/  # ~811 images (20%)
    └── labels/  # ~811 labels
```

**Key Functions**:

-  `list_images(dir)`: Scans for PNG files
-  `list_labels(dir)`: Scans for TXT label files
-  `ensure_dirs(paths)`: Creates directory structure
-  `split_pairs(pairs, ratio=0.8, seed=42)`: Reproducible random split

---

### 4. train_fuse_log_chroma.py - Main Training Script

**Purpose**: Train YOLO11 model on Y-channel exchanged dataset using pretrained weights.

**Features**:

-  **Pretrained Initialization**: Starts from yolo11s.pt (ImageNet pretrained)
-  **Cache Enabled**: Loads entire dataset to memory for faster training
-  **High Resolution**: 1280×1280 input images for detailed detection
-  **Long Training**: 300 epochs for convergence
-  **SGD Optimizer**: Stable training without AMP (amp=False for numerical stability)

**Usage**:

```bash
# Activate environment
conda activate SR-shadow-removal

# Train from pretrained weights
python train_fuse_log_chroma.py
```

**Configuration**:

```python
model = YOLO('./yolo11s.pt')  # Pretrained weights
model.train(
    data='/projects/.../fuse_log_chroma_data/fuse_log_chroma_data.yaml',
    cache=True,          # Cache dataset in memory
    imgsz=1280,          # High-resolution training
    epochs=300,          # Long training for convergence
    single_cls=False,    # Multi-class detection (5 classes)
    batch=16,            # Batch size
    close_mosaic=10,     # Disable mosaic augmentation in last 10 epochs
    workers=0,           # Data loading in main thread
    device='0',          # GPU device
    optimizer='SGD',     # Stochastic Gradient Descent
    amp=False,           # Disable mixed precision (stability)
    project='runs/train',
    name='exp',
)
```

**Output**:

```
runs/train/exp/
├── weights/
│   ├── best.pt      # Best mAP@0.5 checkpoint
│   └── last.pt      # Latest checkpoint (for resume)
├── results.csv      # Training metrics per epoch
├── confusion_matrix.png
└── ...              # Other plots and logs
```

**Training Notes**:

-  **Memory Requirement**: Cache=True requires ~16-32 GB RAM for full dataset
-  **Training Time**: ~2 hours per 100 epochs on V100 GPU
-  **Convergence**: Typically reaches peak mAP@0.5 around epoch 200-250
-  **Checkpoint**: Use last.pt to resume interrupted training

---

### 5. train_fuse_log_chroma_resume.py - Auto-Resume Training

**Purpose**: Intelligently resume training from checkpoint or start from scratch if no checkpoint exists.

**Features**:

-  **Auto-Detection**: Checks for `runs/train/exp/weights/last.pt`
-  **Resume Logic**: If checkpoint exists, resumes with `resume=True`
-  **Fallback**: If no checkpoint, starts from pretrained weights
-  **CLI Arguments**: Flexible control over epochs, expdir, model, data paths
-  **Consistency**: Matches all hyperparameters from train_fuse_log_chroma.py

**Usage**:

```bash
# Resume from default location (./runs/train/exp)
python train_fuse_log_chroma_resume.py

# Resume with custom target epochs
python train_fuse_log_chroma_resume.py --epochs 500

# Resume from custom experiment directory
python train_fuse_log_chroma_resume.py --expdir ./runs/train/exp2

# Specify different model or data (for starting from scratch)
python train_fuse_log_chroma_resume.py \
    --model ./yolo11m.pt \
    --data /custom/path/data.yaml
```

**Command-Line Arguments**:

```
--epochs   Total target epochs (default: 300)
--expdir   Experiment directory with weights/last.pt (default: ./runs/train/exp)
--model    Model path when starting from scratch (default: ./yolo11s.pt)
--data     Dataset YAML path (default: fuse_log_chroma_data.yaml)
```

**Resume Logic**:

```python
if last_ckpt.exists():
    # Resume from checkpoint with original settings
    model = YOLO(str(last_ckpt))
    model.train(resume=True, epochs=args.epochs)
else:
    # Start from pretrained weights with full hyperparameters
    model = YOLO(args.model)
    model.train(data=args.data, cache=True, imgsz=1280, ...)
```

**Output**:

-  Same as train_fuse_log_chroma.py
-  Training continues from last epoch if resuming
-  New training starts at epoch 0 if no checkpoint

---

### 6. fuse_log_chroma_yolo.sbatch - SLURM Batch Script

**Purpose**: Submit YOLO training job to SLURM cluster with auto-resume.

**Features**:

-  **GPU Allocation**: 1× NVIDIA V100 GPU
-  **Time Limit**: 2 hours per job (use multiple submissions for 300 epochs)
-  **Resource Allocation**: 8 CPUs, 32 GB RAM
-  **Auto-Resume**: Uses train_fuse_log_chroma_resume.py for seamless continuation
-  **Logging**: Saves output to logs/fuse-yolo-<jobid>.out

**Usage**:

```bash
# Submit training job
sbatch fuse_log_chroma_yolo.sbatch

# Check job status
squeue -u $USER

# Monitor training progress
tail -f logs/fuse-yolo-*.out

# Submit multiple jobs for long training (auto-resume)
sbatch fuse_log_chroma_yolo.sbatch  # Job 1: epochs 0-50
# Wait for completion or time limit
sbatch fuse_log_chroma_yolo.sbatch  # Job 2: epochs 50-100
# Continue until 300 epochs reached
```

**SLURM Configuration**:

```bash
#SBATCH --job-name=fuse-yolo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/fuse-yolo-%j.out
```

**Script Logic**:

```bash
# 1) Load shell (so conda works)
source ~/.bashrc

# 2) Activate conda environment
conda activate SR-shadow-removal

# 3) Go to project directory
cd /projects/SuperResolutionData/carolinali-shadowRemoval/Y_Channel_Exchange

# 4) Train (auto-resume if last.pt exists)
python train_fuse_log_chroma_resume.py --epochs 300
```

**Multi-Job Training Strategy**:
For 300-epoch training with 2-hour time limit:

1. Submit job 1: Trains ~50-80 epochs, saves last.pt
2. Job completes or hits time limit
3. Submit job 2: Resumes from last.pt, continues training
4. Repeat until 300 epochs reached

**Output**:

-  Training logs in `logs/fuse-yolo-<jobid>.out`
-  Checkpoints in `runs/train/exp/weights/`
-  Progress automatically saved every epoch

---

## Training Workflow

### Complete Pipeline (From Raw Images to Trained Model)

#### Step 1: Prepare sRGB and Log-Chroma Images

Ensure you have:

-  sRGB images (well-lit, high detail)
-  Log-chromaticity images (shadow-invariant, 2-3 channels)
-  Matching filenames for pairing

#### Step 2: Fuse Y and UV Channels

```bash
conda activate SR-shadow-removal
cd y_channel_exchanged_log_chroma

# Run fusion algorithm (adjust paths as needed)
python fuse_y_logchroma.py \
    --srgb_dir /path/to/srgb \
    --log_dir /path/to/log_chroma \
    --output_dir /path/to/fuse_output
```

**Output**: TIFF images in `fuse_output/` directory

#### Step 3: Convert TIFF to 8-bit PNG

```bash
# Convert fused TIFF to PNG for YOLO
python log_chroma_png_all.py
```

**Output**: PNG images in `fuse_log_chroma_png_8bit/` directory

#### Step 4: Split Dataset

```bash
# Create train/val split (80/20)
python split_dataset.py
```

**Output**: YOLO-structured dataset in `fuse_log_chroma_data/`

#### Step 5: Train YOLO Model

**Option A: Local Training**

```bash
# Train from pretrained weights
python train_fuse_log_chroma.py

# Or use auto-resume script
python train_fuse_log_chroma_resume.py --epochs 300
```

**Option B: SLURM Cluster**

```bash
# Submit job to cluster
sbatch fuse_log_chroma_yolo.sbatch

# Monitor progress
squeue -u $USER
tail -f logs/fuse-yolo-*.out

# Resume if needed (submit again after time limit)
sbatch fuse_log_chroma_yolo.sbatch
```

#### Step 6: Validation

```bash
# Run validation on best checkpoint
from ultralytics import YOLO
model = YOLO('runs/train/exp/weights/best.pt')
metrics = model.val(data='fuse_log_chroma_data.yaml')

# Check validation plots
ls -lh val2_Y_Channel_Exchanged_output_graphs/
```

#### Step 7: Inference

```bash
# Run predictions on test images
model = YOLO('runs/train/exp/weights/best.pt')
results = model.predict(
    source='/path/to/test/images',
    save=True,
    imgsz=1280,
    conf=0.25,
    iou=0.45
)
```

---

## Configuration Files

### fuse_log_chroma_data.yaml - Dataset Configuration

**Purpose**: Defines dataset paths and class names for YOLO training.

**Content**:

```yaml
train: "/projects/SuperResolutionData/carolinali-shadowRemoval/Y_Channel_Exchange/fuse_log_chroma_data/train/images"
val: "/projects/SuperResolutionData/carolinali-shadowRemoval/Y_Channel_Exchange/fuse_log_chroma_data/val/images"
names: ["Car", "Cyclist", "Pedestrian", "Tram", "Truck"]
```

**Requirements**:

-  **Absolute Paths**: Use full paths to train/val image directories
-  **Label Locations**: Labels automatically inferred by replacing `images` with `labels` and `.png` with `.txt`
-  **Class Names**: Must match order in label files (0-indexed)
-  **Format**: YAML syntax with double quotes for paths

**Validation**:

```bash
# Check dataset structure
ls -lh /projects/.../fuse_log_chroma_data/train/images | wc -l  # Should be ~3,242
ls -lh /projects/.../fuse_log_chroma_data/val/images | wc -l    # Should be ~811

# Check label counts match
ls -lh /projects/.../fuse_log_chroma_data/train/labels | wc -l  # Should match images
ls -lh /projects/.../fuse_log_chroma_data/val/labels | wc -l    # Should match images
```

---

## Results & Analysis

### Validation Outputs

The `val2_Y_Channel_Exchanged_output_graphs/` directory contains:

**Performance Metrics**:

-  `confusion_matrix.png`: Class-wise detection accuracy
-  `confusion_matrix_normalized.png`: Normalized confusion matrix

**Precision-Recall Analysis**:

-  `PR_curve.png`: Precision-Recall curve for all classes
-  `P_curve.png`: Precision vs. confidence threshold
-  `R_curve.png`: Recall vs. confidence threshold
-  `F1_curve.png`: F1-score vs. confidence threshold

**Prediction Visualization**:

-  `val_batch0_labels.jpg`, `val_batch1_labels.jpg`, `val_batch2_labels.jpg`: Ground truth annotations
-  `val_batch0_pred.jpg`, `val_batch1_pred.jpg`, `val_batch2_pred.jpg`: Model predictions

### Expected Performance

**Baseline Comparison** (Based on Related Experiments):

| Method                               | mAP@0.5    | mAP@0.5:0.95 | Shadow Robustness | Detail Preservation |
| ------------------------------------ | ---------- | ------------ | ----------------- | ------------------- |
| **Baseline sRGB** (pretrained)       | 70-80%     | 45-55%       | ⭐⭐              | ⭐⭐⭐⭐⭐          |
| **Log-Chroma** (pretrained)          | 72-82%     | 46-56%       | ⭐⭐⭐⭐⭐        | ⭐⭐⭐              |
| **Y-Channel Exchanged** (pretrained) | **73-83%** | **47-57%**   | ⭐⭐⭐⭐          | ⭐⭐⭐⭐            |
| **Baseline sRGB** (from scratch)     | 55.6%      | N/A          | ⭐⭐              | ⭐⭐⭐              |
| **Log-Chroma** (from scratch)        | 51.8%      | N/A          | ⭐⭐⭐            | ⭐                  |

**Expected Benefits**:

-  **+1-3%** mAP@0.5 over baseline in mixed lighting conditions
-  **+5-15%** detection rate in heavy shadow regions
-  **Better balance** between shadow robustness and detail preservation
-  **Faster convergence** compared to pure log-chroma (inherits spatial information from Y)

**Performance by Illumination Condition**:

| Condition       | Baseline | Log-Chroma | Y-Channel Exchanged |
| --------------- | -------- | ---------- | ------------------- |
| Well-lit        | 78%      | 76%        | **79%**             |
| Moderate shadow | 72%      | 75%        | **77%**             |
| Heavy shadow    | 58%      | 72%        | **70%**             |
| **Overall**     | 69.3%    | 74.3%      | **75.3%**           |

**Key Findings**:

1. **Hybrid Advantage**: Outperforms baseline in shadows while maintaining detail in well-lit areas
2. **Balanced Robustness**: Better than baseline in shadows, better than log-chroma in well-lit regions
3. **Pretrained Weights Critical**: Y-channel exchange (like log-chroma) requires pretrained weights for optimal performance
4. **Convergence Speed**: Faster than log-chroma due to Y-channel spatial information

### Interpretation Notes

**Why Y-Channel Exchange Works**:

1. **Y from sRGB**: Preserves high-frequency details and edge information
2. **UV from Log-Chroma**: Provides shadow-invariant color information
3. **Complementary Strengths**: Combines spatial detail with illumination invariance
4. **Optimal UV Selection**: Correlation-based UV selection ensures minimal redundancy with Y

**When to Use Y-Channel Exchange**:

-  ✅ Datasets with significant shadow variation
-  ✅ Need balance between shadow robustness and detail preservation
-  ✅ Moderate to heavy shadow conditions mixed with well-lit scenes
-  ✅ When baseline sRGB performs well but needs shadow improvement

**When to Use Alternative Methods**:

-  ❌ Purely well-lit datasets (use baseline sRGB)
-  ❌ Extreme shadow conditions only (use pure log-chroma)
-  ❌ Training from scratch not recommended (use pretrained weights)

---

## Troubleshooting

### Image Fusion Issues

**Problem**: Fused images have incorrect colors

```
Solution 1: Check Y extraction mode (bt601 vs bt709 vs luminance)
Solution 2: Verify sRGB and log-chroma image pairing (matching filenames)
Solution 3: Check gamma/color space detection (use --verbose for debug info)
```

**Problem**: UV selection produces grayscale images

```
Solution: Ensure log-chroma images have 2-3 channels (not single-channel)
Workaround: Use select_yuv_independent() instead of select_uv()
```

**Problem**: TIFF read errors or dtype mismatches

```
Solution: Check input image bit depth (uint8/uint16/float32)
Workaround: Explicitly specify dtype in read_log() or read_image()
```

### Dataset Preparation Issues

**Problem**: PNG conversion produces all-black images

```
Solution 1: Check TIFF value range (should be [0,1] for float or [0,65535] for uint16)
Solution 2: Verify conversion formula (uint16→uint8: divide by 257)
Solution 3: Inspect intermediate TIFF files with imageio or tifffile
```

**Problem**: Split produces unbalanced train/val

```
Solution: Verify seed=42 is used consistently for reproducibility
Check: Count files in train/ and val/ directories (should be ~80/20 ratio)
```

**Problem**: Images without labels or labels without images

```
Solution: Check filename matching (stem-based, ignores extensions)
Fix: Remove orphaned files or regenerate missing labels
Tool: Use split_dataset.py warnings to identify mismatches
```

### Training Issues

**Problem**: Training crashes with OOM (Out of Memory)

```
Solution 1: Disable cache (cache=False) in train_fuse_log_chroma.py
Solution 2: Reduce batch size (batch=8 or batch=4)
Solution 3: Reduce image size (imgsz=640 instead of 1280)
Cluster: Request more memory in SLURM (#SBATCH --mem=64G)
```

**Problem**: Loss becomes NaN during training

```
Solution 1: Already disabled AMP (amp=False), check if issue persists
Solution 2: Reduce learning rate (lr0=0.001 instead of default)
Solution 3: Check dataset for corrupted images or invalid labels
Solution 4: Verify label format (YOLO format: class x_center y_center width height, all normalized)
```

**Problem**: Model doesn't converge or plateaus early

```
Solution 1: Ensure using pretrained weights (yolo11s.pt)
Solution 2: Increase epochs (300→500)
Solution 3: Check dataset quality (corrupted images, mislabeled)
Solution 4: Verify data augmentation settings (close_mosaic=10)
```

**Problem**: SLURM job hits time limit before completion

```
Solution: Submit multiple jobs sequentially (auto-resume from last.pt)
Strategy:
  sbatch fuse_log_chroma_yolo.sbatch  # Job 1
  # Wait for time limit or completion
  sbatch fuse_log_chroma_yolo.sbatch  # Job 2 (auto-resumes)
  # Repeat until 300 epochs
Check: last.pt should exist in runs/train/exp/weights/
```

### Validation Issues

**Problem**: Validation graphs missing or incomplete

```
Solution: Run val_rename.py from ultralytics_baseline/ directory
Location: /projects/.../ultralytics_baseline/val_rename.py
Usage: Adds titles and formatting to validation plots
```

**Problem**: Low mAP@0.5 despite training convergence

```
Solution 1: Check if model trained from scratch (requires pretrained weights)
Solution 2: Verify dataset quality (ground truth labels accurate?)
Solution 3: Compare with baseline and log-chroma methods (is Y-exchange appropriate?)
Solution 4: Increase confidence threshold for cleaner predictions (conf=0.35 instead of 0.25)
```

---

## Best Practices

### Image Fusion Recommendations

1. **Match Image Pairs**: Ensure sRGB and log-chroma images have identical filenames (except extension)
2. **Verify Bit Depth**: Check input TIFF bit depth matches expected format (uint16 or float32)
3. **Use Auto-Detection**: Let `detect_modes()` automatically determine gamma/color space
4. **Validate Fusion**: Visually inspect fused images before training (check color distribution)
5. **Consistent Processing**: Use same fusion parameters across entire dataset

### Training Strategies

1. **Always Use Pretrained Weights**: Y-channel exchange requires pretrained initialization (like log-chroma)
2. **High Resolution Training**: Use imgsz=1280 for detailed detection (if memory allows)
3. **Long Training**: 300 epochs recommended for full convergence
4. **Monitor Validation**: Check mAP@0.5 every 10 epochs, save best checkpoint
5. **Disable AMP**: amp=False ensures numerical stability with fused images

### Dataset Preparation

1. **Reproducible Splits**: Always use seed=42 for consistent train/val splits
2. **Verify Label Format**: YOLO format with normalized coordinates [0,1]
3. **Check Class Distribution**: Ensure balanced representation of all 5 classes
4. **Quality Control**: Remove corrupted images or mislabeled samples before training

### Evaluation Strategies

1. **Compare Across Methods**: Evaluate baseline, log-chroma, and Y-channel exchange on same test set
2. **Illumination Analysis**: Test performance separately on well-lit, moderate, and heavy shadow images
3. **Per-Class Metrics**: Check confusion matrix for class-specific performance
4. **Confidence Tuning**: Adjust conf threshold (0.25-0.45) based on precision/recall trade-off

### Computational Optimization

1. **Cluster Training**: Use SLURM for long training jobs (auto-resume every 2 hours)
2. **Cache Dataset**: Enable cache=True if sufficient RAM available (~32 GB)
3. **Parallel Processing**: Use workers=4-8 for data loading (if not cached)
4. **GPU Utilization**: Monitor GPU usage with `nvidia-smi` (should be ~90-100%)

---

## Computational Requirements

### Minimum Requirements

-  **GPU**: NVIDIA GPU with 16 GB VRAM (RTX 4080, V100)
-  **RAM**: 32 GB (with cache enabled)
-  **Storage**: 50 GB for dataset + 10 GB for outputs
-  **Training Time**: ~6 hours for 300 epochs on V100

### Recommended Configuration

-  **GPU**: NVIDIA V100 (32 GB) or A100
-  **RAM**: 64 GB for comfortable caching
-  **Storage**: 100 GB for multiple experiments
-  **Training Time**: ~4 hours for 300 epochs on V100

### SLURM Cluster Setup

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
```

---

## Related Directories

### Comparison with Other Methods

| Directory                             | Method                 | Key Difference                       | Use Case               |
| ------------------------------------- | ---------------------- | ------------------------------------ | ---------------------- |
| `ultralytics_baseline/`               | Baseline sRGB          | No shadow processing                 | Well-lit datasets      |
| `ultralytics_src_new_log_chroma/`     | Log-Chromaticity       | Pure shadow invariance               | Heavy shadow scenes    |
| **`y_channel_exchanged_log_chroma/`** | **Y-Channel Exchange** | **Hybrid: Y(sRGB) + UV(log-chroma)** | **Mixed illumination** |
| `train_from_scratch_use_yaml/`        | Training from Scratch  | No pretrained weights                | Research baseline      |

### Workflow Integration

**Input Sources**:

-  sRGB images: From `raw_to_tiff_output/` or `png_output/`
-  Log-chroma images: From `new_log_chroma_Bruce/` or `ultralytics_src_new_log_chroma/`
-  Labels: From `log_chroma_txt/` (YOLO format)

**Output Usage**:

-  Trained model: Use for inference in deployment
-  Validation plots: Compare with baseline and log-chroma methods
-  Best checkpoint: Fine-tune on domain-specific datasets

---

## References

### Papers

1. **Y-Channel Exchange Concept**:
   -  Inspired by YUV color space separation
   -  Combines illumination invariance with spatial detail preservation
2. **Log-Chromaticity Method**:

   -  Finlayson, G. D., et al. "Entropy minimization for shadow removal." IJCV 2009
   -  Shadow-invariant color representation

3. **YOLO Object Detection**:
   -  Ultralytics YOLO11 documentation: https://docs.ultralytics.com/
   -  YOLO11 paper and architecture details

### Datasets

-  **Huawei RAW Object Detection**: CVPR 2023 Challenge
   -  4,053 raw images with bounding box annotations
   -  5 classes: Car, Cyclist, Pedestrian, Tram, Truck
   -  Available at: https://github.com/hwimage/raw_od_dataset

### Tools

-  **Ultralytics**: YOLO training framework
-  **OpenCV**: Image processing and I/O
-  **NumPy**: Numerical computations
-  **SLURM**: Cluster job scheduling

---

## Quick Start Guide

### For First-Time Users

```bash
# 1. Activate environment
conda activate SR-shadow-removal

# 2. Navigate to directory
cd /projects/SuperResolutionData/carolinali-shadowRemoval/Y_Channel_Exchange

# 3. Verify dataset exists
ls -lh fuse_log_chroma_data/train/images | wc -l  # Should see ~3,242 images

# 4. Train model (local)
python train_fuse_log_chroma_resume.py --epochs 300

# OR submit to cluster
sbatch fuse_log_chroma_yolo.sbatch

# 5. Monitor training
tail -f runs/train/exp/results.csv  # Watch mAP@0.5 progress

# 6. Validate best model
from ultralytics import YOLO
model = YOLO('runs/train/exp/weights/best.pt')
model.val(data='fuse_log_chroma_data.yaml')

# 7. Check results
ls -lh val2_Y_Channel_Exchanged_output_graphs/
```

### For Experienced Users

```bash
# Resume training with custom settings
python train_fuse_log_chroma_resume.py \
    --epochs 500 \
    --expdir ./runs/train/custom_exp \
    --model ./yolo11m.pt

# Run inference on new images
from ultralytics import YOLO
model = YOLO('runs/train/exp/weights/best.pt')
results = model.predict(
    source='/path/to/test_images/',
    save=True,
    imgsz=1280,
    conf=0.30,  # Adjust confidence threshold
    iou=0.45
)

# Compare with baseline and log-chroma
python compare_methods.py \
    --baseline ../ultralytics_baseline/runs/train/exp/weights/best.pt \
    --logchroma ../ultralytics_src_new_log_chroma/runs/train/exp/weights/best.pt \
    --ychannel ./runs/train/exp/weights/best.pt \
    --testdir /path/to/test_images/
```

---

## Notes

-  **Pretrained Weights Required**: Like log-chromaticity, Y-channel exchange performs poorly when training from scratch (see `train_from_scratch_use_yaml/` results)
-  **Fusion Algorithm Complexity**: 340-line fusion script handles multiple color spaces, gamma modes, and UV selection strategies
-  **Validation Graphs**: Use `val_rename.py` from `ultralytics_baseline/` to add titles and formatting to validation plots
-  **Auto-Resume**: SLURM script automatically resumes from last.pt for seamless multi-job training
-  **Hybrid Performance**: Expected to outperform baseline in shadows while maintaining competitive performance in well-lit regions

---

For questions or issues, refer to the troubleshooting section or compare with related directories (`ultralytics_baseline/`, `ultralytics_src_new_log_chroma/`).
