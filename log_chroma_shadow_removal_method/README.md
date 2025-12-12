# Log-Chromaticity Shadow Removal Method

This directory contains the core implementation for generating illumination-invariant log-chromaticity representations from linear TIFF images using the Illuminant Spectral Direction (ISD) method with U-Net predictions.

---

## Overview

The log-chromaticity shadow removal method projects 3D log-RGB images onto a 2D plane orthogonal to the estimated illuminant direction, effectively removing shadow intensity variations while preserving chromatic information essential for object detection.

**Key Concept:**

-  **ISD (Illuminant Spectral Direction):** Unit vector representing the illumination direction in log-RGB space
-  **Log-Chromaticity Plane:** 2D subspace perpendicular to ISD, invariant to intensity changes
-  **U-Net Model:** Predicts ISD maps from 16-bit linear TIFF images

---

## Pipeline

```
16-bit Linear TIFF
       ↓
U-Net ISD Prediction
       ↓
Log-RGB Transform
       ↓
ISD Plane Projection (anchor: 10.4)
       ↓
2D Log-Chromaticity Image (16-bit TIFF)
       ↓
YOLO Training & Prediction
```

---

## Scripts

### 1. `run.py` - Interactive GUI Processor (Development/Visualization)

Full-featured processor with OpenCV GUI for interactive visualization and parameter tuning.

**Purpose:**

-  Visual debugging of ISD predictions
-  ROI-based analysis
-  Comparison with gamma correction
-  Research and development tool

**Features:**

-  **GUI Controls:** Interactive region selection
-  **Real-time Visualization:** ISD maps, log-chroma output, gamma-corrected comparisons
-  **Manual Adjustment:** Fine-tune anchor points and projection parameters
-  **Logging:** Detailed processing logs

**Usage:**

```bash
python run.py
```

**Configuration (edit in script):**

```python
RAW_IMAGE_DIR = "/projects/.../first_11_raw_files"
SR_MAP_DIR = "/projects/.../raw_to_tiff_output_test_1"
LOG_CHROMA_OUTPUT_DIR = "/projects/.../log_chroma_output_test_1"
```

**Controls:**

-  Mouse selection for ROI
-  Keyboard shortcuts for processing steps
-  Interactive parameter adjustment

**Not Recommended for:** Batch processing on compute cluster (requires display)

---

### 2. `run_log_chroma_batch.py` - Batch Processing (Cluster Production)

Non-interactive batch processor for converting entire datasets on compute clusters.

**Purpose:**

-  High-throughput log-chroma generation
-  Cluster-friendly (no GUI dependencies)
-  Production pipeline for training data

**Configuration:**

```python
INPUT_DIR = "/projects/.../raw_to_tiff_output"
OUTPUT_DIR = "/projects/.../log_chroma_output"
MODEL_PATH = "/projects/.../UNET_run_x10_01_last_model.pth"

DEVICE = "cpu"          # or "cuda" for GPU
ANCHOR = 10.4           # log-space projection anchor
SAVE_PREVIEW = True     # save 8-bit PNG for QC
FACTOR = 32             # UNet dimension requirement
```

**Usage:**

```bash
python run_log_chroma_batch.py
```

**Features:**

-  Automatic directory traversal
-  Progress tracking
-  Error handling and logging
-  Optional 8-bit preview generation (PNG)
-  Memory-efficient processing

**Output:**

-  Primary: 16-bit TIFF log-chromaticity images
-  Secondary: 8-bit PNG previews (if enabled)

**Processing Logic:**

1. Load 16-bit linear TIFF
2. Pad to UNet-compatible dimensions (divisible by 32)
3. Predict ISD map with U-Net
4. Compute log-chromaticity projection
5. Save results with original filename + `_log_chroma.tif`

---

### 3. `run_log_chroma_resume.py` - Resumable Batch Processing

Identical to batch processor but supports resuming from specific file in case of interruption.

**Purpose:**

-  Resume interrupted batch jobs
-  Skip already-processed files
-  Handle large datasets with checkpointing

**Configuration:**

```python
START_FROM_NAME = "day-05039.tif"  # Resume from this file (inclusive)
```

**Usage:**

```bash
python run_log_chroma_resume.py
```

**Use Cases:**

-  Cluster job timeout recovery
-  Network interruption handling
-  Selective reprocessing

**Behavior:**

-  Alphabetically sorts all TIFF files
-  Skips files before `START_FROM_NAME`
-  Processes from `START_FROM_NAME` onward
-  Same output format as `run_log_chroma_batch.py`

---

### 4. `predict_log_chroma.py` - YOLO Inference on Log-Chroma Images

Runs trained log-chromaticity YOLO model on log-chroma TIFF images for object detection.

**Purpose:**

-  Inference with log-chroma trained models
-  Generate predictions on log-chroma test sets
-  Evaluate shadow-invariant detection performance

**Configuration:**

```python
LOGC_WEIGHTS = Path("/projects/.../exp_log_chroma/weights/best.pt")
LOGCHROMA_TIFF_DIR = Path("/projects/.../output_yolo_all/log_chroma")
```

**Inference Settings:**

```python
imgsz=1280              # Image size for inference
conf=0.5                # Confidence threshold
iou=0.6                 # NMS IoU threshold
```

**Usage:**

```bash
python predict_log_chroma.py
```

**Output:**

-  Annotated images: `runs/predict_log_chroma/exp_log_chroma_predict/`
-  Bounding box visualizations
-  Optional: YOLO txt format labels (set `save_txt=True`)

**Cluster Paths:**

-  **Predictions:** `/projects/.../log_chroma_shadow_removed_method/runs/predict_log_chroma/exp_log_chroma_predict`
-  **Weights:** `/projects/.../log_chroma_weights`

---

### 5. `convert_log_chroma_to_jpg.py` - Label Conversion for Log-Chroma Images

Converts JSON annotations to YOLO format specifically for log-chroma JPG/PNG images (excludes TIFF).

**Purpose:**

-  Convert LabelMe JSON to YOLO format
-  Handle JPG/PNG log-chroma derivatives
-  Support alternative image formats

**Supported Formats:**

-  `.jpg`, `.jpeg`, `.png` (NO TIFF support in this script)

**Usage:**

```bash
python convert_log_chroma_to_jpg.py \
    --label_dir /path/to/json_labels \
    --image_dir /path/to/log_chroma_jpgs \
    --out_dir /path/to/yolo_labels
```

**Features:**

-  Reads actual image dimensions with cv2
-  Converts rectangles/polygons to YOLO bounding boxes
-  Normalizes coordinates to [0, 1]
-  Auto-detects class names from JSON files

**Note:** For TIFF log-chroma images, use scripts in `../data_src_rename_tif_labels/`

---

### 6. `unet_models2.py` - U-Net Architecture for ISD Prediction

Defines ResNet50-based U-Net architecture for predicting Illuminant Spectral Direction (ISD) maps.

**Architecture Components:**

#### **ResNet50UNet**

-  **Encoder:** Pretrained ResNet50 backbone (ImageNet)
-  **Decoder:** Upsampling blocks with skip connections
-  **Output:** 3-channel ISD map (unit vectors in log-RGB space)

**Key Features:**

-  **SE Blocks (Squeeze-and-Excitation):** Channel attention mechanism
-  **Dropout:** Configurable dropout for regularization
-  **Skip Connections:** Multi-scale feature fusion
-  **Flexible Input:** Supports various input resolutions (padded to multiples of 32)

**Model Initialization:**

```python
model = ResNet50UNet(
    in_channels=3,
    out_channels=3,
    pretrained=True,
    se_block=True,
    dropout=0.0
)
```

**Architecture Details:**

```
Input: 3 x H x W (16-bit linear RGB)
Encoder Stages:
  - Stage 1: 64 channels (H/2 x W/2)
  - Stage 2: 256 channels (H/4 x W/4)
  - Stage 3: 512 channels (H/8 x W/8)
  - Stage 4: 1024 channels (H/16 x W/16)
  - Bottleneck: 2048 channels (H/32 x W/32)

Decoder Stages:
  - UpBlock 1: 1024 → 512 (H/16 x W/16)
  - UpBlock 2: 512 → 256 (H/8 x W/8)
  - UpBlock 3: 256 → 128 (H/4 x W/4)
  - UpBlock 4: 128 → 64 (H/2 x W/2)
  - Final: 64 → 3 (H x W) - ISD map
```

**Output Normalization:**

-  ISD vectors normalized to unit length
-  Ensures valid direction representation in log-RGB space

**Training:**

-  Model weights: `UNET_run_x10_01_last_model.pth`
-  Trained on paired linear TIFF images and ground-truth ISD maps
-  Loss: Cosine similarity or angular loss

---

## Log-Chromaticity Mathematics

### Transformation Steps

1. **Linear to Log-RGB:**

   ```
   L = log(I)  where I is linear 16-bit image
   ```

2. **Illuminant Direction Projection:**

   ```
   L_shifted = L - anchor
   dot_product = <L_shifted, ISD>
   projection = dot_product · ISD
   ```

3. **Plane Projection:**

   ```
   L_plane = L_shifted - projection + anchor
   ```

4. **Back to Linear:**
   ```
   I_chroma = exp(L_plane)
   ```

**Anchor Point:** `10.4` (empirically determined for ROD dataset)

**Invariance Properties:**

-  Removes multiplicative intensity scaling (shadows)
-  Preserves chromatic ratios
-  Maintains object boundaries

---

## Data Locations (Cluster)

**Input:**

-  Linear 16-bit TIFFs: `/projects/SuperResolutionData/carolinali-shadowRemoval/raw_to_tiff_output`

**Models:**

-  U-Net weights: `/projects/SuperResolutionData/carolinali-shadowRemoval/model/UNET_run_x10_01_last_model.pth`

**Output:**

-  Log-chroma TIFFs: `/projects/SuperResolutionData/carolinali-shadowRemoval/log_chroma_output`
-  8-bit previews: Same directory as TIFFs (optional PNG)

**YOLO Training:**

-  Renamed log-chroma: `/projects/SuperResolutionData/carolinali-shadowRemoval/output_yolo_all/log_chroma`
-  Train/val split: `/projects/SuperResolutionData/carolinali-shadowRemoval/data/log_chroma_splitted_data`

**Predictions:**

-  YOLO outputs: `/projects/SuperResolutionData/carolinali-shadowRemoval/log_chroma_shadow_removed_method/runs/predict_log_chroma/`
-  Trained weights: `/projects/SuperResolutionData/carolinali-shadowRemoval/log_chroma_weights`

---

## Typical Workflow

### Full Pipeline Example

```bash
# Step 1: Generate log-chroma images (batch processing)
python run_log_chroma_batch.py

# Step 2: Rename files (remove _log_chroma suffix)
cd ../data_src_rename_tif_labels
python rename_tif_files.py

# Step 3: Convert labels to YOLO format
python convert_labels_to_yolo.py

# Step 4: Split dataset
python split_dataset.py

# Step 5: Train YOLO on log-chroma images
yolo detect train \
    data=data_log_chroma.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=1280

# Step 6: Run predictions
cd ../log_chroma_shadow_removal_method
python predict_log_chroma.py
```

---

## Technical Details

### Image Processing

**Bit Depth Handling:**

-  Input: 16-bit unsigned integer (0-65535)
-  Processing: 32-bit float for log operations
-  Output: 16-bit unsigned integer (rescaled)

**Padding Strategy:**

-  U-Net requires dimensions divisible by 32
-  Images padded with reflection mode
-  Output cropped to original size

**Memory Management:**

-  Processes one image at a time
-  GPU memory cleared after each iteration
-  CPU mode available for memory-constrained environments

### Model Device Support

**CPU Mode:**

```python
DEVICE = "cpu"
```

-  Slower but works on any machine
-  Suitable for small datasets

**GPU Mode:**

```python
DEVICE = "cuda"
```

-  Requires CUDA-compatible GPU
-  10-50x speedup depending on hardware
-  Recommended for batch processing

---

## Performance Considerations

**Processing Time (per image):**

-  CPU: ~5-10 seconds per 1920x1280 image
-  GPU (single GPU): ~0.5-1 second per image

**Bottlenecks:**

-  U-Net inference (85% of processing time)
-  Log/exp operations (10%)
-  File I/O (5%)

**Optimization Strategies:**

-  Batch inference (process multiple images simultaneously)
-  Mixed precision (FP16 on GPU)
-  Pre-compute ISD maps (if model is fixed)

---

## Troubleshooting

**Common Issues:**

1. **CUDA Out of Memory:**

   -  Switch to CPU mode
   -  Reduce batch size
   -  Process smaller images

2. **Dimension Errors:**

   -  Ensure FACTOR = 32
   -  Check input image dimensions
   -  Verify padding logic

3. **Black/White Output Images:**

   -  Check anchor point (should be ~10.4)
   -  Verify ISD map normalization
   -  Inspect input image bit depth

4. **Model Loading Fails:**
   -  Verify model path exists
   -  Check PyTorch version compatibility
   -  Ensure model architecture matches checkpoint

---

## Related Directories

-  `../raw_to_tiff_output/` - Input linear TIFF images
-  `../data_src_rename_tif_labels/` - Label conversion and splitting
-  `../my_training_data/` - Custom training data preparation
-  `../new_log_chroma_Bruce/` - Alternative log-chroma implementation

---

## References

**ISD Method:**

-  Finlayson, G.D., et al. "Illuminant and device invariant colour using histogram equalisation"
-  Maxwell, B.A., et al. "Illumination-Invariant Color Object Recognition via Compressed Chromaticity Histograms"

**U-Net Architecture:**

-  Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
-  ResNet: He, K., et al. "Deep Residual Learning for Image Recognition"
