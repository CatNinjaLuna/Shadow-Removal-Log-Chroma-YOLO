# Dataset Splitting & Visualization Utilities

This directory contains utilities for converting annotations, splitting datasets, and visualizing YOLO labels for both baseline (sRGB) and log-chromaticity training data.

---

## Overview

This module provides a complete pipeline for preparing annotated object detection datasets for YOLO training:

1. **Label Format Conversion:** JSON (LabelMe) → YOLO format (normalized bounding boxes)
2. **Filename Standardization:** Remove suffix artifacts from log-chromaticity filenames
3. **Dataset Splitting:** Create train/validation splits with reproducible random seeds
4. **Label Visualization:** Draw bounding boxes on images for quality assurance

**Use Cases:**

-  Preparing Huawei ROD dataset annotations for YOLO training
-  Converting between image formats (TIFF → JPG) while maintaining labels
-  Visual inspection of annotation quality
-  Ensuring train/val splits preserve image-label pairing

---

## Directory Structure

```
split_and_viz_data/
├── convert_labels_to_yolo.py       # JSON → YOLO (TIFF images)
├── convert_labels_to_yolo_jpg.py   # JSON → YOLO (JPG images, strict checking)
├── rename_tif_files.py             # Remove "_log_chroma" suffix from filenames
├── split_dataset_log_chroma.py     # Train/val split with reproducibility
├── visualize_labels.py             # Draw boxes on TIFF images
├── visualize_labels_jpg.py         # Draw boxes on JPG images
├── data.yaml                       # YOLO dataset configuration
└── README.md                       # This file
```

---

## Scripts

### 1. `convert_labels_to_yolo.py`

**Purpose:** Convert LabelMe JSON annotations to YOLO format for TIFF images.

**Features:**

-  Reads LabelMe JSON files with rectangle shapes
-  Extracts bounding box coordinates and class labels
-  Normalizes coordinates to [0, 1] range
-  Auto-detects image size from actual files or JSON metadata
-  Supports multiple image formats (`.tif`, `.tiff`, `.png`, `.jpg`)

**YOLO Format:**

```
<class_id> <center_x> <center_y> <width> <height>
```

All coordinates normalized to [0, 1] relative to image dimensions.

**Usage:**

```bash
python convert_labels_to_yolo.py \
    --label_dir /path/to/json/labels \
    --image_dir /path/to/images \
    --out_dir /path/to/yolo_txt \
    --exts .tif .png .jpg
```

**Arguments:**

-  `--label_dir`: Directory containing LabelMe JSON files
-  `--image_dir`: Directory containing corresponding images
-  `--out_dir`: Output directory for YOLO `.txt` files
-  `--exts`: Supported image extensions (default: `.tif`, `.png`, `.jpg`, `.jpeg`)

**Default Paths (Cluster):**

```python
label_dir:  "/projects/.../data/label"
image_dir:  "/projects/.../output_yolo_all/log_chroma"
out_dir:    "/projects/.../data/annotated_log_chroma_txt"
```

**Class Mapping:**

```python
classes = ['Car', 'Cyclist', 'Pedestrian', 'Tram', 'Truck']
# Mapped to indices: 0, 1, 2, 3, 4
```

**Coordinate Transformation:**

```python
# LabelMe JSON: [[x1, y1], [x2, y2]] (absolute pixels)
# YOLO format: [center_x, center_y, width, height] (normalized)

center_x = (x1 + x2) / 2 / image_width
center_y = (y1 + y2) / 2 / image_height
width = abs(x2 - x1) / image_width
height = abs(y2 - y1) / image_height
```

**Clipping:** All coordinates clipped to [0, 1] to handle boundary cases.

---

### 2. `convert_labels_to_yolo_jpg.py`

**Purpose:** Convert JSON annotations to YOLO format for **JPG images** with strict size validation.

**Key Difference from `convert_labels_to_yolo.py`:**

-  **Strict Size Checking:** Raises error if JSON metadata size ≠ actual image size
-  **Validation Logic:** Ensures annotations were created for the correct image resolution
-  **Use Case:** Converting labels for JPG versions of TIFF images where resolution might change

**Error Handling:**

```python
if h_json > 0 and w_json > 0 and (h_json != h_img or w_json != w_img):
    raise ValueError(
        f"Size mismatch for {base}: "
        f"json=({w_json}x{h_json}), image=({w_img}x{h_img})"
    )
```

**Why This Matters:**

-  TIFF → JPG conversion may change resolution (e.g., downsampling)
-  Annotations drawn on 1280×1280 TIFFs may not match 640×640 JPGs
-  Prevents incorrect bounding box scaling

**Usage:**

```bash
python convert_labels_to_yolo_jpg.py \
    --label_dir /path/to/json/labels \
    --image_dir /path/to/jpg/images \
    --out_dir /path/to/yolo_txt_jpg
```

**Default Paths:**

```python
label_dir:  "/projects/.../data/label"
image_dir:  "/projects/.../data/log_chroma_unsplitted_jpg"
out_dir:    "/projects/.../data/log_chroma_jpg_txt"
```

---

### 3. `rename_tif_files.py`

**Purpose:** Remove `_log_chroma` suffix from TIFF filenames for consistency with original dataset naming.

**Problem:**

-  Log-chromaticity pipeline outputs: `day-02000_log_chroma.tif`
-  YOLO training expects: `day-02000.tif`
-  Label files reference: `day-02000.json`
-  Mismatch causes train/val split to fail

**Solution:**

```python
"day-02000_log_chroma.tif" → "day-02000.tif"
```

**Features:**

-  Copies (not moves) files to new directory
-  Preserves original timestamps
-  Skips files without `_log_chroma` in name
-  Reports processed/skipped counts

**Usage:**

```bash
python rename_tif_files.py \
    --source_dir /path/to/log_chroma_output \
    --target_dir /path/to/renamed_output
```

**Arguments:**

-  `--source_dir`: Directory with `*_log_chroma.tif` files
-  `--target_dir`: Output directory for renamed files

**Default Paths:**

```python
source_dir: "/projects/.../my_training_data/log_chroma_output"
target_dir: "/projects/.../output_yolo_all/log_chroma"
```

**Example Output:**

```
Processed: day-02000_log_chroma.tif -> day-02000.tif
Processed: day-02001_log_chroma.tif -> day-02001.tif
...
Process completed
Numbers of files processed: 4053
Skipping 0 files
```

---

### 4. `split_dataset_log_chroma.py`

**Purpose:** Split image-label pairs into train/validation sets with reproducible random shuffling.

**Features:**

-  **Reproducible Splits:** Fixed random seed (default: 42)
-  **Maintains Pairing:** Only includes images with corresponding labels
-  **Format Agnostic:** Supports TIFF, PNG, JPG images
-  **YOLO Directory Structure:** Creates `train/images`, `train/labels`, `val/images`, `val/labels`

**Algorithm:**

```python
1. List all images: {stem: path}
2. List all labels: {stem: path}
3. Find intersection: stems present in both
4. Shuffle with seed
5. Split by ratio: train_set[:N], val_set[N:]
6. Copy files to output structure
```

**Usage:**

```bash
python split_dataset_log_chroma.py \
    --images /path/to/images \
    --labels /path/to/yolo_labels \
    --out /path/to/split_output \
    --ratio 0.8 \
    --seed 42
```

**Arguments:**

-  `--images`: Directory containing images (any supported format)
-  `--labels`: Directory containing YOLO `.txt` labels
-  `--out`: Output directory for split dataset
-  `--ratio`: Train ratio (0.8 = 80% train, 20% val)
-  `--seed`: Random seed for reproducibility

**Default Paths:**

```python
images: "/projects/.../output_yolo_all/log_chroma"
labels: "/projects/.../data/annotated_log_chroma_txt"
out:    "/projects/.../data/split"
ratio:  0.8
seed:   42
```

**Output Structure:**

```
split/
├── train/
│   ├── images/
│   │   ├── day-02000.tif
│   │   ├── day-02001.tif
│   │   └── ... (3242 images)
│   └── labels/
│       ├── day-02000.txt
│       ├── day-02001.txt
│       └── ... (3242 labels)
└── val/
    ├── images/
    │   ├── day-05000.tif
    │   └── ... (811 images)
    └── labels/
        ├── day-05000.txt
        └── ... (811 labels)
```

**Example Output:**

```bash
split done: total 4053 train 3242 val 811
```

---

### 5. `visualize_labels.py`

**Purpose:** Draw YOLO bounding boxes on TIFF images for visual quality assurance.

**Features:**

-  **16-bit TIFF Support:** Converts uint16 → uint8 for visualization
-  **Color-Coded Classes:** Each object class gets unique color
-  **Class Labels:** Text annotations on bounding boxes
-  **Batch Processing:** Processes all matched image-label pairs

**Supported Image Formats:**

-  16-bit TIFF (converted to 8-bit for display)
-  8-bit PNG, JPG
-  Multi-channel RGB or grayscale

**Color Mapping:**

```python
colors = [
    (255,0,0),     # Class 0: Car (Red)
    (0,255,0),     # Class 1: Cyclist (Green)
    (0,0,255),     # Class 2: Pedestrian (Blue)
    (255,255,0),   # Class 3: Tram (Yellow)
    (255,0,255),   # Class 4: Truck (Magenta)
]
```

**Usage:**

```bash
python visualize_labels.py \
    --images /path/to/images \
    --labels /path/to/yolo_txt \
    --out /path/to/visualizations \
    --names Car Cyclist Pedestrian Tram Truck
```

**Arguments:**

-  `--images`: Directory containing images
-  `--labels`: Directory containing YOLO `.txt` labels
-  `--out`: Output directory for visualization PNGs
-  `--names`: Class names in order (index 0, 1, 2, ...)

**Default Paths:**

```python
images: "/projects/.../output_yolo_all/log_chroma"
labels: "/projects/.../data/log_chroma_txt"
out:    "/projects/.../data/log_chroma_vis"
names:  ['Car', 'Cyclist', 'Pedestrian', 'Tram', 'Truck']
```

**16-bit TIFF Conversion:**

```python
# Convert uint16 [0, 65535] → uint8 [0, 255]
arr8 = (arr_uint16 / 257.0).astype(np.uint8)
```

**Output:**

-  PNG files with bounding boxes drawn
-  Same filename as input (e.g., `day-02000.png`)
-  RGB color images even if input is grayscale

---

### 6. `visualize_labels_jpg.py`

**Purpose:** Visualize YOLO labels on JPG images (same as `visualize_labels.py` but for JPG-specific paths).

**Key Differences:**

-  Default paths point to JPG directories
-  Optimized for 8-bit images (no TIFF conversion overhead)

**Default Paths:**

```python
images: "/projects/.../data/log_chroma_unsplitted_jpg"
labels: "/projects/.../data/log_chroma_jpg_txt"
out:    "/projects/.../data/log_chroma_jpg_vis"
```

**Usage:**

```bash
python visualize_labels_jpg.py \
    --images /path/to/jpg/images \
    --labels /path/to/yolo_txt \
    --out /path/to/visualizations
```

---

### 7. `data.yaml`

**Purpose:** YOLO dataset configuration file specifying train/val paths and class names.

**Content:**

```yaml
train: "/projects/.../data/baseline_splitted_data/train/images"
val: "/projects/.../data/baseline_splitted_data/val/images"
names: ["Car", "Cyclist", "Pedestrian", "Tram", "Truck"]
```

**Usage in YOLO Training:**

```bash
yolo detect train \
    data=data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=1280
```

**Customization:**

-  Update `train` and `val` paths to match your split dataset location
-  Ensure `names` list matches class order in YOLO labels (index-based)

---

## Complete Workflow

### Pipeline 1: TIFF Log-Chromaticity Dataset

```bash
# Step 1: Remove "_log_chroma" suffix from filenames
python rename_tif_files.py \
    --source_dir /projects/.../log_chroma_output \
    --target_dir /projects/.../output_yolo_all/log_chroma

# Step 2: Convert JSON labels to YOLO format
python convert_labels_to_yolo.py \
    --label_dir /projects/.../data/label \
    --image_dir /projects/.../output_yolo_all/log_chroma \
    --out_dir /projects/.../data/annotated_log_chroma_txt

# Step 3: Split into train/val sets
python split_dataset_log_chroma.py \
    --images /projects/.../output_yolo_all/log_chroma \
    --labels /projects/.../data/annotated_log_chroma_txt \
    --out /projects/.../data/log_chroma_splitted_data_tiff \
    --ratio 0.8 \
    --seed 42

# Step 4: Visualize labels for QA
python visualize_labels.py \
    --images /projects/.../output_yolo_all/log_chroma \
    --labels /projects/.../data/annotated_log_chroma_txt \
    --out /projects/.../data/log_chroma_vis

# Step 5: Train YOLO (update data.yaml first)
yolo detect train \
    data=data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=1280
```

### Pipeline 2: JPG Dataset (8-bit sRGB)

```bash
# Step 1: Convert TIFF to JPG (use external tool or PIL)
# (Not included in this directory)

# Step 2: Convert JSON labels to YOLO with size validation
python convert_labels_to_yolo_jpg.py \
    --label_dir /projects/.../data/label \
    --image_dir /projects/.../data/log_chroma_unsplitted_jpg \
    --out_dir /projects/.../data/log_chroma_jpg_txt

# Step 3: Split dataset
python split_dataset_log_chroma.py \
    --images /projects/.../data/log_chroma_unsplitted_jpg \
    --labels /projects/.../data/log_chroma_jpg_txt \
    --out /projects/.../data/log_chroma_splitted_data_jpg \
    --ratio 0.8 \
    --seed 42

# Step 4: Visualize
python visualize_labels_jpg.py \
    --images /projects/.../data/log_chroma_unsplitted_jpg \
    --labels /projects/.../data/log_chroma_jpg_txt \
    --out /projects/.../data/log_chroma_jpg_vis
```

---

## Data Locations (Cluster)

**Input Data:**

```
/projects/SuperResolutionData/carolinali-shadowRemoval/data/label/
    # Original LabelMe JSON annotations
```

**Log-Chromaticity TIFF:**

```
/projects/.../my_training_data/log_chroma_output/
    # Raw log-chroma outputs with "_log_chroma" suffix

/projects/.../output_yolo_all/log_chroma/
    # Renamed TIFFs without suffix

/projects/.../data/annotated_log_chroma_txt/
    # YOLO format labels

/projects/.../data/log_chroma_splitted_data_tiff/
    # Train/val split dataset
```

**Log-Chromaticity JPG:**

```
/projects/.../data/log_chroma_unsplitted_jpg/
    # 8-bit JPG images

/projects/.../data/log_chroma_jpg_txt/
    # YOLO format labels

/projects/.../data/log_chroma_jpg_vis/
    # Visualization outputs
```

**Baseline sRGB:**

```
/projects/.../data/annotated_baseline_txt/
    # YOLO labels for baseline

/projects/.../data/baseline_splitted_data/
    # Train/val split for baseline
```

---

## Class Mapping

**Huawei ROD Dataset Classes:**

| Index | Class Name | Color (Visualization) | Typical Count per Image |
| ----- | ---------- | --------------------- | ----------------------- |
| 0     | Car        | Red (255,0,0)         | 5-20 instances          |
| 1     | Cyclist    | Green (0,255,0)       | 0-5 instances           |
| 2     | Pedestrian | Blue (0,0,255)        | 0-10 instances          |
| 3     | Tram       | Yellow (255,255,0)    | 0-2 instances           |
| 4     | Truck      | Magenta (255,0,255)   | 0-5 instances           |

**Class Distribution (Approximate):**

-  **Car:** ~60% of all annotations
-  **Pedestrian:** ~20%
-  **Cyclist:** ~10%
-  **Truck:** ~7%
-  **Tram:** ~3%

---

## Validation Checklist

After running the pipeline, verify:

-  [ ] **File Counts Match:**

   ```bash
   ls images/*.tif | wc -l  # Should equal number of .txt files
   ls labels/*.txt | wc -l
   ```

-  [ ] **Train/Val Ratio Correct:**

   ```bash
   train_count / (train_count + val_count) ≈ 0.8
   ```

-  [ ] **No Empty Labels:**

   ```bash
   find labels/ -name "*.txt" -size 0  # Should be empty
   ```

-  [ ] **Coordinate Range Valid:**

   ```python
   # All values in [0, 1]
   for line in open('labels/day-02000.txt'):
       values = [float(x) for x in line.split()[1:]]
       assert all(0 <= v <= 1 for v in values)
   ```

-  [ ] **Visualizations Look Correct:**

   -  Bounding boxes align with objects
   -  No boxes off-image or wildly incorrect
   -  Class labels match visual content

-  [ ] **Image-Label Pairing:**
   ```bash
   # Stems should match
   ls train/images/*.tif | xargs -n1 basename | sed 's/.tif//' > /tmp/img_stems
   ls train/labels/*.txt | xargs -n1 basename | sed 's/.txt//' > /tmp/lbl_stems
   diff /tmp/img_stems /tmp/lbl_stems  # Should be empty
   ```

---

## Troubleshooting

### Common Issues

**1. Size Mismatch Error (JPG conversion):**

```
ValueError: Size mismatch for day-02000:
json=(1280x1280), image=(640x640)
```

**Cause:** JSON annotations created on 1280×1280 TIFFs, but JPGs are 640×640  
**Solution:**

-  Re-annotate JPGs at correct resolution
-  Scale bounding boxes proportionally
-  Use TIFFs directly for training

**2. Missing Image-Label Pairs:**

```
split done: total 0 train 0 val 0
```

**Cause:** Filename stems don't match between images and labels  
**Solution:**

-  Check for suffix mismatches (`_log_chroma`)
-  Verify file extensions match expected formats
-  Use `rename_tif_files.py` to standardize names

**3. Empty Visualization Images:**

```
All visualization PNGs are blank
```

**Cause:** YOLO labels in wrong format or coordinate issues  
**Solution:**

-  Check label files are not empty
-  Verify coordinate values in [0, 1] range
-  Ensure class indices are valid (0-4)

**4. TIFF Conversion Artifacts:**

```
Visualizations show blocky/incorrect colors
```

**Cause:** 16-bit → 8-bit conversion overflow  
**Solution:**

```python
# Correct conversion
arr8 = (arr_uint16 / 257.0).astype(np.uint8)

# Not: arr8 = arr_uint16 // 256  (loses precision)
```

**5. Train/Val Imbalance:**

```
split done: total 4053 train 4053 val 0
```

**Cause:** `--ratio` set to 1.0 or rounding error  
**Solution:**

-  Use `--ratio 0.8` for 80/20 split
-  Ensure total count > 10 for meaningful split

---

## Performance Benchmarks

**Label Conversion (4053 images):**

-  Time: ~2 minutes
-  Rate: ~35 files/sec
-  Memory: <500 MB

**Dataset Split (4053 pairs):**

-  Time: ~30 seconds (copying files)
-  Disk Usage: 2× original (copies, not moves)

**Visualization (4053 images):**

-  Time: ~10 minutes
-  Rate: ~7 images/sec
-  Output Size: ~5 MB per PNG (from 10 MB TIFF)

---

## Best Practices

**1. Always Use Fixed Random Seed:**

```python
--seed 42  # Reproducible splits for experiments
```

**2. Validate Before Training:**

-  Inspect 10-20 random visualizations
-  Check class distribution balance
-  Verify bounding box accuracy

**3. Backup Original Data:**

-  `rename_tif_files.py` copies (not moves)
-  Keep original JSON labels untouched
-  Version control `data.yaml` configurations

**4. Consistent File Naming:**

-  Remove all suffixes (`_log_chroma`, `_baseline`, etc.)
-  Use lowercase extensions (`.tif`, not `.TIF`)
-  Match stems exactly between images and labels

**5. Document Split Seeds:**

```python
# experiments/exp001/config.txt
split_seed: 42
split_ratio: 0.8
total_images: 4053
train_count: 3242
val_count: 811
```

---

## Future Improvements

**Potential Enhancements:**

1. **Stratified Splitting:**

   -  Ensure class balance in train/val sets
   -  Maintain temporal distribution (day vs. night scenes)

2. **Data Augmentation:**

   -  Horizontal flipping
   -  Random crops
   -  Color jittering (for sRGB only)

3. **Label Validation:**

   -  Detect overlapping boxes
   -  Flag suspiciously large/small boxes
   -  Check for off-image coordinates

4. **Multi-Format Support:**

   -  Automatic TIFF → JPG conversion
   -  PNG support with transparency
   -  HDR image handling

5. **Statistics Generation:**
   -  Class distribution histograms
   -  Bounding box size analysis
   -  Spatial distribution heatmaps

---

## Related Directories

-  `../raw_to_tiff_output/` – RAW to 16-bit TIFF conversion
-  `../log_chroma_shadow_removal_method/` – ISD generation and log-chroma processing
-  `../my_training_data/` – Batch TIFF to PNG/JPG conversion
-  `../baseline/` – Baseline sRGB training experiments
-  `../data_src_rename_tif_labels/` – Alternative label processing utilities

---

## References

**YOLO Format Specification:**

-  Official Ultralytics docs: [https://docs.ultralytics.com/datasets/detect/](https://docs.ultralytics.com/datasets/detect/)

**LabelMe Annotation Tool:**

-  GitHub: [https://github.com/wkentaro/labelme](https://github.com/wkentaro/labelme)

**Dataset Split Best Practices:**

-  Goodfellow, I., et al. "Deep Learning" Chapter 5: Machine Learning Basics

---

## License

Copyright 2025 Carolina Li

Scripts in this directory are provided for research and educational purposes.
