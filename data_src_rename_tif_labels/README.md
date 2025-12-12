# Data Processing: TIF Renaming and Label Conversion

This directory contains utility scripts for processing log-chromaticity TIFF images and converting annotations for YOLO training. These scripts handle file renaming, label format conversion, dataset splitting, and visualization.

---

## Overview

This pipeline is designed to:

1. Rename log-chroma TIFF files by removing suffix identifiers
2. Convert JSON annotations to YOLO format
3. Split datasets into train/validation sets
4. Visualize bounding box annotations

**Use Case:** Preparing log-chromaticity images for YOLO object detection training.

---

## Scripts

### 1. `rename_tif_files.py`

Renames log-chroma TIFF files by removing the `_log_chroma` suffix from filenames.

**Purpose:**

-  Standardize filenames for consistency with label files
-  Prepare log-chroma images for YOLO training pipeline
-  Maintain original files while creating renamed copies

**Usage:**

```bash
python rename_tif_files.py \
    --source_dir /path/to/log_chroma_output \
    --target_dir /path/to/renamed_output
```

**Default Paths (Cluster):**

-  **Source:** `/projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_output`
-  **Target:** `/projects/SuperResolutionData/carolinali-shadowRemoval/output_yolo_all/log_chroma`

**Behavior:**

-  Copies files (preserves originals)
-  Only processes `.tif` files containing `_log_chroma` in filename
-  Example: `image001_log_chroma.tif` → `image001.tif`
-  Prints processing summary with counts

**Error Handling:**

-  Creates target directory if it doesn't exist
-  Skips files without `_log_chroma` suffix
-  Reports processing errors for individual files

---

### 2. `convert_labels_to_yolo (1).py`

Converts JSON annotation files (from LabelMe or similar tools) to YOLO format.

**Purpose:**

-  Transform polygon/rectangle annotations to YOLO bounding boxes
-  Generate normalized coordinates for YOLO training
-  Extract class names automatically from JSON files

**YOLO Format:**

```
<class_id> <x_center> <y_center> <width> <height>
```

All values normalized to [0, 1] range.

**Usage:**

```bash
python "convert_labels_to_yolo (1).py" \
    --label_dir /path/to/json_labels \
    --image_dir /path/to/images \
    --out_dir /path/to/yolo_labels
```

**Features:**

-  **Auto-class detection:** Scans JSON files to extract all class names
-  **Multiple image formats:** Supports .tif, .tiff, .png, .jpg, .jpeg
-  **Size validation:** Reads actual image dimensions for accurate normalization
-  **Bounding box conversion:** Handles rectangles and polygons (converts to bounding boxes)

**Output:**

-  One `.txt` file per image with YOLO annotations
-  Prints detected classes: `['Car', 'Cyclist', 'Pedestrian', 'Tram', 'Truck']`

**Coordinate Normalization:**

-  Centers and dimensions normalized to image width/height
-  Ensures boxes stay within [0, 1] bounds
-  Handles edge cases (zero-width boxes, out-of-bounds coordinates)

---

### 3. `split_dataset.py`

Splits image-label pairs into training and validation sets with reproducible random sampling.

**Purpose:**

-  Create train/val splits for YOLO training
-  Ensure paired images and labels are split together
-  Support reproducible experiments with seed control

**Usage:**

```bash
python split_dataset.py \
    --images /path/to/images \
    --labels /path/to/labels \
    --out /path/to/output \
    --ratio 0.8 \
    --seed 42
```

**Default Configuration (Cluster):**

-  **Images:** `/projects/SuperResolutionData/carolinali-shadowRemoval/output_yolo_all/log_chroma`
-  **Labels:** `/projects/SuperResolutionData/carolinali-shadowRemoval/data/log_chroma_txt`
-  **Output:** `/projects/SuperResolutionData/carolinali-shadowRemoval/data/log_chroma_splitted_data`
-  **Train/Val Ratio:** 80/20
-  **Random Seed:** 42

**Output Structure:**

```
output_dir/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

**Features:**

-  **Pair matching:** Only splits files with both image and label
-  **Random shuffling:** Ensures diverse train/val splits
-  **Reproducibility:** Fixed seed for consistent splits across runs
-  **Format flexibility:** Supports multiple image formats (.tif, .png, .jpg)

**Processing Summary:**

```
split done: total 1000 train 800 val 200
```

---

### 4. `visualize_labels.py`

Visualizes YOLO bounding box annotations overlaid on images.

**Purpose:**

-  Quality control: verify label accuracy
-  Debug annotation errors
-  Generate annotated images for documentation

**Usage:**

```bash
python visualize_labels.py \
    --images /path/to/images \
    --labels /path/to/yolo_labels \
    --out /path/to/visualizations
```

**Features:**

-  **16-bit TIFF support:** Automatically converts 16-bit TIFFs to 8-bit for visualization
-  **Color-coded boxes:** Different colors for different classes
-  **Class labels:** Optional text labels on bounding boxes
-  **Batch processing:** Processes all image-label pairs in directories

**Visualization Details:**

-  **Bounding boxes:** Drawn with class-specific colors
-  **Line width:** Adjustable for visibility
-  **Image formats:** Handles TIFF (16-bit/8-bit), PNG, JPG
-  **Color palette:** 6 distinct colors for classes

**Color Scheme:**

```python
colors = [
    (255,0,0),    # Red
    (0,255,0),    # Green
    (0,0,255),    # Blue
    (255,255,0),  # Yellow
    (255,0,255),  # Magenta
    (0,255,255)   # Cyan
]
```

**16-bit TIFF Handling:**

-  Automatically detects bit depth
-  Converts uint16 → uint8 by dividing by 257
-  Preserves visual appearance
-  Supports both grayscale and RGB TIFFs

---

## Typical Workflow

### Full Pipeline Example

```bash
# Step 1: Rename log-chroma TIFF files
python rename_tif_files.py \
    --source_dir /projects/.../log_chroma_output \
    --target_dir /projects/.../output_yolo_all/log_chroma

# Step 2: Convert JSON labels to YOLO format
python "convert_labels_to_yolo (1).py" \
    --label_dir /projects/.../json_labels \
    --image_dir /projects/.../output_yolo_all/log_chroma \
    --out_dir /projects/.../log_chroma_txt

# Step 3: Visualize labels (optional - quality check)
python visualize_labels.py \
    --images /projects/.../output_yolo_all/log_chroma \
    --labels /projects/.../log_chroma_txt \
    --out /projects/.../visualized_labels

# Step 4: Split into train/val sets
python split_dataset.py \
    --images /projects/.../output_yolo_all/log_chroma \
    --labels /projects/.../log_chroma_txt \
    --out /projects/.../log_chroma_splitted_data \
    --ratio 0.8 \
    --seed 42
```

---

## Data Locations (Cluster)

**Input:**

-  Log-chroma TIFF images: `/projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_output`
-  JSON labels: `/projects/SuperResolutionData/driving/ROD_dataset/Dataset/LinearFiles/dataset`

**Intermediate:**

-  Renamed TIFFs: `/projects/SuperResolutionData/carolinali-shadowRemoval/output_yolo_all/log_chroma`
-  YOLO labels: `/projects/SuperResolutionData/carolinali-shadowRemoval/data/log_chroma_txt`

**Output:**

-  Train/val split: `/projects/SuperResolutionData/carolinali-shadowRemoval/data/log_chroma_splitted_data`

---

## Classes

**Object Detection Classes:**

-  Car
-  Cyclist
-  Pedestrian
-  Tram
-  Truck

Class IDs are assigned alphabetically:

```
0: Car
1: Cyclist
2: Pedestrian
3: Tram
4: Truck
```

---

## Notes

**File Matching:**

-  Scripts match images and labels by stem name (filename without extension)
-  Mismatched pairs are automatically skipped
-  Case-insensitive extension matching

**Image Format Support:**

-  Primary: 16-bit TIFF (log-chroma output)
-  Supported: .tif, .tiff, .png, .jpg, .jpeg
-  Auto-detection of bit depth

**Reproducibility:**

-  Use `--seed` parameter for consistent train/val splits
-  Seed 42 is used by default
-  Important for comparing experiments

**Error Handling:**

-  Scripts continue on individual file errors
-  Summary statistics printed at completion
-  Missing files are skipped with warnings

---

## Related Directories

-  `../split_and_viz_data/` - Similar scripts for baseline sRGB data
-  `../log_chroma_shadow_removal_method/` - Log-chroma generation pipeline
-  `../my_training_data/` - Custom training data preparation

---

## Troubleshooting

**Common Issues:**

1. **No matched pairs found:**

   -  Check that image and label filenames match (excluding extensions)
   -  Verify both directories contain files
   -  Ensure label format is correct (.txt for YOLO, .json for conversion)

2. **16-bit TIFF visualization fails:**

   -  Ensure PIL/Pillow is installed with TIFF support
   -  Check file permissions and disk space
   -  Verify TIFF files are valid (not corrupted)

3. **Class names incorrect:**

   -  Run label conversion first to see detected classes
   -  Check JSON files for correct 'label' fields
   -  Ensure consistent naming across all annotations

4. **Memory issues with large datasets:**
   -  Process in batches
   -  Use lower resolution images for visualization
   -  Consider streaming/iterator approaches for very large datasets
