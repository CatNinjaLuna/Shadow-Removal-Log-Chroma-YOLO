# Custom Training Data Preparation

This directory contains utilities for preparing and organizing log-chromaticity training data, including TIFF-to-PNG conversion, file batching, and directory management.

---

## Overview

These scripts handle post-processing of log-chromaticity images for efficient storage, transfer, and training workflows. The primary tasks include:

1. Converting 16-bit TIFF log-chroma images to 8-bit PNG for compatibility
2. Batching large datasets into manageable chunks (<1 GB each)
3. Organizing files across multiple directories

---

## Scripts

### 1. `seperate_log_chroma_files.py`

**Purpose:** Multi-function script that performs TIFF-to-PNG conversion and batches both TIFF and PNG files into <1 GB chunks.

**Main Functions:**

#### a) TIFF to 8-bit PNG Conversion

Converts 16-bit log-chromaticity TIFFs to 8-bit PNGs for easier handling and visualization.

**Conversion Method:**

-  16-bit → 8-bit: `value_8bit = round(value_16bit / 257.0)`
-  Maintains visual appearance while reducing file size
-  Supports both grayscale and RGB images

**Usage:**

```bash
python seperate_log_chroma_files.py \
    --tiff_src /path/to/tiff_input \
    --png_dst /path/to/png_output \
    --overwrite
```

**Default Paths:**

-  **Input:** `/projects/.../my_training_data/log_chroma_tiff_4k`
-  **Output:** `/projects/.../my_training_data/log_chroma_8_bit_png`

#### b) File Batching

Splits large directories of TIFF or PNG files into batches of <1 GB for easier transfer and storage.

**Batch Creation Logic:**

-  Sorts files alphabetically
-  Groups files until batch size approaches 1 GB limit (1,000,000,000 bytes)
-  Creates numbered batch directories: `batch_001/`, `batch_002/`, etc.
-  Copies (or moves) files to respective batch folders

**Usage:**

```bash
# Batch TIFFs
python seperate_log_chroma_files.py \
    --batch_tiff \
    --tiff_src /path/to/tiffs \
    --tiff_batch_root /path/to/tiff_batches

# Batch PNGs
python seperate_log_chroma_files.py \
    --batch_png \
    --png_src /path/to/pngs \
    --png_batch_root /path/to/png_batches
```

**Default Batch Locations:**

-  **TIFF Batches:** `/projects/.../my_training_data/log_chroma_tiff_batches/`
-  **PNG Batches:** `/projects/.../my_training_data/log_chroma_png_batches/`

**Batch Size Limit:** 1,000,000,000 bytes (~0.93 GiB)

#### c) All-in-One Processing

Process complete pipeline in one command:

```bash
python seperate_log_chroma_files.py --all
```

**Pipeline Steps:**

1. Convert TIFFs → 8-bit PNGs
2. Batch original TIFFs
3. Batch generated PNGs

---

### 2. `log_chroma_png_all.py`

**Purpose:** Consolidates PNG files from multiple batch directories into a single unified directory.

**Use Case:**

-  Combine PNG batches back into one directory for training
-  Merge files from distributed storage
-  Flatten hierarchical batch structure

**Features:**

-  Recursive search through source directory
-  Automatic duplicate name handling (appends `_1`, `_2`, etc.)
-  Progress tracking with file count
-  Safe move operations (preserves originals until confirmed)

**Usage:**

```bash
python log_chroma_png_all.py
```

**Default Configuration:**

```python
src = "/projects/.../my_training_data/log_chroma_png_batches"
dst = "/projects/.../my_training_data/log_chroma_8_bit_png"
```

**Behavior:**

-  Recursively finds all `.png` files in source directory
-  Moves files to destination (flattened structure)
-  Handles filename collisions with automatic renaming
-  Prints total number of files moved

**Duplicate Handling Example:**

```
image.png          → image.png
batch_02/image.png → image_1.png
batch_03/image.png → image_2.png
```

---

## Data Organization

### Directory Structure (Cluster)

```
my_training_data/
├── 00Train/
│   └── [legacy training data]
│
├── log_chroma_tiff_4k/
│   └── [~4000 16-bit TIFF log-chroma images]
│
├── log_chroma_16_bit_tiff/
│   └── [alternative 16-bit TIFF storage]
│
├── log_chroma_output/
│   └── [raw log-chroma output from processing pipeline]
│
├── log_chroma_8_bit_png/
│   └── [consolidated 8-bit PNG images for training]
│
├── log_chroma_tiff_batches/
│   ├── batch_001/
│   ├── batch_002/
│   └── ...
│
└── log_chroma_png_batches/
    ├── batch_001/
    ├── batch_002/
    └── ...
```

### Cluster Paths

| Directory        | Path                                                     | Purpose                   |
| ---------------- | -------------------------------------------------------- | ------------------------- |
| **00Train**      | `/projects/.../my_training_data/00Train`                 | Legacy training data      |
| **16-bit TIFFs** | `/projects/.../my_training_data/log_chroma_16_bit_tiff`  | Original log-chroma TIFFs |
| **8-bit PNGs**   | `/projects/.../my_training_data/log_chroma_8_bit_png`    | Converted PNG images      |
| **Raw Output**   | `/projects/.../my_training_data/log_chroma_output`       | Pipeline output           |
| **PNG Batches**  | `/projects/.../my_training_data/log_chroma_png_batches`  | Batched PNG files         |
| **TIFF Batches** | `/projects/.../my_training_data/log_chroma_tiff_batches` | Batched TIFF files        |

---

## Typical Workflows

### Workflow 1: TIFF to PNG Conversion

```bash
# Convert all TIFFs to 8-bit PNGs
python seperate_log_chroma_files.py \
    --tiff_src /projects/.../log_chroma_tiff_4k \
    --png_dst /projects/.../log_chroma_8_bit_png
```

**Output:** ~4000 8-bit PNG files ready for visualization or training

---

### Workflow 2: Create Batches for Transfer

```bash
# Batch TIFFs for archival/transfer
python seperate_log_chroma_files.py \
    --batch_tiff \
    --tiff_src /projects/.../log_chroma_tiff_4k \
    --tiff_batch_root /projects/.../log_chroma_tiff_batches

# Batch PNGs for distributed storage
python seperate_log_chroma_files.py \
    --batch_png \
    --png_src /projects/.../log_chroma_8_bit_png \
    --png_batch_root /projects/.../log_chroma_png_batches
```

**Output:** Multiple batch directories, each <1 GB

---

### Workflow 3: Consolidate Batches

```bash
# Merge all PNG batches back into single directory
python log_chroma_png_all.py
```

**Output:** All PNGs in one directory (`log_chroma_8_bit_png/`)

---

### Workflow 4: Complete Pipeline

```bash
# One command for full processing
python seperate_log_chroma_files.py --all
```

**Performs:**

1. TIFF → PNG conversion
2. TIFF batching
3. PNG batching

---

## Technical Details

### TIFF to PNG Conversion

**Bit Depth Transformation:**

```python
# 16-bit to 8-bit scaling
img_8 = (img_16.astype(np.float32) / 257.0).round().astype(np.uint8)
```

**Why 257?**

-  Preserves intensity distribution
-  Maps 0-65535 → 0-255 with minimal loss
-  Maintains relative brightness

**File Size Reduction:**

-  16-bit TIFF: ~10-20 MB per image
-  8-bit PNG: ~2-5 MB per image
-  Compression ratio: ~4:1 to 5:1

### Batching Strategy

**Size Calculation:**

```python
BATCH_SIZE_LIMIT = 1_000_000_000  # bytes
```

**Batching Algorithm:**

1. Sort files alphabetically
2. Accumulate file sizes
3. When batch approaches limit, create new batch
4. Copy/move files to batch directory
5. Reset accumulator for next batch

**Benefits:**

-  Easier file transfers (network limits)
-  Better organization for archival
-  Parallel processing on multiple machines
-  Compliance with storage quotas

---

## Command-Line Arguments

### `seperate_log_chroma_files.py`

```
--tiff_src PATH          : Source directory with TIFF files
--png_dst PATH           : Destination for PNG files
--tiff_batch_root PATH   : Root for TIFF batches
--png_batch_root PATH    : Root for PNG batches
--overwrite              : Overwrite existing PNGs
--batch_tiff             : Enable TIFF batching
--batch_png              : Enable PNG batching
--all                    : Run full pipeline
```

**Example:**

```bash
python seperate_log_chroma_files.py \
    --tiff_src /custom/tiff/path \
    --png_dst /custom/png/path \
    --overwrite \
    --all
```

---

## Performance

### Processing Speed

**TIFF to PNG Conversion:**

-  ~100-200 images per minute (CPU)
-  Depends on image resolution and CPU cores
-  Progress logged every 100 images

**Batching:**

-  Fast (file copying/moving)
-  Limited by disk I/O speed
-  ~1000 files per minute on SSD

### Memory Usage

**TIFF to PNG:**

-  Loads one image at a time
-  Peak memory: ~50-100 MB per image
-  Suitable for standard cluster nodes

**Batching:**

-  Minimal memory usage (<100 MB)
-  Only file metadata processed
-  No image loading required

---

## Troubleshooting

**Common Issues:**

1. **"Could not read TIFF file"**

   -  Check file permissions
   -  Verify TIFF file integrity
   -  Ensure OpenCV TIFF support is installed

2. **"Failed to write PNG"**

   -  Check disk space in destination
   -  Verify write permissions
   -  Ensure valid output path

3. **"Batch exceeds size limit"**

   -  Individual files >1 GB are placed in single-file batches
   -  Adjust `BATCH_SIZE_LIMIT` if needed
   -  Consider compressing large files

4. **Duplicate filenames in consolidation:**
   -  Script automatically handles with `_1`, `_2` suffix
   -  Check source for genuine duplicates
   -  Review naming conflicts in logs

---

## Notes

**File Format Support:**

-  **Input:** `.tif`, `.tiff` (16-bit)
-  **Output:** `.png` (8-bit)
-  OpenCV handles bit depth conversion

**Cluster Considerations:**

-  Scripts designed for headless cluster operation
-  No GUI dependencies
-  Progress logging to stdout
-  Safe for SLURM job submission

**Data Preservation:**

-  Original TIFFs remain unchanged
-  PNG conversion creates copies
-  Batching can use move (destructive) or copy (safe)

---

## Related Directories

-  `../log_chroma_shadow_removal_method/` - Log-chroma generation pipeline
-  `../data_src_rename_tif_labels/` - Label conversion and dataset splitting
-  `../split_and_viz_data/` - Data visualization utilities

---

## Integration with Training Pipeline

**After these scripts:**

1. Rename files (if needed): `../data_src_rename_tif_labels/rename_tif_files.py`
2. Convert labels: `../data_src_rename_tif_labels/convert_labels_to_yolo.py`
3. Split dataset: `../data_src_rename_tif_labels/split_dataset.py`
4. Train YOLO: Use consolidated PNG or TIFF images

**Recommended Format for Training:**

-  YOLO training: Use 8-bit PNGs (faster loading)
-  High-fidelity evaluation: Use 16-bit TIFFs (preserves precision)
-  Visualization: Use 8-bit PNGs (easier to view)
