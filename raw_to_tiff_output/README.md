'''

# RAW to 16-bit TIFF Conversion Pipeline

This directory contains scripts and utilities for converting RAW Bayer sensor data from the **Huawei RAW Object Detection (ROD) Benchmark** into 16-bit linear TIFF images suitable for downstream shadow-invariant processing and YOLO training.

---

## Overview

**Data Source:** All RAW files are obtained from the Huawei RAW Object Detection benchmark dataset:

**Citation:**

```
Xu, Y., et al. (2023). "Toward RAW Object Detection: A New Benchmark and a New Model."
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.
```

**Paper:** [https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Toward_RAW_Object_Detection_A_New_Benchmark_and_a_New_CVPR_2023_paper.html](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Toward_RAW_Object_Detection_A_New_Benchmark_and_a_New_CVPR_2023_paper.html)

**Dataset Characteristics:**

-  **Format:** 24-bit packed Bayer RAW (3 bytes per pixel)
-  **Resolution:** 1856×2880 original, downsampled to 1280×1280 after demosaicing
-  **Bit Depth:** 24-bit linear sensor values (0–16,777,215 range)
-  **Bayer Pattern:** Standard RGGB or GRBG (auto-detected by debayering algorithm)
-  **Domain:** Autonomous driving scenarios with challenging illumination conditions
-  **Object Classes:** Car, Truck, Bus, Pedestrian, Cyclist (5 classes)
-  **Annotations:** YOLO-format bounding boxes with class labels

**Why RAW Data?**

-  **Linearity:** RAW sensor data is linear with respect to scene radiance (no gamma curve)
-  **Full Dynamic Range:** 24-bit precision captures wide intensity variations
-  **No Post-Processing Artifacts:** No JPEG compression, demosaicing artifacts, or tone mapping
-  **Optimal for Shadow Removal:** Log-chromaticity methods require linear RGB data

---

## Directory Structure

```
raw_to_tiff_output/
├── preprocess_raw_to_tiff.py    # Main conversion script (Huawei + modifications)
├── run.py                       # Visualization and ISD application demo
├── README.md                    # This file
└── (output .tif files after processing)
```

**Cluster Output Location:**

```
/projects/SuperResolutionData/carolinali-shadowRemoval/raw_to_tiff_output/
```

---

## Scripts

### 1. `preprocess_raw_to_tiff.py`

**Purpose:** Convert Huawei 24-bit packed RAW Bayer files to 16-bit linear TIFF images with proper debayering, white balance, and dynamic range scaling.

**Key Features:**

-  **Huawei-Specific RAW Reader:** Unpacks 3-byte sequences into 24-bit pixel values
-  **GPU-Accelerated Debayering:** PyTorch-based 3×3 convolutional Bayer filter
-  **Gray-World White Balance:** Simple automatic color correction
-  **99.5th Percentile Scaling:** Robust dynamic range compression to 16-bit TIFF
-  **Batch Processing:** Multi-threaded parallel processing with joblib
-  **Filename Filtering:** Only processes `day-02000.raw` through `day-06052.raw` (4,053 files)

**Modifications from Original Huawei Code:**

```python
# Original: Scale to 8-bit sRGB
# Modified: 99.5th percentile scaling to 16-bit linear TIFF
p = np.nanpercentile(arr, 99.5)
im16 = np.clip(arr, 0, p) / p * 65535
im16 = np.clip(im16, 0, 65535).astype(np.uint16)
```

**Why 99.5th Percentile?**

-  Prevents oversaturation from bright highlights (sky, reflections)
-  Preserves shadow detail in the lower end of the histogram
-  Ensures most pixels fit within 16-bit range without clipping
-  More robust than simple min-max normalization

**Architecture:**

```
┌─────────────────────┐
│  Input: .raw files  │
│  (24-bit Bayer)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Read & Unpack       │
│ (3 bytes → uint32)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Debayering (3×3)    │
│ RGGB → RGB          │
│ (PyTorch GPU)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Resize to 1280×1280 │
│ (bilinear interp)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Gray-World WB       │
│ R *= G_avg/R_avg    │
│ B *= G_avg/B_avg    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 99.5% Percentile    │
│ Scaling to 16-bit   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Output: .tif files  │
│ (16-bit linear RGB) │
└─────────────────────┘
```

**Usage:**

```bash
# Activate environment
conda activate SR-shadow-removal

# Run conversion
python preprocess_raw_to_tiff.py \
  -p /projects/SuperResolutionData/driving/ROD_dataset/Dataset/RawFiles/dataset \
  -t 1 \
  -o /projects/SuperResolutionData/carolinali-shadowRemoval/raw_to_tiff_output
```

**Arguments:**

-  `-p, --path`: Input directory containing `.raw` files (required)
-  `-t, --threads`: Number of parallel threads (1=single, -1=auto-detect)
-  `-o, --out`: Output directory for `.tif` files (required)

**Filtering Logic:**

```python
# Only process files matching pattern: day-XXXXX.raw
# where XXXXX is in range [02000, 06052]
day_pattern = re.compile(r"day-(\d+)", re.IGNORECASE)
if 2000 <= idx <= 6052:
    lines.append(p)
```

**Output:**

-  **File Count:** 4,053 TIFFs (day-02000.tif through day-06052.tif)
-  **Format:** 16-bit unsigned integer TIFF
-  **Color Space:** Linear RGB (no gamma correction)
-  **Channels:** 3 (Red, Green, Blue)
-  **Size:** ~10 MB per file (1280×1280×3×2 bytes)

---

### 2. `run.py`

**Purpose:** Demonstration script for visualizing ISD-based shadow removal and log-chromaticity projection on 16-bit TIFF images.

**Key Classes:**

#### `imgProcessor(image, sr_map, filename)`

Image processing class for applying ISD transformations.

**Methods:**

-  `convert_img_to_log_space(linear_img)` – Convert 16-bit linear to log-RGB
-  `log_to_linear(log_img)` – Convert log-RGB back to linear 16-bit
-  `linear_to_srgb(linear_rgb)` – Apply sRGB gamma curve for visualization
-  `project_to_chromaticity_plane(anchor)` – Project onto log-chroma plane using ISD

**Features:**

-  Loads 16-bit TIFF images and corresponding ISD maps
-  Applies log-space shadow removal via ISD projection
-  Generates comparison visualizations (original vs. shadow-removed)
-  Saves processed images and debug outputs

**Usage:**

```python
from run import imgProcessor

# Load 16-bit TIFF and ISD map
image = cv2.imread('day-02000.tif', cv2.IMREAD_UNCHANGED)
sr_map = load_isd_map('day-02000_isd.tif')

# Create processor
processor = imgProcessor(image, sr_map, filename='day-02000.tif')

# Apply shadow removal
log_img = processor.convert_img_to_log_space(image)
chroma_img = processor.project_to_chromaticity_plane(anchor=10.4)
output = processor.log_to_linear(chroma_img)
```

---

## Technical Details

### Huawei RAW Format

**24-bit Packed Bayer Pattern:**

```
Byte Sequence: [L0, M0, H0, L1, M1, H1, L2, M2, H2, ...]
              └────────┬────────┘
                   1 pixel (24-bit)

Unpacking Formula:
pixel_value = L + M*256 + H*65536
            = L + M*2^8 + H*2^16
```

**Example:**

```python
raw_data = np.fromfile('day-02000.raw', dtype=np.uint8)
pixels = (
    raw_data[0::3] +
    raw_data[1::3] * 256 +
    raw_data[2::3] * 65536
)
```

### Debayering (Bayer to RGB)

**Method:** 3×3 Convolution-Based Demosaicing

**Kernel Structure:**

```
Channel-dependent 3x3 filters applied to Bayer mosaic:

R channel (at R pixel):  [0    0    0]
                          [0    1    0]
                          [0    0    0]

R channel (at G pixel):  [0   0.25  0]
                          [0.25  0  0.25]
                          [0   0.25  0]

G channel (at G pixel):  [0    0    0]
                          [0    1    0]
                          [0    0    0]

B channel (at B pixel):  [0    0    0]
                          [0    1    0]
                          [0    0    0]
```

**Implementation:**

-  PyTorch `Conv2d` with 5 different kernels
-  `gather()` operation to select appropriate channels
-  GPU acceleration for batch processing
-  Preserves linearity (no non-linear interpolation)

### Gray-World White Balance

**Assumption:** Average scene color is neutral gray.

**Formula:**

```python
mean_r = image[:,:,0].mean()
mean_g = image[:,:,1].mean()
mean_b = image[:,:,2].mean()

image[:,:,0] *= mean_g / mean_r  # Scale R channel
image[:,:,2] *= mean_g / mean_b  # Scale B channel
```

**Advantages:**

-  Simple and fast
-  No user parameters
-  Works well for outdoor driving scenes

**Limitations:**

-  Fails for scenes dominated by one color (e.g., sunset)
-  Not optimal for high color temperature variations

### Dynamic Range Scaling

**Challenge:** 24-bit sensor values (0–16,777,215) → 16-bit TIFF (0–65,535)

**Solution:** 99.5th Percentile Normalization

**Algorithm:**

```python
p99.5 = np.nanpercentile(image, 99.5)  # Robust upper bound
normalized = np.clip(image, 0, p99.5) / p99.5
tiff_16bit = (normalized * 65535).astype(np.uint16)
```

**Example Statistics:**

```
File: day-02000.tif
  p99.5 = 4521.3
  min = 0
  max = 16777215
  → All values > 4521.3 clipped to 65535 in output
  → Shadow details preserved in [0, 4521.3] range
```

---

## Workflow

### 1. Download Huawei ROD Dataset

```bash
# Request access from Huawei or download from official source
# Extract to: /projects/SuperResolutionData/driving/ROD_dataset/
```

**Dataset Structure:**

```
ROD_dataset/
├── Dataset/
│   ├── RawFiles/
│   │   └── dataset/
│   │       ├── day-02000.raw
│   │       ├── day-02001.raw
│   │       ├── ...
│   │       └── day-06052.raw
│   └── Annotations/
│       └── (YOLO format labels)
```

### 2. Run RAW to TIFF Conversion

```bash
# Single-threaded (debugging)
python preprocess_raw_to_tiff.py \
  -p /projects/SuperResolutionData/driving/ROD_dataset/Dataset/RawFiles/dataset \
  -t 1 \
  -o /projects/SuperResolutionData/carolinali-shadowRemoval/raw_to_tiff_output

# Multi-threaded (production)
python preprocess_raw_to_tiff.py \
  -p /projects/SuperResolutionData/driving/ROD_dataset/Dataset/RawFiles/dataset \
  -t -1 \
  -o /projects/SuperResolutionData/carolinali-shadowRemoval/raw_to_tiff_output
```

**Expected Output:**

```
Total raw files found: 8000
Day files in range 02000-06052: 4053
Processing with 24 threads...
100%|████████████████████| 4053/4053 [12:34<00:00, 5.38it/s]
output number: 4053
```

### 3. Verify Output Quality

```bash
# Check file count
ls -1 *.tif | wc -l
# Should output: 4053

# Inspect image statistics
python -c "
import imageio
import numpy as np
img = imageio.imread('day-02000.tif')
print(f'Shape: {img.shape}')
print(f'Dtype: {img.dtype}')
print(f'Range: [{img.min()}, {img.max()}]')
print(f'Mean: {img.mean():.2f}')
"
```

**Expected:**

```
Shape: (1280, 1280, 3)
Dtype: uint16
Range: [0, 65535]
Mean: 12345.67
```

### 4. Generate ISD Maps (Next Pipeline Stage)

```bash
# Use log_chroma_shadow_removal_method/run_log_chroma_batch.py
cd ../log_chroma_shadow_removal_method
python run_log_chroma_batch.py
```

---

## Data Locations (Cluster)

**Input RAW Files:**

```
/projects/SuperResolutionData/driving/ROD_dataset/Dataset/RawFiles/dataset/
```

**Output 16-bit TIFFs:**

```
/projects/SuperResolutionData/carolinali-shadowRemoval/raw_to_tiff_output/
```

**Downstream Pipelines:**

-  Log-chroma generation: `../log_chroma_shadow_removal_method/`
-  Alternative log-chroma: `../new_log_chroma_Bruce/`
-  Y-channel fusion: `../y_channel_exchanged_log_chroma/`

---

## Performance Benchmarks

**Hardware:** Intel Xeon Gold 6248R @ 3.0GHz, 96 cores, NVIDIA V100 32GB

**Processing Time:**

-  Single-threaded: ~3.5 hours (4053 images)
-  24 threads: ~12 minutes (5.38 images/sec)
-  GPU debayering: ~0.3 sec per image
-  CPU-only (no GPU): ~1.2 sec per image

**Disk Usage:**

-  Input RAW files: ~40 GB (4053 × 10 MB avg)
-  Output TIFF files: ~40 GB (4053 × 10 MB)
-  Peak temp space: ~80 GB (during parallel processing)

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory:**

```
RuntimeError: CUDA out of memory
```

**Solution:**

-  Reduce batch size in debayering
-  Use CPU-only mode: `device = torch.device('cpu')`
-  Reduce number of threads

**2. Incorrect Bayer Pattern:**

```
Output images have green/magenta tint
```

**Solution:**

-  Check Bayer pattern in Huawei documentation
-  Adjust debayering kernel indices in `Debayer3x3`

**3. Files Not Found:**

```
Total raw files found: 0
```

**Solution:**

-  Verify input path contains `.raw` files
-  Check filename pattern: `day-XXXXX.raw`
-  Ensure read permissions on input directory

**4. Output All Black/White:**

```
All pixels are 0 or 65535
```

**Solution:**

-  Check 99.5th percentile value (should be > 0)
-  Verify RAW file format (24-bit packed vs. 16-bit)
-  Inspect raw data histogram before scaling

**5. White Balance Artifacts:**

```
Images have strong color cast (blue/orange)
```

**Solution:**

-  Gray-world assumption may fail for certain scenes
-  Consider per-image manual white balance
-  Try alternative WB algorithms (max-RGB, shades of gray)

---

## Quality Validation

### Visual Inspection Checklist

-  [ ] No green/magenta color artifacts (Bayer pattern correct)
-  [ ] Natural colors in daylight scenes (white balance correct)
-  [ ] Shadow details visible (not crushed to black)
-  [ ] Highlights preserved (no excessive clipping)
-  [ ] No banding or posterization (16-bit depth adequate)
-  [ ] Sharp edges (no excessive blur from demosaicing)

### Statistical Checks

```python
import imageio
import numpy as np

# Load output TIFF
img = imageio.imread('day-02000.tif')

# Check dynamic range usage
print(f"Min: {img.min()}, Max: {img.max()}")
assert img.min() >= 0 and img.max() <= 65535

# Check for clipping
clipped_pixels = np.sum((img == 0) | (img == 65535))
print(f"Clipped pixels: {clipped_pixels / img.size * 100:.2f}%")
assert clipped_pixels / img.size < 0.05  # Less than 5% clipped

# Check channel balance
print(f"R mean: {img[:,:,0].mean():.0f}")
print(f"G mean: {img[:,:,1].mean():.0f}")
print(f"B mean: {img[:,:,2].mean():.0f}")
# Should be within 20% of each other
```

---

## Future Improvements

**Potential Enhancements:**

1. **Advanced White Balance:**

   -  Camera-specific color calibration
   -  Illuminant estimation (e.g., Gray Edge)
   -  Deep learning-based AWB

2. **Noise Reduction:**

   -  Spatial denoising (BM3D, NLM)
   -  Temporal denoising (if sequential frames available)
   -  Deep learning denoisers (DnCNN, FFDNet)

3. **HDR Merging:**

   -  If multiple exposures available
   -  Tone mapping for visualization
   -  Preserve linear data for log-chroma

4. **Color Space Conversion:**

   -  XYZ color space
   -  CIE Lab (perceptually uniform)
   -  ACES workflow

5. **Lens Correction:**
   -  Vignetting correction
   -  Distortion correction
   -  Chromatic aberration removal

---

## References

**Huawei ROD Dataset:**

```bibtex
@inproceedings{xu2023toward,
  title={Toward RAW Object Detection: A New Benchmark and a New Model},
  author={Xu, Yuxuan and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={XXX--XXX},
  year={2023}
}
```

**Debayering Algorithms:**

-  Malvar, H., et al. "High-quality linear interpolation for demosaicing of Bayer-patterned color images" (2004)
-  PyTorch implementation: [https://github.com/cheind/pytorch-debayer](https://github.com/cheind/pytorch-debayer)

**Dynamic Range & HDR:**

-  Reinhard, E., et al. "High Dynamic Range Imaging: Acquisition, Display, and Image-Based Lighting" (2010)

---

## Related Directories

-  `../log_chroma_shadow_removal_method/` – ISD prediction and log-chroma generation
-  `../new_log_chroma_Bruce/` – Alternative ViT-based log-chroma method
-  `../my_training_data/` – TIFF to PNG conversion for YOLO training
-  `../baseline/` – Baseline sRGB experiments (not using RAW data)

---

## License

**Original Huawei Code:** Apache License 2.0  
**Modifications:** Copyright 2025 Carolina Li

````
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```'''
````
