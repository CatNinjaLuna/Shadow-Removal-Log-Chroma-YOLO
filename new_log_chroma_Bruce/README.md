# Alternative Log-Chromaticity Implementation (Bruce's Method)

This directory contains an alternative implementation of log-chromaticity generation using advanced deep learning architectures (U-Net with ViT hybrid and pure Vision Transformer models) for ISD prediction.

---

## Overview

This implementation explores more sophisticated model architectures for predicting Illuminant Spectral Direction (ISD) maps compared to the standard ResNet50-UNet approach. The key innovations include:

-  **Hybrid U-Net + Vision Transformer (ViT)** models
-  **Pure patch-to-patch Vision Transformer** architectures
-  **Flexible input representations** (linear or log-space)
-  **Advanced attention mechanisms** (SE blocks, Transformer self-attention)

**Research Question:** Can Transformer-based architectures improve ISD prediction quality and subsequent log-chromaticity generation?

---

## Architecture Variants

### 1. ResNet50-UNet (Baseline)

Standard encoder-decoder with ResNet50 backbone and U-Net decoder with skip connections.

**Features:**

-  Pretrained ResNet50 encoder (ImageNet)
-  5-stage U-Net decoder with upsampling blocks
-  Squeeze-and-Excitation (SE) attention blocks
-  Dropout regularization
-  Supports arbitrary input channels

### 2. ResNet50-UNet + ViT Hybrid

Enhanced U-Net with Transformer bottleneck between encoder and decoder.

**Architecture:**

```
ResNet50 Encoder (2048 channels)
    ↓
1x1 Conv Projection → ViT Embedding
    ↓
Transformer Encoder (multi-head self-attention)
    ↓
1x1 Conv Unprojection → 2048 channels
    ↓
U-Net Decoder with Skip Connections
```

**Key Parameters:**

-  `vit_embed_dim`: Transformer embedding dimension (default: 512)
-  `vit_depth`: Number of Transformer layers (default: 4)
-  `vit_heads`: Number of attention heads (default: 8)
-  `vit_ff_dim`: Feedforward network dimension (default: 2048)

**Freezing Options:**

-  Freeze encoder: `freeze_encoder=True`
-  Freeze decoder: `freeze_decoder=True`
-  Fine-tune only Transformer bottleneck

### 3. Pure Vision Transformer (ViT) - Patch-to-Patch

Two variants of patch-based Transformer models:

#### Version 1: Sinusoidal Positional Encoding

```
Patchify (non-overlapping patches)
    ↓
Linear Projection → Embedding
    ↓
+ Sinusoidal Positional Encoding
    ↓
Transformer Encoder
    ↓
Linear Decoder → Patch Space
    ↓
Unpatchify → Image
```

#### Version 2: CNN Decoder with Learned Positional Embeddings

```
Conv2d Patch Embedding (stride = patch_size)
    ↓
+ Learned Positional Embeddings + Dropout
    ↓
Transformer Encoder
    ↓
CNN Decoder with PixelShuffle Upsampling
    ↓
Output Image
```

**Advantages:**

-  Global receptive field (Transformer attention)
-  Better for long-range dependencies
-  Potentially more robust to large shadow regions

**Disadvantages:**

-  Requires more data to train effectively
-  Computationally expensive
-  Sensitive to patch size selection

---

## Scripts

### 1. `models_unet.py`

Defines ResNet-based U-Net architectures with optional ViT bottleneck.

**Key Classes:**

#### `SEBlock(channels, reduction=16)`

Squeeze-and-Excitation attention module for channel recalibration.

#### `UpBlock(in_c, skip_c, out_c, se_block=False, dropout=0.0)`

Decoder block with:

-  Bilinear upsampling (2x)
-  Skip connection concatenation
-  Conv-BN-ReLU stacks
-  Optional SE attention
-  Optional dropout

#### `ResNet50UNet(in_channels=3, out_channels=3, pretrained=True, checkpoint=None, se_block=True, dropout=0.0)`

Standard ResNet50-based U-Net.

**Parameters:**

-  `in_channels`: Input channels (3 for RGB)
-  `out_channels`: Output channels (3 for ISD map)
-  `pretrained`: Use ImageNet pretrained weights
-  `checkpoint`: Load custom checkpoint
-  `se_block`: Enable SE attention in decoder
-  `dropout`: Dropout rate (0.0 = no dropout)

#### `ResNet18UNet(...)`

Lighter variant with ResNet18 encoder.

#### `ResNet50UNet_ViT(...)`

Hybrid model with Transformer bottleneck.

**Additional Parameters:**

-  `vit_embed_dim`: Transformer embedding dimension
-  `vit_depth`: Number of Transformer layers
-  `vit_heads`: Multi-head attention heads
-  `vit_ff_dim`: Feedforward dimension
-  `freeze_encoder`: Freeze ResNet encoder
-  `freeze_decoder`: Freeze U-Net decoder

**Architecture Stages:**

```
Input: 3 x H x W

Encoder:
  conv1:   64 x H/2 x W/2
  layer1: 256 x H/4 x W/4
  layer2: 512 x H/8 x W/8
  layer3: 1024 x H/16 x W/16
  layer4: 2048 x H/32 x W/32 (bottleneck)

[Optional ViT Bottleneck]

Decoder:
  up1: 2048+1024 → 1024 x H/16 x W/16
  up2: 1024+512 → 512 x H/8 x W/8
  up3: 512+256 → 256 x H/4 x W/4
  up4: 256+64 → 128 x H/2 x W/2
  final: 128 → 3 x H x W (ISD map)
```

---

### 2. `models_vit.py`

Pure Vision Transformer architectures for patch-to-patch learning.

**Key Classes:**

#### `PositionalEncoding(emb_size, max_len=1000)`

Sinusoidal positional encoding for sequence embeddings.

#### `ViT_Patch2Patch(img_size, patch_size, in_ch, out_ch, emb_dim, depth, heads, ff_dim)`

Vision Transformer with sinusoidal positional encoding.

**Parameters:**

-  `img_size`: Input image size (assumes square)
-  `patch_size`: Patch size (e.g., 8, 16, 32)
-  `in_ch`: Input channels (3 for RGB)
-  `out_ch`: Output channels (3 for ISD)
-  `emb_dim`: Embedding dimension
-  `depth`: Number of Transformer layers
-  `heads`: Number of attention heads
-  `ff_dim`: Feedforward dimension

**Example:**

```python
model = ViT_Patch2Patch(
    img_size=512,
    patch_size=16,
    in_ch=3,
    out_ch=3,
    emb_dim=768,
    depth=12,
    heads=12,
    ff_dim=3072
)
```

#### `ViT_Patch2Patch_ver2(...)`

Enhanced ViT with CNN-style patch embedding and decoder.

**Improvements:**

-  Conv2d-based patch embedding (stride = patch_size)
-  Learned positional embeddings with dropout
-  CNN decoder with PixelShuffle upsampling
-  Better detail preservation

**Typical Configuration:**

```python
model = ViT_Patch2Patch_ver2(
    img_size=512,
    patch_size=8,
    in_ch=3,
    out_ch=3,
    emb_dim=512,
    depth=8,
    heads=8,
    ff_dim=2048,
    pos_dropout=0.1
)
```

---

### 3. `srvizlib.py`

Data loading and transformation utilities for ISD prediction.

**Key Classes:**

#### `BasicDataset_16BitTIFF(image_paths, transform=None)`

Custom dataset loader for 16-bit TIFF images.

**Features:**

-  Loads and caches 16-bit TIFFs
-  Normalizes to [0, 1] range
-  Thread-based parallel loading
-  Memory-efficient caching

**Usage:**

```python
dataset = BasicDataset_16BitTIFF(
    image_paths=['img1.tif', 'img2.tif'],
    transform=transforms.Compose([
        ToLogRGB(),
        CenterCropToSize((512, 512)),
        ToTensor()
    ])
)
```

#### `ToTensor()`

Converts NumPy arrays to PyTorch tensors with proper channel ordering.

**Transformation:** `(H, W, C) → (C, H, W)`

#### `ToLogRGB()`

Converts linear 16-bit RGB to log-space representation.

**Formula:**

```python
log_img = log(linear_img * 65535)
log_img_normalized = clamp(log_img / log(65535), 0, 1)
```

**Handles:**

-  Zero values (sets log(0) = 0)
-  Normalization to [0, 1]
-  GPU acceleration

#### `CenterCropToSize(target_size)`

Center crops images to fixed size.

**Parameters:**

-  `target_size`: (height, width) tuple

**Usage:**

```python
crop = CenterCropToSize((512, 512))
sample = crop(sample)
```

#### `ProjectLogChrom(anchor=10.4)`

Projects log-RGB onto log-chromaticity plane using ISD map.

**Pipeline:**

1. Convert to log-space: `L = log(I)`
2. Shift: `L_shifted = L - anchor`
3. Project: `dot = <L_shifted, ISD>`
4. Orthogonal plane: `L_plane = L_shifted - dot * ISD + anchor`
5. Back to linear: `I_chroma = exp(L_plane)`

**Parameters:**

-  `anchor`: Log-space anchor point (default: 10.4)

---

### 4. `srvizrun.py`

Inference script for generating ISD predictions and log-chromaticity images.

**Purpose:**

-  Run trained models on new images
-  Generate ISD maps and log-chroma outputs
-  Support batch processing of directories
-  Cluster-friendly (no GUI)

**Usage:**

```bash
# Single image
python srvizrun.py -m unet -r log image.tif

# Multiple images
python srvizrun.py -m vit -r linear img1.tif img2.tif img3.tif

# Directory batch processing
python srvizrun.py -m unet -r log /path/to/images/
```

**Arguments:**

-  `-m, --model`: Model type (`unet`, `vit`, `mamba`)
-  `-r, --representation`: Input space (`log`, `linear`)
-  `images`: Image paths or directory paths

**Model Loading:**

-  Automatically loads trained weights
-  Supports GPU or CPU inference
-  Handles variable image sizes

**Outputs:**

-  ISD maps (3-channel unit vectors)
-  Log-chromaticity images (16-bit TIFF)
-  Optional: 8-bit PNG previews

---

### 5. `srvizrun_continue.py`

Resumable version of `srvizrun.py` for handling interrupted batch jobs.

**Features:**

-  Resume from specific filename
-  Skip already-processed files
-  Progress tracking
-  Error recovery

**Usage:**

```bash
python srvizrun_continue.py \
    -m unet \
    -r log \
    --start_from "image_100.tif" \
    /path/to/images/
```

---

### 6. `split_dataset.py`

Dataset splitting utility for train/validation sets.

**Purpose:**

-  Split image-label pairs for YOLO training
-  Reproducible random splits
-  Preserves pairing

**Configuration:**

```python
--images: "/projects/.../new_log_chroma_Bruce/log_chroma_outputs/png"
--labels: "/projects/.../new_log_chroma_Bruce/log_chroma_txt"
--out:    "/projects/.../new_log_chroma_Bruce/log_chroma_splitted_data"
--ratio:  0.8  # 80% train, 20% val
--seed:   42   # Random seed
```

**Usage:**

```bash
python split_dataset.py \
    --images /path/to/images \
    --labels /path/to/labels \
    --out /path/to/output \
    --ratio 0.8 \
    --seed 42
```

**Output Structure:**

```
output/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

---

## Training Results & Analysis

The `exp_log_chroma_Bruce_result/` directory contains validation metrics and visualizations from training the alternative log-chromaticity model.

### Performance Metrics

#### Precision-Recall Analysis

**PR Curve (`PR_curve.png`):**

-  Shows model performance across confidence thresholds
-  Higher area under PR curve indicates better overall performance
-  Comparison with baseline can reveal improvements from advanced architectures

**Precision Curve (`P_curve.png`):**

-  Precision vs. confidence threshold
-  Higher precision at lower thresholds suggests fewer false positives
-  Important for evaluating Transformer-based attention mechanisms

**Recall Curve (`R_curve.png`):**

-  Recall vs. confidence threshold
-  High recall indicates good detection of true positives
-  Critical for shadow-invariant detection performance

#### F1 Score Analysis (`F1_curve.png`)

The F1 curve identifies optimal operating points for each object class:

-  **Peak F1 scores:** Best balance between precision and recall
-  **Comparison points:**
   -  Baseline sRGB model
   -  Standard log-chroma (ResNet50-UNet)
   -  Alternative log-chroma (ViT-based)

**Expected Observations:**

-  ViT models may show improved performance on small objects (Pedestrians, Cyclists)
-  Transformer attention may better capture long-range dependencies across shadow boundaries
-  Trade-offs between computational cost and performance gains

### Confusion Matrix Analysis

**Raw Confusion Matrix (`confusion_matrix.png`):**

-  Absolute counts of predictions vs. ground truth
-  Diagonal elements: correct predictions
-  Off-diagonal: confusion patterns

**Normalized Confusion Matrix (`confusion_matrix_normalized.png`):**

-  Percentage-based view for class balance
-  High diagonal values (>0.8) indicate strong class-specific performance

**Key Comparison Points:**

1. **Shadow Robustness:** Compare confusion in shadowed vs. non-shadowed regions
2. **Small Object Detection:** Evaluate Pedestrian/Cyclist performance
3. **Class Discrimination:** Assess Truck vs. Car, Cyclist vs. Pedestrian confusion

### Validation Sample Analysis

**Visual Inspection (`val_batch*.jpg` files):**

**Labels vs. Predictions:**

-  **Green boxes:** Ground truth annotations
-  **Colored boxes:** Model predictions with class-specific colors

**Comparative Analysis:**

1. **vs. Baseline sRGB:**

   -  Improved detection in shadowed regions?
   -  More consistent confidence scores?
   -  Better small object localization?

2. **vs. Standard Log-Chroma:**
   -  ViT attention benefits visible?
   -  Improved boundary precision?
   -  Trade-offs in computational cost?

**Expected Advantages:**

-  **Transformer Attention:** Better global context understanding
-  **Hybrid Architecture:** Combines local (CNN) and global (ViT) features
-  **Shadow Invariance:** More robust to illumination variations

**Potential Challenges:**

-  **Training Data Requirements:** ViT may need more data
-  **Computational Cost:** Slower inference than pure CNN
-  **Overfitting Risk:** More parameters to tune

---

## Data Locations (Cluster)

**Model Weights:**

-  U-Net weights: `/projects/.../new_log_chroma_Bruce/checkpoints/unet_best.pth`
-  ViT weights: `/projects/.../new_log_chroma_Bruce/checkpoints/vit_best.pth`

**Input Data:**

-  Linear 16-bit TIFFs: `/projects/.../raw_to_tiff_output`

**Output Data:**

-  Log-chroma outputs: `/projects/.../new_log_chroma_Bruce/log_chroma_outputs/`
-  8-bit PNG previews: `/projects/.../new_log_chroma_Bruce/log_chroma_outputs/png/`
-  YOLO labels: `/projects/.../new_log_chroma_Bruce/log_chroma_txt/`

**Training/Validation Split:**

-  Split dataset: `/projects/.../new_log_chroma_Bruce/log_chroma_splitted_data/`

**Results:**

-  Training metrics: `exp_log_chroma_Bruce_result/`

---

## Typical Workflow

### 1. ISD Prediction & Log-Chroma Generation

```bash
# Generate log-chroma images using U-Net
python srvizrun.py -m unet -r log /path/to/linear/tiffs/

# Alternative: Use ViT model
python srvizrun.py -m vit -r log /path/to/linear/tiffs/
```

### 2. Dataset Preparation

```bash
# Split into train/val sets
python split_dataset.py \
    --images /path/to/log_chroma_pngs \
    --labels /path/to/yolo_labels \
    --out /path/to/output \
    --ratio 0.8
```

### 3. YOLO Training

```bash
yolo detect train \
    data=data_log_chroma_bruce.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=1280
```

### 4. Evaluation & Comparison

```bash
# Run predictions
yolo detect predict \
    model=/path/to/best.pt \
    source=/path/to/test/images \
    save=True

# Compare with baseline and standard log-chroma
# Analyze precision, recall, F1 scores
# Visual inspection of val_batch predictions
```

---

## Model Comparison Summary

| Aspect                     | ResNet50-UNet | ResNet50-UNet+ViT | Pure ViT       |
| -------------------------- | ------------- | ----------------- | -------------- |
| **Parameters**             | ~30M          | ~35M              | ~25M (patch16) |
| **Inference Speed**        | Fast          | Medium            | Slow           |
| **Training Data Needs**    | Moderate      | Moderate-High     | High           |
| **Local Details**          | Excellent     | Excellent         | Good           |
| **Global Context**         | Good          | Excellent         | Excellent      |
| **Shadow Robustness**      | Good          | Better            | Best (theory)  |
| **Small Object Detection** | Good          | Better            | Better         |
| **Memory Usage**           | Moderate      | High              | Very High      |

---

## Hyperparameter Recommendations

### U-Net Training:

```python
lr = 1e-4
batch_size = 8
epochs = 100
optimizer = Adam
scheduler = ReduceLROnPlateau
```

### ViT Training:

```python
lr = 1e-4 (warmup to 3e-4)
batch_size = 4  # larger patches
epochs = 150  # needs more training
optimizer = AdamW (weight_decay=0.05)
scheduler = CosineAnnealingLR
```

### Hybrid U-Net+ViT:

```python
lr = 5e-5
batch_size = 6
epochs = 120
freeze_encoder = True (first 20 epochs)
```

---

## Performance Benchmarks

**Inference Time (per image, 1920x1280):**

-  ResNet50-UNet: ~0.5s (GPU), ~5s (CPU)
-  ResNet50-UNet+ViT: ~0.8s (GPU), ~10s (CPU)
-  Pure ViT (patch16): ~1.2s (GPU), ~20s (CPU)

**Memory Requirements:**

-  U-Net: ~4GB GPU memory
-  U-Net+ViT: ~6GB GPU memory
-  Pure ViT: ~8GB GPU memory

---

## Troubleshooting

**Common Issues:**

1. **OOM Errors with ViT:**

   -  Reduce batch size
   -  Use larger patch size (16→32)
   -  Reduce image resolution
   -  Enable gradient checkpointing

2. **ViT Training Instability:**

   -  Use learning rate warmup
   -  Apply stronger regularization (dropout, weight decay)
   -  Increase training data
   -  Try mixed precision training

3. **Poor ViT Performance:**

   -  Needs more training data (augmentation helps)
   -  Try pretrained ViT backbones
   -  Adjust patch size to image content
   -  Tune positional encoding

4. **Slow Inference:**
   -  Use smaller models for real-time applications
   -  Batch processing where possible
   -  Consider model quantization
   -  Use ONNX export for optimization

---

## Future Directions

**Potential Improvements:**

1. **Mamba Vision Models:** State-space models for efficient long-range modeling
2. **Diffusion Models:** Probabilistic ISD prediction
3. **Multi-Scale ViT:** Hierarchical Vision Transformers (e.g., Swin Transformer)
4. **Self-Supervised Pretraining:** Leverage unlabeled TIFF data
5. **Ensemble Methods:** Combine U-Net and ViT predictions

---

## References

**Vision Transformers:**

-  Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ICLR 2021)

**Hybrid CNN-Transformer:**

-  Chen, J., et al. "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation" (2021)

**Log-Chromaticity:**

-  Finlayson, G.D., et al. "Illuminant and device invariant colour using histogram equalisation"
-  Maxwell, B.A., et al. "Illumination-Invariant Color Object Recognition"

**U-Net:**

-  Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"

---

## Related Directories

-  `../log_chroma_shadow_removal_method/` - Standard log-chroma implementation
-  `../baseline/` - Baseline sRGB experiments
-  `../ultralytics_src_new_log_chroma/` - Modified Ultralytics for log-chroma training
