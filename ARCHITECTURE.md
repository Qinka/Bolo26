# YOLO v26 Architecture

## Overview

YOLO v26 is an object detection model that predicts bounding boxes and class probabilities directly from full images in a single evaluation. The model is designed for real-time object detection with high accuracy.

## Model Components

### 1. Backbone: CSPDarknet

The backbone network extracts features from the input image using Cross Stage Partial (CSP) connections.

**Architecture:**
- **Stem**: Initial convolution (3 channels → 32 channels)
- **Stage 1**: CSP block (32 → 64 channels)
- **Stage 2**: CSP block (64 → 128 channels)
- **Stage 3**: CSP block (128 → 256 channels) - P3 output
- **Stage 4**: CSP block (256 → 512 channels) - P4 output
- **Stage 5**: CSP block (512 → 1024 channels) - P5 output

**CSP Block:**
```
Input → Split
    ├─> Conv1 (1x1) → Conv3 (3x3) → Part 1
    └─> Conv2 (1x1) ──────────────> Part 2
         ↓
    Concatenate [Part 1, Part 2]
         ↓
    Conv4 (1x1) → Output
```

### 2. Neck: PAFPN (Path Aggregation Feature Pyramid Network)

The neck combines multi-scale features using both top-down and bottom-up pathways.

**Top-Down Pathway:**
- P5 → lateral conv → upsample → fuse with P4
- P4 → lateral conv → upsample → fuse with P3

**Bottom-Up Pathway:**
- P3 → downsample → fuse with N4
- N4 → downsample → fuse with N5

**Output:**
- N3: Small object detection (256 channels)
- N4: Medium object detection (512 channels)
- N5: Large object detection (1024 channels)

### 3. Detection Head

Multi-scale detection heads predict bounding boxes, objectness scores, and class probabilities.

**Each head predicts:**
- 4 values: bounding box coordinates (x, y, w, h)
- 1 value: objectness score
- C values: class probabilities (C = number of classes)

**Output per anchor:**
- Total: (4 + 1 + C) × 3 anchors

**Output shapes:**
- Small scale (P3): [B, (5+C)×3, H/8, W/8]
- Medium scale (P4): [B, (5+C)×3, H/16, W/16]
- Large scale (P5): [B, (5+C)×3, H/32, W/32]

## Data Flow

```
Input Image [B, 3, 640, 640]
    ↓
Backbone (CSPDarknet)
    ├─> P3 [B, 256, 80, 80]
    ├─> P4 [B, 512, 40, 40]
    └─> P5 [B, 1024, 20, 20]
    ↓
Neck (PAFPN)
    ├─> N3 [B, 256, 80, 80]
    ├─> N4 [B, 512, 40, 40]
    └─> N5 [B, 1024, 20, 20]
    ↓
Detection Heads
    ├─> Small [B, (5+80)×3, 80, 80]
    ├─> Medium [B, (5+80)×3, 40, 40]
    └─> Large [B, (5+80)×3, 20, 20]
```

## Loss Function

The YOLO loss consists of three components:

### 1. Box Regression Loss
- Measures how well predicted boxes match ground truth
- Uses IoU-based loss (GIoU or CIoU)
- Weight: 0.05

### 2. Objectness Loss
- Binary cross-entropy for object presence
- Positive samples: boxes with IoU > threshold
- Negative samples: boxes with IoU < threshold
- Weight: 1.0

### 3. Classification Loss
- Cross-entropy for class predictions
- Only computed for positive samples
- Weight: 0.5

**Total Loss:**
```
L_total = λ_box × L_box + λ_obj × L_obj + λ_cls × L_cls
```

## Inference Pipeline

1. **Forward Pass**: Run image through model
2. **Decode Predictions**: Convert raw outputs to boxes
3. **Filter by Confidence**: Remove low-confidence predictions
4. **Non-Maximum Suppression**: Remove duplicate detections
5. **Output**: Final bounding boxes with class labels

## Post-Processing

### Non-Maximum Suppression (NMS)

```
for each class:
    1. Sort boxes by confidence score
    2. Take box with highest score
    3. Remove all overlapping boxes (IoU > threshold)
    4. Repeat for remaining boxes
```

**Parameters:**
- Confidence threshold: 0.25
- IoU threshold: 0.45

## Key Features

- **Multi-scale Detection**: Detects objects at different sizes
- **Anchor-free Design**: Modern anchor-free detection approach
- **Efficient Architecture**: Optimized for speed and accuracy
- **CSP Connections**: Better gradient flow during training

## Performance Characteristics

- **Input Size**: 640×640 (configurable)
- **Parameters**: ~25M (approximate)
- **FLOPs**: ~52 GFLOPs
- **Inference Speed**: Real-time on modern GPUs
- **Accuracy**: Competitive with state-of-the-art detectors
