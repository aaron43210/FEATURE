# SVAMITVA Feature Extraction: System Architecture

## Overview
The SVAMITVA pipeline is a high-resolution drone imagery processing system designed to extract complex geospatial features (buildings, roads, utilities) from orthophotos. It uses an ensemble approach combining **SAM2-based foundational vision** with **task-specific heads** and **YOLOv8 precision object detection**.

---

## 🏗️ Model Architecture

The core model follows a **foundational-encoder / multi-task-decoder** design:

### 1. Backbone: SAM2 (Segment Anything Model 2)
- **Engine**: Hiera-Tiny Image Encoder.
- **Output**: Multi-scale feature maps at 4 levels (Stride 4, 8, 16, 32).
- **Optimization**: Frozen during Stage 1; fine-tuned in Stage 2 with a 0.1x learning rate multiplier for stability.

### 2. Decoder: FPN + Attention
- **Feature Pyramid Network (FPN)**: Fuses multi-scale features to capture both fine details (Utility points) and global context (Road networks).
- **CBAM Attention**: Spatial and Channel attention modules refine the feature fusion process before passing to task heads.

### 3. Task-Specific Heads (11 Tasks)
The model predicts 11 distinct outputs simultaneously:
- **Buildings**: Binary Mask + Roof-Type (5-class categorization).
- **Roads**: Surface Polygon + Centerline String.
- **Waterbodies**: Polygon + Edge Line + Point (Well).
- **Utilities**: Line (Pipeline/Wire) + Point (Transformer/Tank).
- **Infrastructure**: Bridges + Railway lines.

---

## ⚙️ Training Pipeline Workflow

The system is optimized for **DGX-grade performance** using a 3-stage process:

### Stage 1: YOLO Precision Dataset Preparation
- Extracts tiles centered on point-annotated features (Wells, Transformers).
- Auto-generates a YOLOv8-ready dataset to handle detections that classic segmentation might miss.

### Stage 2: Foundation Training (SAM2 + FPN)
- **Loss Strategy**: Hybrid `BCE` + `Dice` + `Lovasz-Softmax` + `Boundary Loss`.
- **Optimization**: Mixed-Precision (AMP) + Feature Caching.
- **Hardware Agnostic**: Automatic fallback from 8-GPU DDP → Single CUDA → Apple Silicon (MPS) → CPU.

### Stage 3: YOLO High-Precision Tuning
- Independent tuning of YOLOv8 for small/sparse objects.
- Resulting weights are stored separately to maintain modularity.

---

## 🛰️ Inference & GIS Integration

The **Tiled Predictor Engine** ensures seamless results across multi-gigabyte GeoTIFFs:

1.  **Gaussian Blending**: Overlapping tiles are merged using a 2D Gaussian kernel to eliminate "tiling artifacts" and sharp edges.
2.  **Dual-Model Fusion**: Results from the SAM2 segmentation and YOLO detection are probabilistic fused (Max-pooling) to produce the final point layers.
3.  **Raster-to-Vector**: Centroids, Skeletonization, and Polygonization algorithms convert raw masks into GIS-standard `.gpkg` (GeoPackage) datasets.

---

## 📈 Hardware Hierarchy (Self-Healing)
The system intelligently scales based on available environment:
1.  **DGX DDP + Cache**: Full speed extraction (~20 mins/epoch for 60GB).
2.  **DGX Single GPU**: Robust fallback if DDP is busy.
3.  **Local Mac/MPS**: High-efficiency development environment.
4.  **CPU**: Basic validation mode.
