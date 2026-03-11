---
title: SVAMITVA MAP FEATURE EXTRACTER
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# 🛰️ SVAMITVA Feature Extraction: Unified AI Pipeline
### *Developed by Students of Digital University Kerala (DUK)*

This repository contains a state-of-the-art AI solution developed for **Problem Statement 1: Feature Extraction from Drone Images**. Our pipeline is specifically designed to automate the extraction of high-precision geospatial features from SVAMITVA drone orthophotos, achieving a target accuracy of **95%**.

---

## 🏆 Hackathon Solution Overview
**Problem Statement:** Feature Extraction from Drone Images (SVAMITVA Scheme)
**Objective:** Develop an AI model capable of identifying key features in high-resolution orthophotos with high precision, optimized for efficient processing and deployment.

### 🎯 Key Features Extracted
- **Building footprint extraction**: High-precision polygonal footprints.
- **Roof-top Classification**: Automated classification into **RCC, Tiled, Tin, and Others**.
- **Road features**: Continuous road network extraction (Polygons & Centerlines).
- **Waterbodies**: Accurate delineation of ponds, rivers, and tanks.
- **Point Feature Identification**: Automated localization of **Distribution Transformers, Over-head Tanks, and Wells**.

---

## 🧠 Technical Architecture

### 1. Foundation Model Backbone
We utilize **SAM-2 (Segment Anything Model 2)** with the `hiera_tiny` configuration as our core feature extractor. This provides a robust, zero-shot capable understanding of object boundaries in complex aerial imagery.

For a detailed breakdown of the model design, see [architecture.md](architecture.md).

### 2. Multi-Head FPN Decoder
Our custom **Feature Pyramid Network (FPN)** fuses multi-scale features from the encoder to handle objects of varying sizes (from huge buildings to tiny wells).
- **Segmentation Heads**: Binary masks for Buildings, Roads, and Waterbodies.
- **Classification Head**: Multi-class branch for Roof Categorization.
- **Connectivity Heads**: Dilated convolution branches for linear features (roads/pipelines).
- **Point Detection**: Specialized heads for localized utility features.

### 4. Advanced Post-Processing Pipeline
Our export module applies research-backed geometric refinement to every extracted feature layer:
- **Buildings & Bridges**: Dominant-angle orthogonalization (Schuegraf et al. ISPRS 2024) for sharp 90° corners; `minAreaRect` fallback for small structures.
- **Roads**: Morphological closing (7px kernel) to bridge tree-canopy gaps; hole filling for continuous surfaces.
- **All Lines (Centerlines, Utilities, Railways)**: `skan`-based skeleton pruning → Chaikin corner-cutting smoothing → dead-end snapping for connected networks.
- **Waterbodies**: Large morphological closing (9px) for smooth natural shorelines; convex hull for tiny ponds.
- **Point Features**: YOLOv8 centroid extraction (no geometric post-processing needed).

### 5. Security & Production Hardening
- Per-class adaptive confidence thresholds (instead of global 0.5)
- Input file size validation (500 MB limit)
- File extension whitelist validation
- Output filename sanitization (path traversal prevention)
- Temp file cleanup on process exit
- Lazy `%`-style logging (prevents format string injection)
- Pinned dependency versions for reproducible builds

---

## 🚀 Efficient Processing & Deployment

### ⚡ Optimization for Large-Scale Data
- **Feature Caching**: Pre-calculates SAM2 embeddings to speed up training by 2.5x.
- **Intelligent Tiling**: 512x512 tiling with 192px overlap to ensure NoData handling and seamless edge reconstruction.
- **Negative Sampling Quota**: Automatically skips 99% empty tiles (farmland/forest) to focus GPU cycles on feature-rich areas.

### 💻 Deployment
A built-in **Streamlit Dashboard** (`app.py`) provides:
- One-click processing for GeoTIFF orthophotos.
- Real-time visualization of all 11 feature layers.
- **GIS Export**: Direct download of vectorized `.gpkg` (GeoPackage) datasets ready for QGIS/ArcGIS.

---

## 🛠️ Usage Guidelines

### Installation
```bash
git clone https://github.com/aaron43210/FEATURE.git
cd FEATURE
pip install -r requirements.txt
```

### Training (Unified Pipeline)
The `train.py` script unifies the three-stage training process into a single command, automating dataset preparation and multi-model training.

```bash
# Train everything (Segmentation + YOLO)
# For 10 villages on DGX (8 GPUs), this takes ~8-12 hours
python train.py --train_dirs ./data/villages/ --epochs 150 --batch_size 16 --cache_features
```

**Workflow:**
1.  **YOLO Prep**: Converts shapefile points to YOLO bounding boxes and image tiles.
2.  **Segmentation**: Trains the SAM2+FPN ensemble for buildings, roads, water, and roof types.
3.  **YOLO Train**: Trains the YOLOv8 point detector for high-precision utility localization.

For DGX environments, the script automatically leverages all available GPUs via DataParallel/DDP.

### Inference
```bash
streamlit run app.py
```

---

**Developed with ❤️ by Digital University Kerala Students**
*Committed to advancing the SVAMITVA scheme through innovative AI/ML techniques.*
