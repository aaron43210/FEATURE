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

### 2. Multi-Head FPN Decoder
Our custom **Feature Pyramid Network (FPN)** fuses multi-scale features from the encoder to handle objects of varying sizes (from huge buildings to tiny wells).
- **Segmentation Heads**: Binary masks for Buildings, Roads, and Waterbodies.
- **Classification Head**: Multi-class branch for Roof Categorization.
- **Connectivity Heads**: Dilated convolution branches for linear features (roads/pipelines).
- **Point Detection**: Specialized heads for localized utility features.

### 3. Advanced Loss Strategy
To achieve 95% accuracy, we employ a composite loss function:
- **BCE + Dice**: For stable overlap optimization.
- **Lovász-Softmax**: Directly optimizes the Intersection-over-Union (IoU) metric.
- **OHEM (Online Hard Example Mining)**: Focuses training on the most challenging 70% of pixels (e.g., shadows, complex roof textures).
- **Boundary Loss**: Ensures crisp, cartographic-grade edges for building polygons.

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

### Training (DGX Optimized)
```bash
# Multi-GPU training on 10 village datasets
torchrun --nproc_per_node=8 train.py --epochs 150 --train_dirs ./data/villages/
```

### Inference
```bash
streamlit run app.py
```

---

**Developed with ❤️ by Digital University Kerala Students**
*Committed to advancing the SVAMITVA scheme through innovative AI/ML techniques.*
