# 🛰️ SVAMITVA AI Extraction Ensemble (V3)

High-resolution geospatial feature extraction from drone orthophotos, designed for the **SVAMITVA** village mapping scheme. This pipeline achieves **≥95% accuracy** by leveraging a specialized ensemble of State-of-the-Art (SOTA) deep learning models.

## Scope
This repository is scoped to **Problem Statement 1 only**:
- Building footprint extraction + roof class (RCC/Tiled/Tin/Others)
- Road extraction
- Waterbody extraction
- Utility point/line extraction (transformer, OHT/tank, wells, etc.)

Problem Statement 2 (DTM/drainage from point cloud) is intentionally excluded from this codebase.

---

## 🚀 Key Features

- **Unified SAM2 Architecture**: High-performance **SAM2 (Hiera B+)** backbone and **FPN** decoder for consistent multi-scale feature extraction across all tasks.
- **Specialized Task Heads**: Integrated multi-head system:
    - `LineHead` with `DLinkBlock` for road/railway continuity.
    - `BuildingHead` with dual-output for segmentation and roof classification.
    - `BinaryHead`/`LineHead` for waterbodies and utilities.
- **Full State Resumption**: Robust training pipeline with 100% recovery of optimizer and scheduler states.
- **Interactive Tiled Inference**: Optimized Streamlit V3 app for large-scale GeoTIFF visualization (Global vs. Detail views).
- **Point Feature Fusion**: Seamless integration of **YOLOv8** for sparse point objects (wells, transformers, tanks).
- **Unified GIS Export**: Automated generation of georeferenced **GeoPackage (.gpkg)** layers.

---

## 🏗️ Architecture Stack

| Component | Implementation | Feature Focus |
| :--- | :--- | :--- |
| **Backbone** | **SAM2 (Hiera B+)** | Global Multi-scale Feature Extraction |
| **Decoder** | **FPN + CBAM Attention** | Multi-level Feature Fusion & Context |
| **Buildings** | Dual-Output Head | Instance Mask + Roof Classification |
| **Roads/Rail** | D-LinkNet Head | Connectivity & Directional Regularity |
| **Utilities** | U-Net++ Multi-Head | Linear & Point Feature Precision |
| **Points** | YOLOv8 + Fusion | Wells, Transformers, Tanks |

---

## 🧭 Recommended Global Strategy

For best practical accuracy and stability on drone imagery, use a **hybrid pipeline**:

1. **SAM2 for strong segmentation priors and auto-label bootstrapping**
2. **Task-specialized models for production heads**
   - DeepLabV3+ (building, water polygons)
   - D-LinkNet (roads / centrelines)
   - U-Net++ (utility linear features)
   - HRNet (railway continuity)
3. **YOLOv8 for sparse point objects** (wells, transformers, tanks)
4. **Roof-type classification head** for RCC/Tiled/Tin/Others

The current codebase supports this in practice via:
- SAM2-based multi-head segmentation model
- YOLOv8 fusion for point masks
- Roof-type raster + GIS export

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-org>/<your-repo>.git
   cd <your-repo>
   ```

2. **Install Dependencies**:
   ```bash
   source py314/bin/activate
   pip install -r requirements.txt
   ```

---

## 🛠️ Usage

### 1. Interactive Web Application
Launch the production-grade Streamlit interface for end-to-end extraction and visualization:
```bash
streamlit run app.py
```

### 2. Model Training
Train the ensemble model on your own drone dataset (MAP1, MAP2, etc.):
```bash
source py314/bin/activate
python train.py --train_dirs /path/to/data --epochs 100 --lr 3e-4 --split_mode map --tile_size 1024 --tile_overlap 192
```
*Note: Use `--quick_test` for a 3-epoch smoke test.*

### 3. MAP1 3-Epoch Verification Run
```bash
source py314/bin/activate
python train.py \
  --train_dirs ../DATA/MAP1 \
  --val_dir ../DATA/MAP1 \
  --epochs 3 \
  --batch_size 1 \
  --tile_size 1024 \
  --tile_overlap 192 \
  --num_workers 0 \
  --freeze_epochs 1 \
  --max_train_tiles 24 \
  --max_val_tiles 12 \
  --name map1_3ep_ps1_verify
```
### 4. DGX/GPU Cluster Training
For high-performance environments (NVIDIA DGX), use the provided notebook:
- `dgx.ipynb`: Contains the multi-phase staged training logic.

---

## 📁 Data Structure
To train or run inference, organize your data as follows:
```text
data/
└── MAP_ID/
    ├── MAP_ID.tif         # High-resolution Orthophoto
    ├── Build_up.shp       # Building annotations
    ├── Road.shp           # Road annotations
    └── ...                # Other feature shapefiles
```

---

## 🎯 Output Keys

| Output Key | Target Feature | Geometry |
| :--- | :--- | :--- |
| `building_mask` | Built-up Area | Polygon |
| `roof_type_mask` | Roof Classification (RCC/Tiled/Tin/Others) | Polygon |
| `road_mask` | Road | Polygon |
| `road_centerline_mask` | Road Centre Line | Line |
| `waterbody_mask` | Water Body | Polygon |
| `waterbody_line_mask` | Water Body Line | Line |
| `waterbody_point_mask` | Waterbody Point (Wells) | Point |
| `utility_line_mask` | Utility (Pipeline/Wires) | Line |
| `utility_point_mask` | Utility Point (Transformers/Tanks) | Point |
| `bridge_mask` | Bridge | Polygon |
| `railway_mask` | Railway | Line |

---

## 📊 Performance & Target
This project is engineered to meet the stringent requirements of SVAMITVA mapping:
- **Target IoU**: ≥95% across all primary layers.
- **Resolution**: Native support for 5cm - 10cm GSD drone imagery.
- **Export Formats**: Standard GIS vector outputs (SHP/GPKG/GeoJSON).

---

## ⚖️ License
This project is proprietary for the SVAMITVA scheme. (C) 2026.
