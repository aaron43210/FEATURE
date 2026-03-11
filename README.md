# SVAMITVA Feature Extraction V3 🛰️

A state-of-the-art production ensemble pipeline for multi-class semantic segmentation and remote sensing feature extraction from high-resolution drone orthophotos. Built on top of Meta's SAM-2 (Segment Anything) and YOLOv8, combined with a custom Feature Pyramid Network (FPN) and advanced GIS post-processing processing.

## 🚀 Key Features

- **Multi-Task Extraction:** Simultaneously extracts 11 different geospatial features (Buildings, Roof Types, Roads, Road Centerlines, Waterbodies, Utility Lines, etc.).
- **Foundation Model Backbone:** Leverages `sam2.1_hiera_tiny` as a powerful frozen backbone for high-resolution semantic understanding.
- **Robust Loss Function:** Employs a 5-part composite loss (BCE + OHEM + Dice + Focal + Lovász + Boundary) specifically tuned for crisp polygon boundaries.
- **Production Web UI:** Includes a Streamlit application (`app.py`) for drag-and-drop inference and visualization.
- **GIS Export Pipeline:** Automatically vectors predictions into ready-to-use QGIS/ArcGIS GeoPackage (`.gpkg`) files with associated attributes (e.g., Roof Types).

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/aaron43210/FEATURE.git
cd FEATURE

# Install dependencies
pip install -r requirements.txt
```

## 🧠 Training

The trainer supports both single-GPU and Multi-GPU (DDP & DataParallel) environments automatically. 

```bash
# Train on a single map or directory of maps
python train.py --epochs 100 --batch_size 16 --train_dirs /path/to/data/MAP1 /path/to/data/MAP2

# Enable feature caching for 2x faster epochs
python train.py --epochs 100 --batch_size 16 --cache_features --train_dirs /path/to/data/MAP1
```

## 🌍 Inference & Deployment

Launch the Streamlit web application to run the unified ensemble pipeline:

```bash
streamlit run app.py
```

From the UI, you can:
1. Upload an orthophoto (`.tif`)
2. Adjust confidence thresholds and TTA (Test-Time Augmentation)
3. Visualize outputs natively overlaid on the map
4. Click **"Generate GIS Layers"** to download vectorized `.gpkg` outputs.

## 📁 Repository Structure

- `models/`: Contains the SAM2 Encoder, FPN Decoder, and Specialized Heads.
- `train_engine/`: Production training loop, caching, and config files.
- `inference/`: Predictor classes, YOLO ensembles, and vectorization logic (`export.py`).
- `data/`: Custom PyTorch Dataset loaders handling large-scale TIF tiling and raw shapefile rasterization.
- `app.py`: The main Streamlit web interface.
