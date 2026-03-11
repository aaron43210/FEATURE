#!/usr/bin/env python3
"""
🚀 SVAMITVA Unified Training Pipeline (Hackathon V3)
Developed by Students of Digital University Kerala (DUK)

This script orchestrates the training of both the SAM2-based segmentation 
model and the YOLOv8 point-detection model simultaneously.

Usage:
    python train.py --train_dirs data/MAP1 data/MAP2 --epochs 100
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-25s │ %(levelname)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("unified_train")

def parse_args():
    p = argparse.ArgumentParser(description="Unified Training for SVAMITVA Hackathon")
    
    # Shared Data Path
    p.add_argument("--train_dirs", nargs="+", default=["data/MAP1"], 
                   help="Directories containing MAP*.tif + shapefiles")
    
    # Training Hyperparameters
    p.add_argument("--epochs", type=int, default=100, help="Epochs for both models")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for segmentation")
    p.add_argument("--yolo_batch", type=int, default=16, help="Batch size for YOLO")
    
    # Optimization
    p.add_argument("--cache_features", action="store_true", help="Enable SAM2 feature caching")
    p.add_argument("--multi_gpu", action="store_true", default=True, help="Use 8x GPUs if available")
    
    return p.parse_args()

def run_step(cmd: List[str], step_name: str):
    logger.info(f"▶️ Starting Step: {step_name}")
    start_time = time.time()
    try:
        # We use check=True to raise an error if the step fails
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = (time.time() - start_time) / 60
        logger.info(f"✅ Step '{step_name}' completed in {elapsed:.2f} mins")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Step '{step_name}' failed with error: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    
    logger.info("=" * 60)
    logger.info("🌟 SVAMITVA UNIFIED TRAINING PIPELINE STARTING 🌟")
    logger.info(f"  Training Dirs: {args.train_dirs}")
    logger.info(f"  Total Epochs:  {args.epochs}")
    logger.info("=" * 60)

    # Step 1: Prepare YOLO Point Dataset
    # This converts shapefile points into YOLO .txt labels + image tiles
    logger.info("\n[STAGE 1/3] Preparing YOLO Dataset...")
    yolo_prep_cmd = [
        sys.executable, str(project_root / "scripts" / "prepare_yolo_dataset.py"),
        "--map_dirs"
    ] + args.train_dirs + [
        "--output", "yolo_dataset",
        "--tile_size", "1024"
    ]
    run_step(yolo_prep_cmd, "YOLO Prep")

    # Step 2: Train Segmentation & Roof Model (Ensemble V3)
    # This handles buildings, roads, water, and roof classifications
    logger.info("\n[STAGE 2/3] Training Segmentation & Roof Model...")
    seg_cmd = [
        sys.executable, str(project_root / "train_engine" / "train_segmentation.py"),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--train_dirs"
    ] + args.train_dirs + [
        "--checkpoint_dir", "checkpoints",
        "--name", "ensemble_v3_final"
    ]
    if args.cache_features:
        seg_cmd.append("--cache_features")
    
    run_step(seg_cmd, "Segmentation Training")

    # Step 3: Train YOLO Point Detection Model
    # This handles Transformers, Wells, and Overhead Tanks
    logger.info("\n[STAGE 3/3] Training YOLO Point Detector...")
    yolo_train_cmd = [
        sys.executable, str(project_root / "scripts" / "train_yolo.py"),
        "--data", "yolo_dataset/svamitva_points.yaml",
        "--epochs", str(args.epochs),
        "--batch", str(args.yolo_batch),
        "--project", "checkpoints/yolo_runs"
    ]
    run_step(yolo_train_cmd, "YOLO Training")

    logger.info("\n" + "=" * 60)
    logger.info("🏆 ALL MODELS TRAINED SUCCESSFULLY! 🏆")
    logger.info("  1. Segmentation: checkpoints/ensemble_v3_final/best.pt")
    logger.info("  2. Point Detection: check/yolo_svamitva_best.pt")
    logger.info("\nRun 'streamlit run app.py' to launch the unified dashboard.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
