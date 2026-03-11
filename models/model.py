"""
Unified SVAMITVA segmentation model (SegFormer encoder + UPerNet/FPN decoder).

This model predicts all raster outputs directly:
- building_mask, roof_type_mask
- road_mask, road_centerline_mask
- waterbody_mask, waterbody_line_mask, waterbody_point_mask
- utility_line_mask, utility_point_mask
- bridge_mask, railway_mask

YOLO-based point detections are integrated at inference-time in `inference/predict.py`
and fused with point masks.
"""

import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_encoder import DEFAULT_SEGFORMER_MODEL, SegformerEncoder
from .decoder import FPNDecoder
from .heads import create_all_heads

logger = logging.getLogger(__name__)


class EnsembleSvamitvaModel(nn.Module):
    """
    Unified Production Architecture for SVAMITVA Feature Extraction.

    Integrates:
    - SegFormer-B4 Backbone (Multi-scale Mix Transformer)
    - FPN Decoder with CBAM Attention
    - Specialized Task Heads (Building, Line, Detection)
    """

    def __init__(
        self,
        num_roof_classes: int = 5,
        pretrained: bool = True,
        model_name: str = DEFAULT_SEGFORMER_MODEL,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1. Backbone (SegFormer)
        self.encoder = SegformerEncoder(
            model_name=model_name, load_pretrained=pretrained, freeze=False
        )

        # SegFormer outputs exactly 4 levels: 64, 128, 320, 512 channels
        # Order is intrinsically hierarchical (Scale 1 -> Scale 4)
        self.fpn_in_channels = {
            "feat_s1": self.encoder.feature_channels[0],  # 64
            "feat_s2": self.encoder.feature_channels[1],  # 128
            "feat_s3": self.encoder.feature_channels[2],  # 320
            "feat_s4": self.encoder.feature_channels[3],  # 512
        }

        # 2. Decoder (FPN + CBAM)
        self.decoder = FPNDecoder(in_channels=self.fpn_in_channels, out_channels=256)

        # 3. Heads (11 tasks)
        self.heads = create_all_heads(
            in_channels=256, num_roof_classes=num_roof_classes, dropout=dropout
        )

    def forward(self, x: torch.Tensor, task: str = "all") -> Dict[str, torch.Tensor]:
        """
        Forward pass through the unified pipeline.

        Args:
            x: Input image tensor (B, 3, H, W)
            task: Task filter (e.g., "buildings", "roads", or "all")
        """
        # Feature extraction
        backbone_feats_list = self.encoder(x)
        
        # Package into a dict mapping the exact names expected by the decoder
        backbone_feats = {
            "feat_s1": backbone_feats_list[0],  # 64 channels
            "feat_s2": backbone_feats_list[1],  # 128 channels
            "feat_s3": backbone_feats_list[2],  # 320 channels
            "feat_s4": backbone_feats_list[3],  # 512 channels
        }

        # Multi-scale fusion
        fused_feat = self.decoder(backbone_feats)
        target_size = x.shape[2:]

        outputs = {}

        task_norm = task.lower().strip()
        run_all = task_norm in {"all", "*", "full"}

        # Buildings + roof
        if run_all or task_norm in {
            "buildings",
            "building",
            "building_mask",
            "roof",
            "roof_type",
            "roof_type_mask",
        }:
            mask, roof = self.heads["building"](fused_feat)
            outputs["building_mask"] = F.interpolate(
                mask, size=target_size, mode="bilinear", align_corners=False
            )
            outputs["roof_type_mask"] = F.interpolate(
                roof, size=target_size, mode="bilinear", align_corners=False
            )

        # Roads
        if run_all or task_norm in {"roads", "road", "road_mask"}:
            outputs["road_mask"] = F.interpolate(
                self.heads["road"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if run_all or task_norm in {
            "roads",
            "road",
            "road_centerline",
            "road_centerline_mask",
        }:
            outputs["road_centerline_mask"] = F.interpolate(
                self.heads["road_centerline"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        # Water
        if run_all or task_norm in {"water", "waterbody", "waterbody_mask"}:
            outputs["waterbody_mask"] = F.interpolate(
                self.heads["waterbody"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if run_all or task_norm in {
            "water",
            "waterbody",
            "waterbody_line",
            "waterbody_line_mask",
        }:
            outputs["waterbody_line_mask"] = F.interpolate(
                self.heads["waterbody_line"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if run_all or task_norm in {
            "water",
            "waterbody",
            "waterbody_point",
            "waterbody_point_mask",
        }:
            outputs["waterbody_point_mask"] = F.interpolate(
                self.heads["waterbody_point"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        # Utilities
        if run_all or task_norm in {"utilities", "utility", "utility_line_mask"}:
            outputs["utility_line_mask"] = F.interpolate(
                self.heads["utility_line"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if run_all or task_norm in {
            "utilities",
            "utility",
            "utility_point",
            "utility_point_mask",
        }:
            outputs["utility_point_mask"] = F.interpolate(
                self.heads["utility_point"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        # Infrastructure
        if run_all or task_norm in {"railway", "railway_mask"}:
            outputs["railway_mask"] = F.interpolate(
                self.heads["railway"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if run_all or task_norm in {"bridge", "bridge_mask"}:
            outputs["bridge_mask"] = F.interpolate(
                self.heads["bridge"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        return outputs

    def freeze_backbone(self):
        """Freeze SAM2 encoder for head-only training."""
        self.encoder.freeze()

    def unfreeze_backbone(self):
        """Unfreeze SAM2 encoder for full fine-tuning."""
        self.encoder.unfreeze()

    def get_param_groups(self, base_lr: float = 1e-4) -> list:
        """Categorize parameters for LR scaling."""
        backbone_params = list(self.encoder.parameters())
        head_params = list(self.decoder.parameters()) + list(self.heads.parameters())

        return [
            {"params": head_params, "lr": base_lr},
            {"params": backbone_params, "lr": base_lr * 0.1},  # Slower backbone LR
        ]


SvamitvaModel = EnsembleSvamitvaModel
