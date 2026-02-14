"""
Pixel Feature Extraction Module
Implements hypercolumn feature extraction using CNN backbones (ResNet-50, etc.)
"""
from pathlib import Path
from src.data.dataset_loader import CityscapesDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, List


class PixelFeatureExtractor(nn.Module):
    """
    Extracts multi-scale hypercolumn features from a CNN backbone.

    Following the paper:
    - Extract features from stage-1 (stride 2), stage-3 (stride 8), stage-5 (stride 32)
    - Project each to 256 channels via MLP
    - Resize all to stride 8
    - Combine with element-wise addition
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 256,
        output_stride: int = 8,
        freeze_backbone: bool = True  # NEW: Freeze backbone by default
    ):
        """
        Args:
            backbone: Backbone architecture ('resnet50', 'resnet101')
            pretrained: Whether to use pretrained weights
            feature_dim: Output feature dimension (paper uses 256)
            output_stride: Output stride of features (paper uses 8)
            freeze_backbone: Whether to freeze backbone weights (no backprop)
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.output_stride = output_stride
        self.freeze_backbone = freeze_backbone

        # Load backbone
        if backbone == 'resnet50':
            if pretrained:
                weights = ResNet50_Weights.IMAGENET1K_V2
                self.backbone = resnet50(weights=weights)
            else:
                self.backbone = resnet50(weights=None)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")

        # Remove fully connected layers
        self.backbone = nn.ModuleDict({
            'conv1': self.backbone.conv1,
            'bn1': self.backbone.bn1,
            'relu': self.backbone.relu,
            'maxpool': self.backbone.maxpool,
            'layer1': self.backbone.layer1,  # stride 4
            'layer2': self.backbone.layer2,  # stride 8
            'layer3': self.backbone.layer3,  # stride 16
            'layer4': self.backbone.layer4,  # stride 32
        })

        # Freeze backbone if requested
        if self.freeze_backbone:
            print("  Freezing backbone (no gradient computation)")
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Set backbone to eval mode permanently
            for module in self.backbone.values():
                module.eval()

        # Feature channels at different stages
        # For ResNet-50:
        # After layer1: 256 channels, stride 4
        # After layer2: 512 channels, stride 8
        # After layer3: 1024 channels, stride 16
        # After layer4: 2048 channels, stride 32

        # We'll use features after maxpool (stride 2), layer2 (stride 8), layer4 (stride 32)
        self.stage_channels = {
            'stage1': 64,    # After maxpool (stride 2)
            'stage3': 512,   # After layer2 (stride 8)
            'stage5': 2048,  # After layer4 (stride 32)
        }

        # MLPs to project features to uniform dimension
        self.stage1_proj = nn.Sequential(
            nn.Conv2d(self.stage_channels['stage1'], feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

        self.stage3_proj = nn.Sequential(
            nn.Conv2d(self.stage_channels['stage3'], feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

        self.stage5_proj = nn.Sequential(
            nn.Conv2d(self.stage_channels['stage5'], feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

        # Additional refinement layer
        self.refinement = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def train(self, mode: bool = True):
        """
        Override train() to keep backbone in eval mode when frozen.
        """
        super().train(mode)
        if self.freeze_backbone:
            # Keep backbone in eval mode even when model is in training mode
            for module in self.backbone.values():
                module.eval()
        return self

    def _forward_backbone(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through backbone to extract multi-scale features

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Dictionary with features at different stages
        """
        features = {}

        # Initial conv
        x = self.backbone['conv1'](x)
        x = self.backbone['bn1'](x)
        x = self.backbone['relu'](x)
        x = self.backbone['maxpool'](x)  # stride 4

        # Stage 1 (we use features after maxpool as "stage 1")
        features['stage1'] = x  # stride 4, but we'll treat as stride 2 conceptually

        # Layer 1
        x = self.backbone['layer1'](x)  # stride 4

        # Layer 2
        x = self.backbone['layer2'](x)  # stride 8
        features['stage3'] = x

        # Layer 3
        x = self.backbone['layer3'](x)  # stride 16

        # Layer 4
        x = self.backbone['layer4'](x)  # stride 32
        features['stage5'] = x

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate hypercolumn features

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Hypercolumn features of shape (B, feature_dim, H, W) - FULL RESOLUTION
        """
        B, C, H, W = x.shape

        # Extract multi-scale features
        features = self._forward_backbone(x)

        # Project each feature map to feature_dim channels
        f1 = self.stage1_proj(features['stage1'])  # stride 4
        f3 = self.stage3_proj(features['stage3'])  # stride 8
        f5 = self.stage5_proj(features['stage5'])  # stride 32

        # Target size is FULL RESOLUTION (H, W)
        target_h, target_w = H, W

        # Resize all features to full resolution
        f1_resized = F.interpolate(f1, size=(target_h, target_w), mode='bilinear', align_corners=False)
        f3_resized = F.interpolate(f3, size=(target_h, target_w), mode='bilinear', align_corners=False)
        f5_resized = F.interpolate(f5, size=(target_h, target_w), mode='bilinear', align_corners=False)

        # Combine features with element-wise addition
        hypercolumn = f1_resized + f3_resized + f5_resized

        # Apply refinement
        #hypercolumn = self.refinement(hypercolumn)

        return hypercolumn

    def get_feature_dim(self) -> int:
        """Return the output feature dimension"""
        return self.feature_dim
