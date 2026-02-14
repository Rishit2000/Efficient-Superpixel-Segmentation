"""
Main Model File: DecoupledSuperpixelViT
Combines all the modules into a single end-to-end model.
"""
import torch
import torch.nn as nn
from typing import Tuple

# Import all the modules
from src.models.feature_extractor import PixelFeatureExtractor
from src.models.superpixel_tokenizer import SuperpixelTokenization
from src.models.superpixel_classifier import SuperpixelClassifier
from src.models.superpixel_associator import SuperpixelAssociation

class DecoupledSuperpixelViT(nn.Module):
    """
    The full, end-to-end model.
    Connects the CNN backbone, SLIC tokenization, ViT encoder,
    and Association/Unfolding module.
    """
    def __init__(
        self,
        # Backbone args
        feature_dim: int = 256,
        output_stride: int = 8,
        freeze_backbone: bool = True,  # NEW: Freeze ResNet by default
        # Tokenizer args
        n_segments: int = 2048,
        # Classifier args
        num_classes: int = 19,
        num_layers: int = 4,
        num_heads: int = 4,
        # Loss args
        ignore_index: int = 255
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # --- 1. CNN Backbone ---
        self.backbone = PixelFeatureExtractor(
            feature_dim=feature_dim,
            output_stride=output_stride,
            freeze_backbone=freeze_backbone
        )
        
        # --- 2. Tokenizer (SLIC + Pool) ---
        self.tokenizer = SuperpixelTokenization(
            n_segments=n_segments
        )
        
        # --- 3. Classifier (ViT + Head) ---
        self.classifier = SuperpixelClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        # --- 4. Association (Unfolder) ---
        self.association = SuperpixelAssociation()
        
        # --- 5. Loss Function ---
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """
        Full forward pass.
        
        Args:
            images: (B, 3, H, W) - Input images
            labels: (B, H, W) - Optional ground truth labels
        
        Returns:
            dense_logits: (B, C, H, W) - Final pixel-level predictions
            loss: (torch.Tensor) - Calculated loss, or None if no labels given
        """
        # --- Stage 1: Pixel Feature Extraction ---
        # (B, 3, H, W) -> (B, F, H/8, W/8)
        features = self.backbone(images)
        
        # --- Stage 2: Superpixel Tokenization ---
        # (images, features) -> (tokens, map, mask, centroids)
        tokens, full_map, mask, centroids = self.tokenizer(images, features)
        
        # --- Stage 3: Superpixel Classification ---
        # (tokens, centroids, mask) -> (B, N_max, C)
        superpixel_logits = self.classifier(tokens, centroids, mask)
        
        # --- Stage 4: Superpixel Association ---
        # (superpixel_logits, full_map) -> (B, C, H, W)
        dense_logits = self.association(superpixel_logits, full_map)
        
        # --- 5. Loss Calculation ---
        loss = None
        if labels is not None:
            # Ensure labels are long
            labels = labels.long()
            loss = self.loss_fn(dense_logits, labels)
            
        return dense_logits, loss