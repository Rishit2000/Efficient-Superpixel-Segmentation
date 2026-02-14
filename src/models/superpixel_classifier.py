"""
Superpixel Classifier (ViT Encoder + Classification Head)
"""
import torch
import torch.nn as nn
from typing import Tuple

class PositionalEncoder(nn.Module):
    """
    MLP that projects normalized (x,y) coordinates to a high-dim embedding.
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), # Input is 2D: (x, y)
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, normalized_centroids: torch.Tensor) -> torch.Tensor:
        # normalized_centroids shape: (B, N_max, 2)
        return self.mlp(normalized_centroids) # Output: (B, N_max, feature_dim)


class SuperpixelClassifier(nn.Module):
    """
    Applies a Transformer Encoder to the superpixel tokens to get global
    context, then classifies each refined token.
    """
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: Dimension of tokens (e.g., 256)
            num_classes: Number of output segmentation classes (e.g., 19 for Cityscapes)
            [cite_start]num_layers: Number of Transformer Encoder layers [cite: 13]
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # [cite_start]1. Positional Encoder [cite: 11]
        self.pos_encoder = PositionalEncoder(feature_dim)
        
        # [cite_start]2. ViT Encoder (Transformer) [cite: 12]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4, # Standard 4x FFN
            dropout=dropout,
            batch_first=True  # Ensures input/output is (B, N, F)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # [cite_start]3. Classification Head [cite: 14]
        self.classifier_head = nn.Linear(feature_dim, num_classes)

    def forward(
        self, 
        tokens: torch.Tensor, 
        centroids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tokens: (B, N_max, F) - Padded superpixel tokens.
            centroids: (B, N_max, 2) - Padded normalized (x,y) centroids.
            attention_mask: (B, N_max) - Boolean mask (True=real, False=padding).
        
        Returns:
            logits: (B, N_max, Num_Classes) - Class predictions for each token.
        """
        
        # 1. Add Positional Encodings
        pos_embedding = self.pos_encoder(centroids)
        tokens_with_pos = tokens + pos_embedding
        
        # 2. ViT Encoder
        # PyTorch Transformer mask expects True for *padded* (ignored) tokens,
        # so we must invert our mask.
        padding_mask = ~attention_mask # (B, N_max)
        
        refined_tokens = self.transformer_encoder(
            src=tokens_with_pos,
            src_key_padding_mask=padding_mask
        )
        
        # 3. Classification Head
        logits = self.classifier_head(refined_tokens)
        
        return logits
