"""
Superpixel Association (Unfolding) Module
Takes the per-superpixel class predictions and "unfolds" them
using the full-resolution SLIC map to create the final segmentation.
"""
import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F

class SuperpixelAssociation(nn.Module):
    """
    Projects per-superpixel logits back to the dense pixel space.
    
    This is the "unfolding" or "painting" step. It uses the full-resolution
    segment map as an index to gather the correct logit vector for each pixel.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        logits: torch.Tensor, 
        full_segment_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            logits: (B, N_max, Num_Classes)
                The class predictions for each superpixel token.
            full_segment_map: (B, H, W)
                The original full-res SLIC map (with integer segment IDs).
        
        Returns:
            dense_logits: (B, Num_Classes, H, W)
                The final dense segmentation map for the loss function.
        """
        B, N_max, Num_Classes = logits.shape
        _, H, W = full_segment_map.shape
        
        # 1. Flatten the segment map to (B, H*W)
        map_flat = full_segment_map.view(B, -1)
        
        # 2. Expand the flat map to match the logit's class dimension
        # Shape: (B, H*W) -> (B, H*W, 1) -> (B, H*W, Num_Classes)
        # This creates an index tensor where each (H*W) pixel location
        # has a segment ID, repeated Num_Classes times.
        map_flat_idx = map_flat.unsqueeze(2).expand(-1, -1, Num_Classes)
        
        # 3. Gather logits
        # torch.gather(input, dim, index)
        # - input: logits (B, N_max, C)
        # - dim: 1 (the N_max dimension)
        # - index: map_flat_idx (B, H*W, C)
        #
        # This operation selects logits[b, map_flat_idx[b, i, c], c]
        # which simplifies to logits[b, map_flat[b, i], c].
        # It's "painting" the logit vector for segment 'k' at all
        # pixel locations 'i' where map_flat[b, i] == k.
        output_flat = torch.gather(logits, dim=1, index=map_flat_idx)
        
        # 4. Un-flatten and permute to standard (B, C, H, W)
        output = output_flat.view(B, H, W, Num_Classes)
        output = output.permute(0, 3, 1, 2)

        output = F.interpolate(
            output,
            size=(1024, 2048),
            mode='bilinear',
            align_corners=False
        )
        
        return output
