"""
Superpixel Tokenization Module (Decoupled)
1. Runs unsupervised SLIC.
2. Uses the resulting map to pool hypercolumn features.
3. Pads the tokens to create a uniform batch for the Transformer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation import slic, mark_boundaries
from cuda_slic.slic import slic as cuda_slic
from torch_scatter import scatter_mean
import numpy as np
from typing import Tuple

class SuperpixelTokenization(nn.Module):
    """
    Combines unsupervised SLIC with feature pooling to generate superpixel tokens.
    
    This module is intentionally "decoupled." The SLIC segmentation is
    treated as a non-differentiable preprocessing step. Gradients will
    flow from the pooled tokens back to the PixelFeatureExtractor, but
    not to the SLIC algorithm itself.
    """
    
    # ImageNet mean/std for un-normalization (SLIC works best on color)
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self, n_segments: int = 2048, compactness: int = 10):
        """
        Args:
            n_segments: Approximate number of superpixels (SLIC centers)
            compactness: Balances color/space proximity in SLIC.
        """
        super().__init__()
        self.n_segments = n_segments
        self.compactness = compactness

    def _un_normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Un-normalizes a single (C, H, W) image tensor."""
        mean = self.IMAGENET_MEAN.view(3, 1, 1).to(image.device)
        std = self.IMAGENET_STD.view(3, 1, 1).to(image.device)
        img_unnorm = (image * std) + mean
        return img_unnorm.clamp(0, 1)

    def _run_slic(self, image: torch.Tensor) -> np.ndarray:
        """
        Runs SLIC on a single image.
        NOTE: SLIC runs on the CPU.
        
        Args:
            image: A single (C, H, W) image tensor, *normalized*.
        
        Returns:
            A (H, W) numpy array of superpixel labels (0-indexed).
        """
        # 1. Un-normalize for better color-based segmentation
        img_unnorm = self._un_normalize_image(image)
        
        # 2. SLIC expects (H, W, C) numpy array on CPU
        img_np = img_unnorm.permute(1, 2, 0).cpu().numpy()
        
        # 3. Run SLIC.
        segments = cuda_slic(
            img_np, 
            n_segments=self.n_segments, 
            compactness=self.compactness
        )
        return segments


    def _compute_centroids(
        self, 
        segment_map_small: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the normalized [0, 1] (x, y) centroids for each segment.
        
        Args:
            segment_map_small: (B, H_feat, W_feat) tensor of segment IDs.
            
        Returns:
            normalized_centroids: (B, N_max, 2) tensor of (x,y) centroids.
        """
        B, H_feat, W_feat = segment_map_small.shape
        device = segment_map_small.device

        # Create (x, y) coordinate grids
        # y_coords: (H_feat, W_feat)
        # x_coords: (H_feat, W_feat)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H_feat, device=device, dtype=torch.float32),
            torch.arange(W_feat, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Normalize coordinates to [0, 1]
        y_coords_norm = y_coords / (H_feat - 1)
        x_coords_norm = x_coords / (W_feat - 1)
        
        # Stack to (H_feat, W_feat, 2) and flatten to (H_feat*W_feat, 2)
        coords_flat = torch.stack([x_coords_norm, y_coords_norm], dim=-1).view(-1, 2)
        
        # Flatten segment map (B, H_feat*W_feat)
        segment_map_flat = segment_map_small.view(B, -1)

        # Use scatter_mean to compute average (x,y) for each segment ID
        centroid_list = []
        for i in range(B):
            map_i = segment_map_flat[i] # (H_feat*W_feat)
            # Tile coords for this batch item
            coords_i = coords_flat # (H_feat*W_feat, 2)
            
            # Compute mean (x,y) for each segment
            centroids_i = scatter_mean(src=coords_i, index=map_i, dim=0)
            centroid_list.append(centroids_i)
            
        # Pad the centroids (same logic as tokens)
        max_segments = max(c.shape[0] for c in centroid_list)
        padded_centroids = torch.zeros((B, max_segments, 2), device=device)
        
        for i, centroids in enumerate(centroid_list):
            num_segs = centroids.shape[0]
            padded_centroids[i, :num_segs] = centroids
            
        return padded_centroids


    def forward(
        self,
        images: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates superpixel tokens from features using SLIC on images.

        Args:
            images: Original, full-res image batch (B, 3, H, W).
            features: Hypercolumn feature map (B, F, H, W) - FULL RESOLUTION!

        Returns:
            padded_tokens: (B, N_max, F)
                The superpixel tokens, padded to the max tokens in the batch.
            full_segment_map: (B, H, W)
                The original full-res SLIC map. Needed for final "unfolding."
            attention_mask: (B, N_max)
                A boolean mask (True=real token, False=padding).
        """
        B, Feat, H_feat, W_feat = features.shape
        _, _, H_img, W_img = images.shape

        # Verify that features and images have same spatial dimensions
        assert H_feat == H_img and W_feat == W_img, \
            f"Features ({H_feat}×{W_feat}) and images ({H_img}×{W_img}) must have same resolution!"

        # 1. Get SLIC maps for the batch (CPU loop)
        # We run this on the *original full-res images*
        segment_maps_cpu = [self._run_slic(img) for img in images]

        # Stack and move to device
        full_segment_map = torch.from_numpy(np.stack(segment_maps_cpu)).to(features.device)

        # 2. NO DOWNSAMPLING! Use full-resolution segment map directly
        segment_map_fullres = full_segment_map  # (B, H, W)

        # 3. Pool features based on segment maps

        # Flatten features and segment maps
        features_flat = features.permute(0, 2, 3, 1).reshape(B, -1, Feat) # (B, H*W, F)
        segment_map_flat = segment_map_fullres.reshape(B, -1)             # (B, H*W)
        
        # 4. Pool and pad (batched)
        # We must pad because each image will have a slightly different
        # number of superpixels (SLIC is approximate).
        token_list = []
        num_segments_list = []

        for i in range(B):
            feats_i = features_flat[i]     # (H_feat*W_feat, F)
            map_i = segment_map_flat[i]    # (H_feat*W_feat)
            
            # scatter_mean groups all `feats_i` vectors by the `map_i` index
            # and computes their mean.
            # dim=0 means we pool along the 0-th dimension.
            tokens_i = scatter_mean(src=feats_i, index=map_i, dim=0)
            
            token_list.append(tokens_i)
            num_segments_list.append(tokens_i.shape[0])

        # Find max segments and create padded tensors
        max_segments = max(num_segments_list)
        
        padded_tokens = torch.zeros((B, max_segments, Feat), device=features.device)
        attention_mask = torch.zeros((B, max_segments), dtype=torch.bool, device=features.device)

        for i, tokens in enumerate(token_list):
            num_segs = tokens.shape[0]
            padded_tokens[i, :num_segs] = tokens
            attention_mask[i, :num_segs] = True # True = real token

        # 5. Compute Centroids (using full-resolution segment map)
        padded_centroids = self._compute_centroids(segment_map_fullres)

        return padded_tokens, full_segment_map, attention_mask, padded_centroids 
