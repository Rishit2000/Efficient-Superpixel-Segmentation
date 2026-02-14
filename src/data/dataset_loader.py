"""
Implements data loading and preprocessing for the Cityscapes dataset.
"""

from pathlib import Path
from typing import Tuple, Optional, Callable
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

class CityscapesDataset(Dataset):
    """
    Cityscapes Dataset for Semantic Segmentation

    Directory structure expected:
    dataset/Cityscapes/
        leftImg8bit/
            train/
                city/
                    *_leftImg8bit.png
            val/
        gtFine/
            train/
                city/
                    *_gtFine_labelIds.png
            val/
    """

    # Mapping from original IDs to training ID based on Cityscapes official mapping
    ID_TO_TRAINID = {
        -1: 255,  # void/unlabeled
        0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
        7: 0,   # road
        8: 1,   # sidewalk
        9: 255, 10: 255,
        11: 2,  # building
        12: 3,  # wall
        13: 4,  # fence
        14: 255, 15: 255, 16: 255,
        17: 5,  # pole
        18: 255,
        19: 6,  # traffic light
        20: 7,  # traffic sign
        21: 8,  # vegetation
        22: 9,  # terrain
        23: 10, # sky
        24: 11, # person
        25: 12, # rider
        26: 13, # car
        27: 14, # truck
        28: 15, # bus
        29: 255, 30: 255,
        31: 16, # train
        32: 17, # motorcycle
        33: 18, # bicycle
    }

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (1024, 2048)
    ):
        """
        Args:
            root: Root directory of Cityscapes dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on labels
            image_size: Target size for images (height, width)
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size

        # Paths
        self.images_dir = self.root / 'leftImg8bit' / split
        self.labels_dir = self.root / 'gtFine' / split

        # Verify paths exist
        if not self.images_dir.exists():
            raise RuntimeError(f"Image directory not found: {self.images_dir}")
        if not self.labels_dir.exists() and split != 'test':
            raise RuntimeError(f"Label directory not found: {self.labels_dir}")

        # Collect image and label pairs
        self.images = []
        self.labels = []
        self._collect_files()

        print(f"Loaded {len(self.images)} {split} images from Cityscapes dataset")

    def _collect_files(self):
        """Collect all image and label file paths"""
        for city_dir in sorted(self.images_dir.iterdir()):
            if not city_dir.is_dir():
                continue

            # Get all images in this city
            for img_path in sorted(city_dir.glob('*_leftImg8bit.png')):
                self.images.append(img_path)

                # Find corresponding label
                if self.split != 'test':
                    # Convert image filename to label filename
                    # e.g., aachen_000000_000019_leftImg8bit.png ->
                    #       aachen_000000_000019_gtFine_labelIds.png
                    label_name = img_path.name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    label_path = self.labels_dir / city_dir.name / label_name

                    if not label_path.exists():
                        raise RuntimeError(f"Label not found: {label_path}")

                    self.labels.append(label_path)

    def _convert_label(self, label: np.ndarray) -> np.ndarray:
        """Convert label IDs to training IDs"""
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.ID_TO_TRAINID.items():
            label_copy[label == k] = v
        return label_copy

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor of shape (3, H, W)
            label: Tensor of shape (H, W) with values in [0, 18] or 255 (ignore)
        """
        # Load image
        image = Image.open(self.images[index]).convert('RGB')

        # Load label (if available)
        if self.split != 'test':
            label = Image.open(self.labels[index])
            label = np.array(label, dtype=np.uint8)
            label = self._convert_label(label)
            label = Image.fromarray(label)
        else:
            # Create dummy label for test set
            label = Image.fromarray(np.zeros(image.size[::-1], dtype=np.uint8))

        # resize 
        image = TF.resize(image, self.image_size, interpolation=Image.BILINEAR)
        #label = TF.resize(label, self.image_size, interpolation=Image.NEAREST)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = TF.to_tensor(image)
            image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = torch.from_numpy(np.array(label, dtype=np.int64))

        return image, label

    def get_image_path(self, index: int) -> str:
        """Get the path of an image by index"""
        return str(self.images[index])
