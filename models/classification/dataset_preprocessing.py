"""
Preprocessing module for Diabetic Foot Ulcer (DFU) classification.
Similar structure to dataset_loader.py for consistency.
"""

import torch
from torchvision import transforms


class DFUPreprocessing:
    """
    Preprocessing manager for DFU classification.
    Provides standardized transforms across all models.
    """
    
    def __init__(
        self,
        use_horizontal_flip=True,
        rotation_degrees=10,
        brightness=0.15,
        contrast=0.15,
        saturation=0.05,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ):
        """
        Initialize DFU preprocessing.
        
        Args:
            use_horizontal_flip (bool): Apply horizontal flip. Default: True
            rotation_degrees (float): Max rotation angle. Default: 10
            brightness (float): Brightness jitter factor. Default: 0.15
            contrast (float): Contrast jitter factor. Default: 0.15
            saturation (float): Saturation jitter factor. Default: 0.05
            mean (tuple): Normalization mean (ImageNet default)
            std (tuple): Normalization std (ImageNet default)
        """
        self.use_horizontal_flip = use_horizontal_flip
        self.rotation_degrees = rotation_degrees
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.mean = mean
        self.std = std
        
        # Build transforms
        self._train_transforms = self._build_train_transforms()
        self._eval_transforms = self._build_eval_transforms()
        
        # Print configuration
        print(f"[DFUPreprocessing] Initialized")
        print(f"[DFUPreprocessing] Horizontal flip: {self.use_horizontal_flip}")
        print(f"[DFUPreprocessing] Rotation: ±{self.rotation_degrees}°")
        print(f"[DFUPreprocessing] Color jitter: brightness={self.brightness}, "
              f"contrast={self.contrast}, saturation={self.saturation}")
        print(f"[DFUPreprocessing] Normalization: ImageNet (mean={self.mean}, std={self.std})")
    
    def _build_train_transforms(self):
        """Build training transforms with augmentation."""
        transform_list = []
        
        if self.use_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        if self.rotation_degrees > 0:
            transform_list.append(transforms.RandomRotation(degrees=self.rotation_degrees))
        
        if self.brightness > 0 or self.contrast > 0 or self.saturation > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                    hue=0.0
                )
            )
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        return transforms.Compose(transform_list)
    
    def _build_eval_transforms(self):
        """Build evaluation transforms (no augmentation)."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
    
    def get_train_transforms(self):
        """Get training transforms."""
        return self._train_transforms
    
    def get_eval_transforms(self):
        """Get evaluation transforms."""
        return self._eval_transforms
    
    def unnormalize_tensor(self, tensor):
        """
        Unnormalize a tensor for visualization.
        
        Args:
            tensor: Normalized tensor (C, H, W)
        
        Returns:
            Unnormalized tensor in [0, 1] range
        """
        mean_t = torch.tensor(self.mean).view(3, 1, 1)
        std_t = torch.tensor(self.std).view(3, 1, 1)
        return (tensor * std_t + mean_t).clamp(0, 1)
    
    def tensor_to_image(self, tensor):
        """
        Convert normalized tensor to numpy image.
        
        Args:
            tensor: Normalized tensor (C, H, W)
        
        Returns:
            Numpy array (H, W, C) in [0, 1] range
        """
        unnorm = self.unnormalize_tensor(tensor)
        return unnorm.permute(1, 2, 0).numpy()


# Default configuration for consistency across all models
DEFAULT_CONFIG = {
    'use_horizontal_flip': True,
    'rotation_degrees': 10,
    'brightness': 0.15,
    'contrast': 0.15,
    'saturation': 0.05,
}
