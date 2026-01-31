"""
Preprocessing for DFU classification.
Unified across all models.
"""

import torch
from torchvision import transforms


class DFUPreprocessing:
    """
    Preprocessing for DFU dataset.
    Provides train and valid/test transforms.
    """
    
    def __init__(self):
        """Initialize with default DFU preprocessing configuration."""
        
        # ImageNet normalization constants
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
        # Build transforms
        self.train_transforms = self._build_train()
        self.valid_test_transforms = self._build_valid_test()
        
        print("[DFUPreprocessing] Initialized")
        print(f"[DFUPreprocessing] Image size: 224x224")
        print(f"[DFUPreprocessing] Train: with augmentation")
        print(f"[DFUPreprocessing] Valid/Test: no augmentation")
    
    def _build_train(self):
        """
        Build training transforms with augmentation.

        Preprocessing steps:
        1. Resize to 224x224 (standard input size)
        2. RandomHorizontalFlip (50% chance) - horizontal flip
        3. RandomVerticalFlip (30% chance) - vertical flip
        4. RandomRotation (±15°) - simulates camera angle variation
        5. ColorJitter - simulates lighting conditions
           - brightness: ±20% (home lighting variance)
           - contrast: ±20% (flash/no flash)
           - saturation: ±5% (conservative, preserves tissue color)
           - hue: ±2% (phone camera white balance differences)
        6. GaussianBlur - simulates focus variance and motion blur
        7. ToTensor - converts PIL image to tensor [0,1]
        8. Normalize - ImageNet normalization for pretrained models
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),               # Resize to standard size
            transforms.RandomHorizontalFlip(p=0.5),      # 50% horizontal flip
            transforms.RandomVerticalFlip(p=0.3),        # 30% vertical flip
            transforms.RandomRotation(degrees=15),       # ±15° rotation
            transforms.ColorJitter(                      # Color augmentation
                brightness=0.2,                          #   ±20% brightness
                contrast=0.2,                            #   ±20% contrast
                saturation=0.05,                         #   ±5% saturation (conservative)
                hue=0.02                                 #   ±2% hue (white balance)
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # Focus/shake blur
            transforms.ToTensor(),                       # Convert to tensor [0,1]
            transforms.Normalize(self.mean, self.std)    # ImageNet normalization
        ])
    
    def _build_valid_test(self):
        """
        Build validation/test transforms without augmentation.
        
        Preprocessing steps:
        1. Resize to 224x224 (standard input size)
        2. ToTensor - converts PIL image to tensor [0,1]
        3. Normalize - ImageNet normalization for pretrained models
        
        No augmentation to ensure consistent evaluation.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),              # Resize to standard size
            transforms.ToTensor(),                       # Convert to tensor [0,1]
            transforms.Normalize(self.mean, self.std)    # ImageNet normalization
        ])
    
    def get_train_transforms(self):
        """Get training transforms (with augmentation)."""
        return self.train_transforms
    
    def get_valid_test_transforms(self):
        """Get validation/test transforms (no augmentation)."""
        return self.valid_test_transforms