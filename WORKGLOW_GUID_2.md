# DFU Dataset Setup Guide

This guide walks through the complete dataset setup process for the Diabetic Foot Ulcer (DFU) classification project.

---

## Step 1: Imports

Import all required libraries and modules:

```python
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image

# Add project to path
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

# Import modules
from models.classification.dataset_loader import SplitFolderDatasetLoader
from models.classification.dataset_preprocessing import DFUPreprocessing

print("✓ Modules imported")
```

---

## Step 2: Load Dataset (Paths + Labels)

Initialize the dataset loader and load train, validation, and test splits:

```python
# Initialize loader
loader = SplitFolderDatasetLoader(root_dir="dataset")

# Get paths and labels
X_train, y_train = loader.load_split_paths("train", shuffle=True)
X_valid, y_valid = loader.load_split_paths("valid", shuffle=False)
X_test, y_test = loader.load_split_paths("test", shuffle=False)

# Get classes info
classes = loader.get_classes()
num_classes = loader.get_num_classes()

print(f"✓ Data loaded: {len(X_train)} train, {len(X_valid)} valid, {len(X_test)} test")
```

---

## Step 3: Initialize Preprocessing

Set up data preprocessing and transformations:

```python
# Initialize preprocessing
preprocessing = DFUPreprocessing()

# Get transforms
train_tfms = preprocessing.get_train_transforms()
valid_test_tfms = preprocessing.get_valid_test_transforms()

print("✓ Preprocessing ready")
```

---

## Step 4: Create Dataset Class

Define a custom PyTorch Dataset class:

```python
class DFUDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]

        # Apply preprocessing
        if self.transform:
            img = self.transform(img)

        return img, label
```

---

## Step 5: Create Datasets

Create dataset instances for training, validation, and testing:

```python
train_dataset = DFUDataset(X_train, y_train, transform=train_tfms)
val_dataset = DFUDataset(X_valid, y_valid, transform=valid_test_tfms)
test_dataset = DFUDataset(X_test, y_test, transform=valid_test_tfms)

print(f"✓ Datasets created:")
print(f"  Train: {len(train_dataset)} (with augmentation)")
print(f"  Valid: {len(val_dataset)} (no augmentation)")
print(f"  Test: {len(test_dataset)} (no augmentation)")
```

---

## Summary

You now have three ready-to-use PyTorch datasets:
- **train_dataset**: Training data with augmentation
- **val_dataset**: Validation data without augmentation
- **test_dataset**: Test data without augmentation

These datasets can be used with PyTorch DataLoaders for model training and evaluation.
