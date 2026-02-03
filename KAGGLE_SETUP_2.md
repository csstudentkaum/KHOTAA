# Kaggle Setup - ANY Model
## KHOTAA Diabetic Foot Ulcer Classification

---


**These replacements appear in:**
- ✅ Notebook naming
- ✅ Checkpoint directories
- ✅ Results file names
- ✅ Model experiment names

---

## ✅ Setup Steps (Do These FIRST)

### Step 1: Create Notebook
1. Go to kaggle.com → **Create → New Notebook**
2. Name: `{model_name}`
3. Select **Python**

### Step 2: Enable GPU
1. Click **⚙️ Settings**
2. Select **GPU P100** (or T4)
3. Click **Save**

### Step 3: Add Dataset
1. Click **Input files**
2. Search: `dfu-dataset-annotated-into-4-classes`
3. Click **Add**

**Dataset Link:**
```
https://www.kaggle.com/datasets/khalidsiddiqui2003/dfu-dataset-annotated-into-4-classes
```

---

## 📋 COPY THESE 9 CELLS

---

### **CELL 1: GPU Verification & Setup**

```python
# GPU VERIFICATION AND SETUP
import os
import sys
import torch

os.chdir('/kaggle/working')
sys.path.insert(0, '/kaggle/working')

gpu_available = torch.cuda.is_available()
print(f"GPU Available: {gpu_available}")

if gpu_available:
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Device Name: {gpu_name}")
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_memory_gb:.2f} GB")
    torch.cuda.set_device(0)
    current_device = torch.cuda.current_device()
    print(f"Current GPU Device Index: {current_device}")
    print("\nGPU Status: Ready for Training")
else:
    print("WARNING: GPU not available")

print(f"\nPyTorch Version: {torch.__version__}")
import sys as sys_module
print(f"Python Version: {sys_module.version.split()[0]}")
```

---

### **CELL 2: Clone KHOTAA Repository**

```python
# Clone KHOTAA repository to get all utils modules
import os
import subprocess
import sys

os.chdir('/kaggle/working')

# Clone the main repository (without specific branch)
repo_url = 'https://github.com/csstudentkaum/KHOTAA.git'

print(f"Cloning repository: {repo_url}")
subprocess.run(['git', 'clone', repo_url], check=True)

# Navigate to models/classification directory
os.chdir('/kaggle/working/KHOTAA/models/classification')

# Print confirmation message 
print("Repository cloned")
```

---

### **CELL 3: Imports & Configuration**

```python
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Kaggle Setup - Set working directory and paths
os.chdir('/kaggle/working/KHOTAA/models/classification')
sys.path.insert(0, '/kaggle/working/KHOTAA/models')
sys.path.insert(0, '/kaggle/working/KHOTAA/models/utils')

# Kaggle Paths - Define where to save checkpoints and results
CHECKPOINT_BASE_DIR = '/kaggle/working/checkpoints'
RESULTS_BASE_DIR = '/kaggle/working/results'
DATASET_ROOT = '/kaggle/input/dfu-dataset-annotated-into-4-classes'

from dataset_loader import SplitFolderDatasetLoader
from dataset_preprocessing import DFUPreprocessing
from utils.checkpoint_manager import CheckpointManager
from utils.training_engine import TrainingEngine, create_optimizer
from utils.metrics_evaluator import (
    calculate_metrics, print_metrics, plot_confusion_matrix,
    plot_roc_curve, plot_training_history
)

print("Imports complete")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

---

## 💻 CELLS 4-8: Changes to Make

### **CELL 4: Load Dataset**

**Change FROM:**
```python
loader = SplitFolderDatasetLoader(root_dir='../../dataset')
```

**Change TO:**
```python
loader = SplitFolderDatasetLoader(root_dir=DATASET_ROOT)
```

**AND change FROM:**
```python
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
```

**Change TO:**
```python
test_loader = DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=0,
    pin_memory=True
)
```

---

### **CELL 5: Model Definition**

Copy as-is from your notebook. No changes needed.

---

### **CELL 6: Training Loop**

**Replace ALL `efficientnetv2s` with `{model_name}`**

**Change FROM:**
```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
checkpoint_manager = CheckpointManager(base_dir='checkpoints', experiment_name=f'efficientnetv2s_fold{fold}')
```

**Change TO:**
```python
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)
checkpoint_manager = CheckpointManager(
    base_dir=CHECKPOINT_BASE_DIR, 
    experiment_name=f'{model_name}_fold{fold}'
)
```

---

### **CELL 7 & 8: Evaluation, Plots & Save Results**

**These cells have the SAME changes. In BOTH cells:**

**Change FROM:**
```python
model_results_dir = 'results/efficientnetv2s'
results_file = f'{model_results_dir}/efficientnetv2s_results.json'
```

**Change TO:**
```python
model_results_dir = f'{RESULTS_BASE_DIR}/{model_name}'
results_file = f'{model_results_dir}/{model_name}_results.json'
```

**Replace {model_name} with your model name .**

---

### **CELL 9: Download Results for Kaggle**

**Replace ALL `efficientnetv2s` with `{model_name}`**
```python
# Download Results for Kaggle
import shutil

print("\n" + "="*60)
print("PREPARING RESULTS FOR DOWNLOAD")
print("="*60)

# Create output directory for Kaggle downloads
os.makedirs('/kaggle/output', exist_ok=True)

CHECKPOINT_BASE_DIR = '/kaggle/output/checkpoints'
RESULTS_BASE_DIR = '/kaggle/output/results'

# Archive checkpoints
if os.path.exists(CHECKPOINT_BASE_DIR):
    output_file = '/kaggle/output/efficientnetv2s_checkpoints'
    shutil.make_archive(output_file, 'zip', CHECKPOINT_BASE_DIR)
    size_mb = os.path.getsize(f'{output_file}.zip') / 1e6
    print(f"Checkpoints archived ({size_mb:.1f} MB)")
else:
    print("Checkpoints directory not found")

# Archive results
if os.path.exists(RESULTS_BASE_DIR):
    output_file = '/kaggle/output/efficientnetv2s_results'
    shutil.make_archive(output_file, 'zip', RESULTS_BASE_DIR)
    size_mb = os.path.getsize(f'{output_file}.zip') / 1e6
    print(f"Results archived ({size_mb:.1f} MB)")
else:
    print("Results directory not found")

print("\n" + "="*60)
print("Both files available in Kaggle Output section")
print("Download them directly from Kaggle interface")
print("="*60)
```

---

## ✅ THAT'S IT!

Run all 9 cells → Download your ZIP files from Kaggle **Output** section.
