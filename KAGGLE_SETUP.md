# Kaggle Setup Guide - KHOTAA

## 🎯 Overview

This guide helps you run the KHOTAA DFU classification models on **Kaggle** with GPU support for faster training.

---

## 📋 Prerequisites

- Kaggle account
- Dataset uploaded to Kaggle (or use public dataset)
- Basic Python knowledge

---

## 🚀 Step 1: Create Kaggle Notebook

1. Go to [kaggle.com](https://kaggle.com)
2. Click **Create** → **New Notebook**
3. Name it: `modelName`
4. Select **Python** as language

---

## ⚡ Step 2: Enable GPU

1. Click **⚙️ Settings** (top right)
2. Find **Accelerator**
3. Select **GPU P100** or **GPU T4**
4. Click **Save**

---

## 📊 Step 3: Add Dataset

### Option A: Public Kaggle Dataset 

1. In Settings, click **Input files**
2. Search for: `dfu-dataset-annotated-into-4-classes`
3. Click **Add**

**Dataset Link:**
```
https://www.kaggle.com/datasets/khalidsiddiqui2003/dfu-dataset-annotated-into-4-classes
```



### Option B: Upload Your Own

1. Click **Add Dataset** → **Upload new dataset**
2. Upload your dataset folder
3. Name it appropriately

---

## 💻 Cell-by-Cell Setup

### **Cell 1: GPU Verification**

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

**Expected Output:**
```
GPU Available: True
Number of GPUs: 1
GPU Device Name: Tesla P100-PCIE-16GB
GPU Memory: 16.00 GB
Current GPU Device Index: 0

GPU Status: Ready for Training

PyTorch Version: 2.0.0
Python Version: 3.10
```

---

### **Cell 2: Clone KHOTAA Project**

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

# Setup paths for imports
sys.path.insert(0, '/kaggle/working/KHOTAA/models')
sys.path.insert(0, '/kaggle/working/KHOTAA/models/utils')
sys.path.insert(0, '/kaggle/working/KHOTAA/models/classification')

print("Repository cloned successfully")
print(f"Current directory: {os.getcwd()}")
```

---

## 🔧 Key Configuration Changes

### Change 1: Dataset Path

**In Load Dataset cell, use:**

```python
# For Kaggle public dataset
loader = SplitFolderDatasetLoader(
    root_dir='/kaggle/input/dfu-dataset-annotated-into-4-classes'
)

# Or if you uploaded your own
loader = SplitFolderDatasetLoader(
    root_dir='/kaggle/input/your-dataset-name/dataset'
)
```

---

### Change 2: Checkpoint Path

**In Training cell, use:**

```python
checkpoint_manager = CheckpointManager(
    base_dir='/kaggle/working/checkpoints',
    experiment_name=f'efficientnetv2s_fold{fold}'
)
```

---

### Change 3: DataLoader Optimization

**For train_loader:**

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
```

**For val_loader:**

```python
val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)
```

**For test_loader:**

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

## ✅ Cell: Verify Results

### Before Final two Cells (after training cell)

```markdown
## Verify Training Results
```

**Python Cell:**

```python
# VERIFY CHECKPOINTS ARE SAVED
import os

results_dir = '/kaggle/working/checkpoints'

if os.path.exists(results_dir):
    print("Checkpoints directory exists")
    print(f"\nContents of {results_dir}:")
    
    for item in os.listdir(results_dir):
        print(f"  - {item}")
        
    # Count files
    total_files = sum(
        len(files) 
        for _, _, files in os.walk(results_dir)
    )
    print(f"\nTotal files saved: {total_files}")
else:
    print("Checkpoints directory not found")
```

---

## 📥 Cell: Download Results

### Markdown Cell

```markdown
## Download Training Results
```

### Python Cell

```python
# DOWNLOAD RESULTS
import shutil
import os

# Create output directory
os.makedirs('/kaggle/output', exist_ok=True)

# Paths
source_dir = '/kaggle/working/checkpoints'
output_path = '/kaggle/output/training_results'

if os.path.exists(source_dir):
    # Create zip file
    shutil.make_archive(output_path, 'zip', source_dir)
    
    print(f"Results saved to: {output_path}.zip")
    print("Available in Kaggle Output section for download")
    
    # List output files
    print("\nOutput files ready for download:")
    for file in os.listdir('/kaggle/output'):
        if file.endswith('.zip'):
            size = os.path.getsize(f'/kaggle/output/{file}') / 1e6
            print(f"  - {file} ({size:.2f} MB)")
else:
    print(f"Checkpoints not found at: {source_dir}")
```

---

## 🎓 Usage for Team Members

### Quick Start

1. **Copy this guide** to your team members
2. **Create new Kaggle Notebook**
3. **Follow Cell Setup** in order
4. **Run each cell** and verify output
5. **Download results** from Output section

### Troubleshooting

| Problem | Solution |
|---------|----------|
| GPU not available | Check Kaggle Settings → Accelerator |
| Dataset not found | Verify dataset name in Input files |
| Import errors | Ensure repository cloned successfully |
| Checkpoints not saving | Verify `/kaggle/working` directory exists |

---

## 📊 Performance Tips

| Setting | Value | Benefit |
|---------|-------|---------|
| `num_workers` | 2 | Good balance for Kaggle |
| `pin_memory` | True | Faster GPU transfer |
| `prefetch_factor` | 2 | Pre-load batches |
| `persistent_workers` | True | Keep workers alive |
| `batch_size` | 32 | Optimal for GPU memory |

---

## 📂 File Structure

After setup, Kaggle will have:

```
/kaggle/working/
├── KHOTAA/
│   ├── models/
│   │   ├── classification/
│   │   │   ├── efficientnetv2s.ipynb
│   │   │   ├── densenet.ipynb
│   │   │   └── ...
│   │   └── utils/
│   └── dataset/
├── checkpoints/  ← Training results
│   ├── efficientnetv2s_fold1/
│   ├── efficientnetv2s_fold2/
│   └── ...
└── training_results.zip  ← Downloaded file
```

---

## ✅ Final Checklist

- [ ] GPU enabled and verified
- [ ] Dataset added to Kaggle
- [ ] Project cloned successfully
- [ ] Paths configured correctly
- [ ] Training running without errors
- [ ] Results saved to `/kaggle/working/checkpoints`
- [ ] ZIP file created in Output section
- [ ] Downloaded results successfully

---

## 🔗 References

- **Kaggle Official Docs**: https://www.kaggle.com/docs
- **Dataset Source**: https://www.kaggle.com/datasets/khalidsiddiqui2003/dfu-dataset-annotated-into-4-classes
- **KHOTAA GitHub**: https://github.com/csstudentkaum/KHOTAA

---

**Last Updated:** February 2, 2026  
**For Team:** KHOTAA Project Team
