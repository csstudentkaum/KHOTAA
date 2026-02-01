# Google Colab Setup Guide for ResNet101 Training

## 📋 Prerequisites

1. **Google Account** with Google Drive access
2. **Dataset folder:** `dfu-dataset-annotated-into-4-classes` on your local machine
3. **This notebook:** `resnet101.ipynb`

---

## 🚀 Step-by-Step Setup

### Step 1: Upload Dataset to Google Drive

1. Go to [drive.google.com](https://drive.google.com)
2. Click **New** → **Folder upload**
3. Select your `dfu-dataset-annotated-into-4-classes` folder
4. Wait for upload to complete
5. Note the location (e.g., `MyDrive/dfu-dataset-annotated-into-4-classes`)

**Folder Structure Should Be:**
```
dfu-dataset-annotated-into-4-classes/
├── train/
│   ├── Grade 1/
│   ├── Grade 2/
│   ├── Grade 3/
│   └── Grade 4/
├── valid/
│   └── (same 4 grades)
└── test/
    └── (same 4 grades)
```

---

### Step 2: Open Notebook in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File** → **Upload notebook**
3. Select `resnet101.ipynb` from your local `KHOTAA/models/classification/` folder
4. Notebook will open in Colab

---

### Step 3: Configure GPU Runtime

1. In Colab, click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Click **Save**
4. The page will refresh

---

### Step 4: Run the Notebook

#### Cell 2: GPU Verification
```python
# Verify T4 GPU is available
# Should show: GPU Name: Tesla T4, GPU Memory: ~15 GB
```

#### Cell 7: Clone Repository
```python
# Clones KHOTAA repo from GitHub
# Navigates to models/classification
```

#### Cell 8: Import Modules
```python
# Imports all required libraries
# Loads custom utilities
```

#### Cell 11: Mount Google Drive & Set Dataset Path
```python
# IMPORTANT: You'll need to authorize Google Drive access
# Click the link, sign in, copy auth code, paste it back
```

**⚠️ UPDATE THIS LINE in Cell 11:**
```python
dataset_path = '/content/drive/MyDrive/dfu-dataset-annotated-into-4-classes'
```

Change to match YOUR Google Drive location!

To find your path:
- Look in Files sidebar (📁 icon on left)
- Navigate: `drive` → `MyDrive` → [your folder]
- Copy the full path

#### Cell 12: Load Dataset
```python
# Loads images and creates data loaders
# Verifies 4 classes found
```

#### Cell 14: Model Definition
```python
# Creates ResNet101 model
# Shows model architecture
```

#### Cell 16: Training (5-Fold Cross-Validation)
```python
# LONGEST STEP: ~2-3 hours on T4 GPU
# Trains 5 folds with early stopping
# Saves checkpoints automatically
```

#### Cell 18: Evaluation
```python
# Tests best model
# Generates metrics and plots
```

#### Cell 20: Save Results
```python
# Saves results to JSON
# Ready for model comparison
```

---

## 💾 Download Results After Training

### Option 1: Download Files Individually
1. In Files sidebar (📁), navigate to `results/` folder
2. Right-click each file → Download:
   - `resnet101_results.json`
   - `resnet101_confusion_matrix.png`
   - `resnet101_roc_curve.png`
   - `resnet101_training_history.png`

### Option 2: Download Notebook
1. **File** → **Download** → **Download .ipynb**
2. Save to your local `KHOTAA/models/classification/` folder

---

## 🔄 Committing Changes to GitHub

After downloading the notebook:

```bash
cd ~/Desktop/KHOTAA

# Add the updated notebook
git add models/classification/resnet101.ipynb

# Optionally add results (if you downloaded them)
git add models/classification/results/resnet101_*

# Commit
git commit -m "Add ResNet101 training results from Colab"

# Push to GitHub
git push origin main
```

---

## 🐛 Troubleshooting

### "Dataset not found" error
- Check `dataset_path` in Cell 11
- Make sure dataset uploaded to Google Drive
- Verify folder structure matches expected format

### "Out of memory" error
- Reduce `batch_size` from 32 to 16 in Cells 12 and 16
- Runtime → Restart runtime → Run all cells again

### "utils module not found" error
- Make sure Cell 7 (clone repo) ran successfully
- Check output shows: `✓ utils/`

### Connection timeout
- Google Colab free tier has 12-hour limit
- If disconnected mid-training:
  - Checkpoints are saved automatically
  - Can resume from last checkpoint (advanced)

---

## 📊 Expected Results

After completion, you should have:
- ✅ Cross-validation accuracy: ~XX% ± XX%
- ✅ Test accuracy on held-out set
- ✅ Confusion matrix visualization
- ✅ ROC curves for all 4 classes
- ✅ Training history plots
- ✅ Complete metrics (Precision, Recall, F1, MCC, AUC)
- ✅ Inference time statistics

---

## ⏱️ Time Estimates

- Dataset upload to Drive: **10-30 minutes** (depending on size and internet speed)
- Training (5-fold CV): **2-3 hours** on T4 GPU
- Total process: **~3-4 hours**

---

## 📝 Notes

- Free Colab GPU has usage limits (~12 hours/session)
- Save your work frequently
- Download results before closing Colab
- Checkpoints saved in `/content/KHOTAA/models/classification/checkpoints/`

---

**Good luck with your training! 🚀**
