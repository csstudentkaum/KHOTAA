# Dataset Information - KHOTAA

## ✅ Dataset Status: LOADED & VERIFIED

The Diabetic Foot Ulcer (DFU) dataset is already set up and working!

### 📊 Dataset Statistics

**Source:** [Kaggle DFU Dataset](https://www.kaggle.com/datasets/khalidsiddiqui2003/dfu-dataset-annotated-into-4-classes)

**Total Images:** 10,062 images
- **Train:** 9,639 images
- **Valid:** 282 images  
- **Test:** 141 images

**Classes:** 4 diabetic foot ulcer grades
- **Grade 1:** 2,370 images total (2,240 train, 86 valid, 44 test)
- **Grade 2:** 2,460 images total (2,345 train, 78 valid, 37 test)
- **Grade 3:** 2,802 images total (2,744 train, 40 valid, 18 test)
- **Grade 4:** 2,430 images total (2,310 train, 78 valid, 42 test)

**Image Size:** 224x224x3 (RGB)

### 📂 Dataset Structure

```
KHOTAA/
├── dataset/
│   ├── train/
│   │   ├── Grade 1/  (2,240 images)
│   │   ├── Grade 2/  (2,345 images)
│   │   ├── Grade 3/  (2,744 images)
│   │   └── Grade 4/  (2,310 images)
│   ├── valid/
│   │   ├── Grade 1/  (86 images)
│   │   ├── Grade 2/  (78 images)
│   │   ├── Grade 3/  (40 images)
│   │   └── Grade 4/  (78 images)
│   └── test/
│       ├── Grade 1/  (44 images)
│       ├── Grade 2/  (37 images)
│       ├── Grade 3/  (18 images)
│       └── Grade 4/  (42 images)
```

## ✅ Verification Test Results

The dataset has been tested and verified:
- ✅ Dataset found at: `./dataset`
- ✅ 4 classes detected: Grade 1, Grade 2, Grade 3, Grade 4
- ✅ All splits loaded successfully (train/valid/test)
- ✅ Images load correctly (224x224 RGB)
- ✅ Balanced class distribution

### Re-run Test (Optional)
```bash
python3 test_dataset_loader.py
```

## 🧪 Using the Dataset in Notebooks

## 🧪 Using the Dataset in Notebooks

The dataset loader is ready to use in your model notebooks:

```python
from dataset_loader import SplitFolderDatasetLoader

# Initialize loader
loader = SplitFolderDatasetLoader(root_dir="../../dataset")

# Get class information
classes = loader.get_classes()
# Output: ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']

num_classes = loader.get_num_classes()
# Output: 4

# Load train data paths
X_train, y_train = loader.load_split_paths("train", shuffle=True)
print(f"Train images: {len(X_train)}")  # 9,639 images

# Load validation data
X_valid, y_valid = loader.load_split_paths("valid")
print(f"Valid images: {len(X_valid)}")  # 282 images

# Load test data
X_test, y_test = loader.load_split_paths("test")
print(f"Test images: {len(X_test)}")    # 141 images

# Check class distribution
print(loader.get_class_counts("train"))
# Output: {'Grade 1': 2240, 'Grade 2': 2345, 'Grade 3': 2744, 'Grade 4': 2310}
```

## 📈 Class Distribution Analysis

The dataset is well-balanced across all grades:

| Grade | Train | Valid | Test | Total | Percentage |
|-------|-------|-------|------|-------|------------|
| Grade 1 | 2,240 | 86 | 44 | 2,370 | 23.6% |
| Grade 2 | 2,345 | 78 | 37 | 2,460 | 24.5% |
| Grade 3 | 2,744 | 40 | 18 | 2,802 | 27.8% |
| Grade 4 | 2,310 | 78 | 42 | 2,430 | 24.1% |
| **Total** | **9,639** | **282** | **141** | **10,062** | **100%** |

**Split Ratios:**
- Training: 95.8%
- Validation: 2.8%
- Test: 1.4%

## 🎯 Next Steps

Now that your dataset is loaded and verified:

1. ✅ **Dataset ready** - All 10,062 images loaded
2. ✅ **Loader tested** - Working correctly
3. 🚀 **Start implementing** - Open `models/classification/resnet50.ipynb`
4. 🧪 **Begin experiments** - Train your first model!

## 📝 Notes

- All images are preprocessed to 224x224 pixels
- Images are in RGB format
- Class labels are folder-based (Grade 1, Grade 2, Grade 3, Grade 4)
- Dataset source: Kaggle DFU dataset (4 classes)
