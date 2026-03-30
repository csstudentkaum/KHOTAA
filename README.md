# KHOTAA — Smart Diabetic Foot Shield

Deep learning classification system for **Diabetic Foot Ulcer (DFU)** images, classifying wound pathology into 4 classes: **Both**, **Infection**, **Ischaemia**, and **None**.

---

## Dataset

- **Source:** [Roboflow — DFU Dataset](https://universe.roboflow.com/diabetic-c6n36/dfu-kew1f) (CC BY 4.0)
- **Total Images:** 4,446
- **Classes:** 4 — Both, Infection, Ischaemia, None
- **Splits:** Train (3,112) / Valid (889) / Test (445)
- **Imbalance Ratio:** 11.3× (Ischaemia: 152 vs None: 1,725)
- **CV Setup:** Train + Valid combined (4,001 images) for 5-fold stratified cross-validation. Test set (445 images) held out entirely.

> **Note:** Because of the heavy class imbalance, **Macro F1** is used as the primary evaluation metric (not accuracy).

---

## Pipeline Overview

```
1. Data Preprocessing & Augmentation
2. 5-Fold Stratified Cross-Validation (6 models)
3. Model Comparison → Select best model (EfficientNetV2-S)
4. Head Architecture Search (21 configs)
5. Final Training (train+valid combined, 30 epochs)
6. Test Evaluation (reserved test set)
```

---

## Preprocessing & Augmentation

All models use the same preprocessing pipeline defined in `dataset_preprocessing.py`, ensuring fair comparison.

**Training transforms (with augmentation):**

| Transform | Value |
|-----------|-------|
| Resize | 224 × 224 |
| Random Horizontal Flip | p = 0.5 |
| Random Vertical Flip | p = 0.5 |
| Random Rotation | ± 20° |
| Random Affine (zoom) | scale ± 20% |
| Color Jitter — Brightness | ± 10% |
| Color Jitter — Contrast | ± 10% |
| Normalize (ImageNet) | mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) |

**Validation / Test transforms (no augmentation):**

| Transform | Value |
|-----------|-------|
| Resize | 224 × 224 |
| Normalize (ImageNet) | mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) |

> Augmentation is applied on-the-fly during training only. Validation and test images are only resized and normalized.

---

## Model Comparison Results

6 architectures evaluated via **5-fold stratified CV**, ranked by **Macro F1** (primary metric for imbalanced data):

| Rank | Model | CV Mean Acc | Macro F1 | MCC | Macro AUC |
|:----:|-------|:-----------:|:--------:|:---:|:---------:|
| **1** | **EfficientNetV2-S** | **80.81%** | **91.00%** | **0.8424** | **0.9848** |
| 2 | MobileNetV2 | 75.38% | 86.87% | 0.7653 | 0.9669 |
| 3 | DenseNet121 | 74.16% | 84.78% | 0.6983 | 0.9532 |
| 4 | ResNet50 | 70.71% | 78.42% | 0.6394 | 0.9309 |
| 5 | ResNet101 | 75.63% | 78.34% | 0.6270 | 0.9375 |
| 6 | GoogLeNet | 78.96% | 40.85% | 0.2726 | 0.7613 |

---

## Head Architecture Search

After selecting EfficientNetV2-S, a **head architecture search** was conducted — 21 random configurations screened on fold 1, top 3 validated across all 5 folds.

**Winner — Trial 4:**

| Parameter | Value |
|-----------|-------|
| Hidden Layers | 1 |
| Hidden Sizes | [128] |
| Dropout | 0.5 |
| Activation | GELU |
| BatchNorm | No |
| **5-Fold Mean Acc** | **83.43% ± 2.90%** |
| **5-Fold Mean F1** | **84.31% ± 1.82%** |

Architecture: `1280 → Linear(128) → GELU → Dropout(0.5) → Linear(4)`

---

## Final Model — Test Set Performance

EfficientNetV2-S + winning head, trained on **all non-test data** (train + valid combined) for **30 fixed epochs**:

| Metric | Score |
|--------|:-----:|
| **Accuracy** | **84.04%** |
| **Macro F1** | **82.93%** |
| **Macro Precision** | **81.64%** |
| **Macro Recall** | **86.53%** |
| **Macro Specificity** | **93.41%** |
| **MCC** | **0.7434** |
| **AUC-ROC** | **0.9591** |

---

## Project Structure

```
KHOTAA/
├── models/
│   ├── classification/
│   │   ├── dataset_loader.py               # Dataset loader utility
│   │   ├── dataset_preprocessing.py        # Augmentation & transforms
│   │   ├── resnet50.ipynb                  # ResNet50 — 5-fold CV
│   │   ├── resnet101.ipynb                 # ResNet101 — 5-fold CV
│   │   ├── densenet121.ipynb               # DenseNet121 — 5-fold CV
│   │   ├── googlenet.ipynb                 # GoogLeNet — 5-fold CV
│   │   ├── mobilenetv2.ipynb               # MobileNetV2 — 5-fold CV
│   │   ├── efficientnetv2s.ipynb           # EfficientNetV2-S — 5-fold CV
│   │   ├── efficientnetv2s_head_search.ipynb   # Head architecture search
│   │   ├── efficientnetv2s-final-training.ipynb # Final training & test eval
│   │   ├── model_comparison.ipynb          # 6-model comparison notebook
│   │   ├── results/                        # All metrics JSONs & figures
│   │   │   ├── *_comprehensive_metrics.json    # Per-model CV + val metrics
│   │   │   ├── head_search_results.json        # Head search results
│   │   │   ├── efficientnetv2s_final_results/  # Final model + test results
│   │   │   └── comparison/                     # Comparison figures & tables
│   │   └── checkpoints/                    # Model checkpoints (per fold)
│   └── utils/
│       ├── training_engine.py              # TrainingEngine, optimizers, schedulers
│       ├── metrics_evaluator.py            # Metrics, plots, confusion matrix, ROC
│       ├── checkpoint_manager.py           # Checkpoint save/load
│       └── model_comparison.py             # Comparison utilities
├── DFU_dataset/                            # DFU dataset (Roboflow, not in repo)
├── requirements.txt                        # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Clone & Setup

```bash
git clone https://github.com/csstudentkaum/KHOTAA.git
cd KHOTAA
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

### 2. Download Dataset

The dataset is downloaded automatically via Roboflow API in each notebook. Alternatively, download manually from [Roboflow](https://universe.roboflow.com/diabetic-c6n36/dfu-kew1f) and place it in `DFU_dataset/`.

### 3. Run on Kaggle

All training notebooks are designed to run on **Kaggle GPU**. Each notebook includes a setup cell that clones the repo and installs dependencies automatically.

---

## Training Configuration

Validated via cross-validation and used consistently across all experiments:

| Component | Setting |
|-----------|---------|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.3, patience=3, min_lr=1e-7) |
| Loss | CrossEntropyLoss (balanced class weights) |
| Batch Size | 32 |
| Input Size | 224 × 224 |
| Augmentation | H-flip, V-flip, rotation (±20°), zoom, brightness, contrast |

---

## Tech Stack

- **Deep Learning:** PyTorch, torchvision
- **Data Processing:** NumPy, OpenCV, Pillow
- **Visualization:** Matplotlib, Seaborn
- **Metrics:** scikit-learn
- **Dataset:** Roboflow
- **Environment:** Jupyter Notebooks, Kaggle GPU

---

## Requirements

See [requirements.txt](requirements.txt) for full dependencies.

Key packages:
- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `numpy >= 1.24.0`
- `scikit-learn >= 1.3.0`
- `matplotlib >= 3.7.0`
- `roboflow`

---

## Acknowledgments

- **Dataset:** [Roboflow — DFU Dataset](https://universe.roboflow.com/diabetic-c6n36/dfu-kew1f) (CC BY 4.0)
- Medical AI research for diabetic foot ulcer classification

---

**Note:** The `DFU_dataset/` folder is not included in the repository. It is downloaded automatically via Roboflow API in each notebook.
