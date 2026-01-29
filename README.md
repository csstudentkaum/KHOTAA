# KHOTAA: Smart Diabetic Foot Shield 🦶🔬

Deep Learning-based classification system for Diabetic Foot Ulcer (DFU) detection and grading.

## 📋 Project Overview

KHOTAA (Smart Diabetic Foot Shield) is a machine learning project focused on classifying diabetic foot ulcers into 4 severity grades using state-of-the-art deep learning models.

### Dataset
- **Source:** [Kaggle DFU Dataset](https://www.kaggle.com/datasets/khalidsiddiqui2003/dfu-dataset-annotated-into-4-classes)
- **Total Images:** 10,062
- **Classes:** 4 (Grade 1, Grade 2, Grade 3, Grade 4)
- **Splits:** Train (9,639), Valid (282), Test (141)

## 🏗️ Project Structure

```
KHOTAA/
├── models/
│   ├── classification/          # Classification model notebooks
│   │   ├── dataset_loader.py    # ✅ Dataset loader (complete)
│   │   ├── resnet50.ipynb       # ResNet50 model
│   │   ├── densenet.ipynb       # DenseNet model
│   │   ├── googlenet.ipynb      # GoogLeNet model
│   │   ├── mobilenet.ipynb      # MobileNet model
│   │   ├── resnet101.ipynb      # ResNet101 model
│   │   ├── efficientnetv2s.ipynb # EfficientNetV2S model
│   │   └── pfcnn_drnn.ipynb     # Custom PFCNN+DRNN model
│   └── utils/                   # Utility notebooks
│       ├── model_manager.ipynb
│       ├── model_runner.ipynb
│       └── training_utils.ipynb
├── dataset/                     # Dataset folder (not in repo)
├── test_dataset_loader.py       # Dataset verification script
├── requirements.txt             # Python dependencies
├── DATASET_SETUP.md            # Dataset information
└── README.md                   # This file
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/KHOTAA.git
cd KHOTAA
```

### 2. Set Up Environment

**Option A: Using pip**
```bash
pip install -r requirements.txt
```

**Option B: Using virtual environment (Recommended)**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the Dataset

1. Download from [Kaggle](https://www.kaggle.com/datasets/khalidsiddiqui2003/dfu-dataset-annotated-into-4-classes)
2. Extract to `KHOTAA/dataset/` folder
3. Verify structure:
   ```
   dataset/
   ├── train/
   │   ├── Grade 1/
   │   ├── Grade 2/
   │   ├── Grade 3/
   │   └── Grade 4/
   ├── valid/
   └── test/
   ```

### 4. Test the Dataset Loader

```bash
python3 test_dataset_loader.py
```

Expected output:
- ✅ Dataset found
- ✅ 10,062 images loaded
- ✅ 4 classes detected
- ✅ All splits working

### 5. Start Training

Open any model notebook and start experimenting:
```bash
jupyter notebook models/classification/resnet50.ipynb
```

## 🧪 Models

We're implementing 7 deep learning models for comparison:

| Model | Type | Status | Notebook |
|-------|------|--------|----------|
| ResNet50 | CNN | 🔄 In Progress | `resnet50.ipynb` |
| DenseNet | CNN | ⏳ Pending | `densenet.ipynb` |
| GoogLeNet | Inception | ⏳ Pending | `googlenet.ipynb` |
| MobileNet | Mobile | ⏳ Pending | `mobilenet.ipynb` |
| ResNet101 | Deep CNN | ⏳ Pending | `resnet101.ipynb` |
| EfficientNetV2S | Efficient | ⏳ Pending | `efficientnetv2s.ipynb` |
| PFCNN + DRNN | Custom | ⏳ Pending | `pfcnn_drnn.ipynb` |

## 👥 Team Collaboration

### For Team Members

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/KHOTAA.git
   cd KHOTAA
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset** (see section 3 above)

4. **Create a branch for your model:**
   ```bash
   git checkout -b feature/model-name
   # Example: git checkout -b feature/resnet50
   ```

5. **Work on your assigned model**

6. **Push your changes:**
   ```bash
   git add .
   git commit -m "Implemented ResNet50 model"
   git push origin feature/model-name
   ```

7. **Create a Pull Request** on GitHub

### Workflow

- Each team member works on **one model**
- Use **separate branches** for each model
- **Commit frequently** with clear messages
- Create **Pull Requests** for review before merging
- Use **Issues** for bug tracking and discussions

## 📊 Dataset Information

See [DATASET_SETUP.md](DATASET_SETUP.md) for detailed dataset information.

**Quick Stats:**
- Total: 10,062 images
- Grade 1: 2,370 images (23.6%)
- Grade 2: 2,460 images (24.5%)
- Grade 3: 2,802 images (27.8%)
- Grade 4: 2,430 images (24.1%)

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, torchvision
- **Data Processing:** NumPy, OpenCV, Pillow
- **Visualization:** Matplotlib, Seaborn
- **Metrics:** scikit-learn
- **Environment:** Jupyter Notebooks

## 📝 Requirements

See [requirements.txt](requirements.txt) for full dependencies.

Key packages:
- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `opencv-python >= 4.8.0`
- `numpy >= 1.24.0`
- `scikit-learn >= 1.3.0`

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is for academic/research purposes.

## 👨‍💻 Authors

- Your Team Name
- Team Members List

## 🙏 Acknowledgments

- Dataset: [Khalid Siddiqui - Kaggle](https://www.kaggle.com/datasets/khalidsiddiqui2003/dfu-dataset-annotated-into-4-classes)
- Inspiration: Medical AI research for diabetic foot ulcer detection

---

**Note:** The `dataset/` folder is not included in the repository. Each team member must download it separately from Kaggle.
