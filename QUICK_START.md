# Quick Start - Upload to GitHub

## 🚀 For You (Project Owner)

### 1. Create GitHub Repository
- Go to: https://github.com/new
- Name: `KHOTAA`
- Description: `Smart Diabetic Foot Shield - DFU Classification`
- Visibility: Choose Public or Private
- **DON'T** check "Initialize with README"
- Click "Create repository"

### 2. Upload Project

```bash
cd /Users/manaralharbi/Desktop/KHOTAA

git add .
git commit -m "Initial commit: KHOTAA project with 7 DL models"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/KHOTAA.git
git push -u origin main
```

**⚠️ Replace `YOUR_USERNAME` with your GitHub username!**

### 3. Add Team Members
- Go to your repository on GitHub
- Settings → Collaborators and teams
- Click "Add people"
- Enter each team member's GitHub username

---

## 👥 For Team Members

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/KHOTAA.git
cd KHOTAA
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset
- Download from Kaggle: https://www.kaggle.com/datasets/khalidsiddiqui2003/dfu-dataset-annotated-into-4-classes
- Extract to `KHOTAA/dataset/`
- Test: `python3 test_dataset_loader.py`

### 4. Start Working

```bash
# Create your branch
git checkout -b feature/resnet50

# Work on your notebook
# models/classification/resnet50.ipynb

# Commit changes
git add .
git commit -m "Implemented ResNet50 model"

# Push to GitHub
git push origin feature/resnet50

# Create Pull Request on GitHub
```

---

## 📋 Model Assignments

| Member | Model | Branch Name |
|--------|-------|-------------|
| Member 1 | ResNet50 | `feature/resnet50` |
| Member 2 | DenseNet | `feature/densenet` |
| Member 3 | GoogLeNet | `feature/googlenet` |
| Member 4 | MobileNet | `feature/mobilenet` |
| Member 5 | ResNet101 | `feature/resnet101` |
| Member 6 | EfficientNetV2S | `feature/efficientnetv2s` |
| Member 7 | PFCNN+DRNN | `feature/pfcnn-drnn` |

---

## ⚠️ Important

- ✅ Dataset is NOT uploaded to GitHub (too large)
- ✅ Each team member downloads dataset separately
- ✅ Work on separate branches
- ✅ Create Pull Requests for review
- ✅ Read GITHUB_SETUP.md for full instructions

---

**Need Help?** Read GITHUB_SETUP.md for detailed instructions!
