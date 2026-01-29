# GitHub Setup Guide - KHOTAA Team Collaboration

This guide will help you upload the KHOTAA project to GitHub and collaborate with your team.

## 📤 Initial Upload (Project Owner)

### Step 1: Initialize Git Repository

```bash
cd /Users/manaralharbi/Desktop/KHOTAA

# Initialize git (if not already done)
git init

# Add all files (dataset will be excluded by .gitignore)
git add .

# Create first commit
git commit -m "Initial commit: KHOTAA project setup with 7 model notebooks"
```

### Step 2: Create GitHub Repository

1. Go to [GitHub.com](https://github.com)
2. Click the **"+"** icon → **"New repository"**
3. Fill in:
   - **Repository name:** `KHOTAA` (or `AI-project` or your choice)
   - **Description:** "Smart Diabetic Foot Shield - DFU Classification using Deep Learning"
   - **Visibility:** 
     - ✅ **Public** (if you want others to see it)
     - ✅ **Private** (for team-only access)
   - ❌ **DO NOT** initialize with README (we already have one)
4. Click **"Create repository"**

### Step 3: Connect Local Repo to GitHub

GitHub will show you commands. Use these:

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/KHOTAA.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Replace `YOUR_USERNAME` with your actual GitHub username!**

### Step 4: Verify Upload

1. Refresh your GitHub repository page
2. You should see all files **except** `dataset/` folder
3. ✅ README.md should be displayed on the main page

---

## 👥 Adding Team Members

### Option A: Add Collaborators (Private Repo)

1. Go to your repository on GitHub
2. Click **Settings** → **Collaborators**
3. Click **"Add people"**
4. Enter each team member's GitHub username
5. They'll receive an invitation email

### Option B: Share Public Repo Link

If the repo is public, just share the link:
```
https://github.com/YOUR_USERNAME/KHOTAA
```

---

## 📥 Team Members - Getting Started

### Step 1: Clone the Repository

```bash
# Navigate to where you want the project
cd ~/Desktop  # or your preferred location

# Clone the repository
git clone https://github.com/YOUR_USERNAME/KHOTAA.git

# Enter the project folder
cd KHOTAA
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all requirements
pip install -r requirements.txt
```

### Step 3: Download Dataset

⚠️ **Important:** The dataset is NOT in the repository (it's too large!)

Each team member must:
1. Download from [Kaggle](https://www.kaggle.com/datasets/khalidsiddiqui2003/dfu-dataset-annotated-into-4-classes)
2. Extract to `KHOTAA/dataset/` folder
3. Run test: `python3 test_dataset_loader.py`

### Step 4: Verify Setup

```bash
# Test dataset loader
python3 test_dataset_loader.py

# Should see: ✅ ALL TESTS PASSED!
```

---

## 🔄 Team Workflow

### Assigning Models

Divide the work among team members:

| Team Member | Model | Notebook |
|------------|-------|----------|
| Member 1 | ResNet50 | `resnet50.ipynb` |
| Member 2 | DenseNet | `densenet.ipynb` |
| Member 3 | GoogLeNet | `googlenet.ipynb` |
| Member 4 | MobileNet | `mobilenet.ipynb` |
| Member 5 | ResNet101 | `resnet101.ipynb` |
| Member 6 | EfficientNetV2S | `efficientnetv2s.ipynb` |
| Member 7 | PFCNN+DRNN | `pfcnn_drnn.ipynb` |

### Working on Your Model

```bash
# 1. Make sure you're on main branch and up to date
git checkout main
git pull origin main

# 2. Create a new branch for your model
git checkout -b feature/resnet50
# or: feature/densenet, feature/googlenet, etc.

# 3. Work on your notebook
# Open: models/classification/resnet50.ipynb
# Implement your model

# 4. Commit your changes regularly
git add models/classification/resnet50.ipynb
git commit -m "Implemented ResNet50 data loading and preprocessing"

# 5. Continue working and committing
git add models/classification/resnet50.ipynb
git commit -m "Added ResNet50 model architecture"

git add models/classification/resnet50.ipynb
git commit -m "Completed ResNet50 training and evaluation"

# 6. Push your branch to GitHub
git push origin feature/resnet50

# 7. Create Pull Request on GitHub
# Go to repository → "Pull requests" → "New pull request"
# Select your branch → Create PR → Request review from team
```

### Syncing with Team Changes

```bash
# Update your local main branch
git checkout main
git pull origin main

# Update your feature branch with latest main
git checkout feature/your-model
git merge main

# If there are conflicts, resolve them, then:
git add .
git commit -m "Merged main branch updates"
```

---

## 📝 Git Commands Cheat Sheet

### Daily Workflow

```bash
# Check current status
git status

# See which branch you're on
git branch

# Add files
git add filename.ipynb
git add .  # Add all changes

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push origin branch-name

# Pull latest changes
git pull origin main

# Switch branches
git checkout branch-name

# Create new branch
git checkout -b feature/new-feature
```

### Checking Changes

```bash
# See what changed in files
git diff

# See commit history
git log --oneline

# See remote repositories
git remote -v
```

---

## 🚨 Important Notes

### DO NOT Upload

The `.gitignore` file prevents these from being uploaded:
- ✅ `dataset/` folder (too large, ~2GB+)
- ✅ `.venv/` virtual environment
- ✅ `__pycache__/` Python cache
- ✅ `.ipynb_checkpoints/` Jupyter checkpoints
- ✅ `*.pth` trained model files (can be large)

### DO Upload

These should be in the repository:
- ✅ All `.ipynb` notebook files
- ✅ `dataset_loader.py`
- ✅ `requirements.txt`
- ✅ `README.md` and documentation
- ✅ `.gitignore`

---

## 🆘 Troubleshooting

### "Permission denied"
Make sure you're added as a collaborator or the repo is public.

### "Dataset not found"
Each team member must download the dataset separately (not in git repo).

### "Merge conflicts"
1. Don't work on the same files simultaneously
2. Each person works on their own model notebook
3. If conflicts occur, communicate with your team

### "Large files" error
If you accidentally try to upload dataset:
```bash
git rm -r --cached dataset/
git commit -m "Remove dataset from git"
git push
```

---

## 📧 Communication

Use GitHub features:
- **Issues:** Report bugs, ask questions, track tasks
- **Pull Requests:** Code review, discuss implementations
- **Discussions:** General team communication
- **Projects:** Track progress (optional)

---

## ✅ Checklist for Project Owner

Before uploading to GitHub:

- [ ] `.gitignore` configured (excludes dataset)
- [ ] `README.md` updated with team info
- [ ] `requirements.txt` complete
- [ ] Remove any sensitive information
- [ ] Test that `git status` doesn't show dataset files
- [ ] Create repository on GitHub
- [ ] Push initial commit
- [ ] Add team members as collaborators
- [ ] Share repository URL with team

---

## ✅ Checklist for Team Members

After cloning:

- [ ] Repository cloned successfully
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded from Kaggle
- [ ] Dataset extracted to `KHOTAA/dataset/`
- [ ] Dataset loader tested (`python3 test_dataset_loader.py`)
- [ ] Branch created for your model
- [ ] Ready to start implementing!

---

**Happy Coding! 🚀**
