# KHOTAA Utilities Workflow Guide

Complete guide on how to use the utility modules in your model notebooks.

**Default Training Method: 5-Fold Cross-Validation** for robust and reliable results.

---

## 📁 File Structure

```
KHOTAA/
├── models/
│   ├── classification/
│   │   ├── dataset_loader.py          # Load dataset
│   │   ├── resnet50.ipynb            # Your model notebooks
│   │   └── ...
│   └── utils/
│       ├── checkpoint_manager.py      # Save/load models
│       ├── training_engine.py         # Training loops
│       └── metrics_evaluator.py       # Metrics & plots
└── dataset/                           # Your DFU dataset
```

---

## 🔄 Complete Training Workflow

### **Step 1: Import All Utilities in Your Notebook**

```python
# Cell 1: Imports
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Add paths for imports
sys.path.append('../')  # For utils
sys.path.append('./')   # For dataset_loader

# Import custom utilities
from dataset_loader import SplitFolderDatasetLoader
from utils.checkpoint_manager import CheckpointManager
from utils.training_engine import TrainingEngine
from utils.metrics_evaluator import (
    calculate_metrics, print_metrics, plot_confusion_matrix,
    plot_roc_curve, plot_training_history, MetricsTracker
)

print("All modules imported successfully!")
```

---

### **Step 2: Load Dataset**

```python
# Cell 2: Load Dataset
# Initialize dataset loader
loader = SplitFolderDatasetLoader(root_dir='../../dataset')

# Get class information
classes = loader.get_classes()
num_classes = loader.get_num_classes()

print(f"Classes: {classes}")
print(f"Number of classes: {num_classes}")

# Check class distribution
print("\nClass Distribution:")
print(loader.get_class_counts('train'))
```

---

### **Step 3: Create PyTorch Dataset & DataLoaders**

```python
# Cell 3: Create Custom Dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold

class DFUDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load data for cross-validation
# Combine train + val for 5-fold cross-validation
X_train, y_train = loader.load_split_paths('train', shuffle=True)
X_val, y_val = loader.load_split_paths('valid')

# Combine train and validation for cross-validation
X_all = X_train + X_val
y_all = np.concatenate([y_train, y_val])

# Load test set separately (final evaluation only)
X_test, y_test = loader.load_split_paths('test')
test_dataset = DFUDataset(X_test, y_test, transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize 5-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Total samples for cross-validation: {len(X_all)}")
print(f"Test samples (final evaluation): {len(X_test)}")
print(f"Number of classes: {num_classes}")  # This is 4 for DFU dataset
print(f"Cross-validation folds: 5")
```

---

### **Step 4: Define Model**

```python
# Cell 4: Create Model (Example: ResNet50)
# Load pretrained ResNet50
model = models.resnet50(pretrained=True)

# Modify final layer for 4 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Model: ResNet50")
print(f"Device: {device}")
print(f"Output classes: {num_classes}")
```

---

### **Step 5: Setup Training Components**

```python
# Cell 5: Setup Training
# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer (momentum=0.8 as per paper)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Initialize checkpoint manager
checkpoint_manager = CheckpointManager(checkpoint_dir='checkpoints/resnet50')

# Initialize training engine
engine = TrainingEngine(model=model, device=device)

print("Training components initialized!")
```

---

### **Step 6: Train Model with 5-Fold Cross-Validation**

```python
# Cell 6: 5-Fold Cross-Validation Training
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all, y_all), 1):
    print(f"\n{'='*60}")
    print(f"FOLD {fold}/5")
    print(f"{'='*60}")
    
    # Split data for this fold
    X_train_fold = [X_all[i] for i in train_idx]
    y_train_fold = y_all[train_idx]
    X_val_fold = [X_all[i] for i in val_idx]
    y_val_fold = y_all[val_idx]
    
    # Create datasets for this fold
    train_dataset_fold = DFUDataset(X_train_fold, y_train_fold, transform=train_transform)
    val_dataset_fold = DFUDataset(X_val_fold, y_val_fold, transform=val_test_transform)
    
    # Create loaders for this fold
    # batch_size=32: Number of images per batch (default as per paper)
    # num_workers=4: Number of CPU processes for parallel data loading
    # Note: num_workers is NOT the number of classes! It's for performance optimization.
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=32, shuffle=True, num_workers=4)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize new model for this fold
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # New optimizer and scheduler for this fold
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Checkpoint manager for this fold
    checkpoint_manager = CheckpointManager(checkpoint_dir=f'checkpoints/resnet50_fold{fold}')
    
    # Training engine for this fold
    engine = TrainingEngine(model=model, device=device)
    
    # Train this fold for 30 epochs with verbose output
    history = engine.train(
        train_loader=train_loader_fold,
        val_loader=val_loader_fold,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=30,  # Default
        scheduler=scheduler,
        checkpoint_manager=checkpoint_manager,
        verbose=True  # Show detailed progress for each epoch
    )
    
    # Store fold results
    best_val_acc = max(history['val_acc'])
    fold_results.append({
        'fold': fold,
        'best_val_acc': best_val_acc,
        'final_val_acc': history['val_acc'][-1],
        'history': history,
        'model_path': f'checkpoints/resnet50_fold{fold}/best_model.pth'
    })
    
    print(f"\nFold {fold} Best Validation Accuracy: {best_val_acc*100:.2f}%")

# Calculate cross-validation statistics
avg_acc = np.mean([r['best_val_acc'] for r in fold_results])
std_acc = np.std([r['best_val_acc'] for r in fold_results])

print(f"\n{'='*60}")
print(f"5-FOLD CROSS-VALIDATION RESULTS")
print(f"{'='*60}")
print(f"Average Accuracy: {avg_acc*100:.2f}% ± {std_acc*100:.2f}%")
print(f"\nIndividual Fold Results:")
for r in fold_results:
    print(f"  Fold {r['fold']}: {r['best_val_acc']*100:.2f}%")

print("\n✓ Cross-validation training completed!")
```

---

### **Step 7: Plot Training History**

```python
# Cell 7: Visualize Training
# Plot training curves
plot_training_history(history, save_path='results/resnet50_training_history.png')
```

---

### **Step 8: Evaluate on Test Set (Best Fold Model)**

```python
# Cell 8: Test Evaluation with Best Fold
# Find the best fold based on validation accuracy
best_fold_idx = np.argmax([r['best_val_acc'] for r in fold_results])
best_fold_result = fold_results[best_fold_idx]
best_fold_num = best_fold_result['fold']

print(f"Best fold: Fold {best_fold_num} with validation accuracy: {best_fold_result['best_val_acc']*100:.2f}%")

# Load the best model from the best fold
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Load best model from best fold
checkpoint_manager = CheckpointManager(checkpoint_dir=f'checkpoints/resnet50_fold{best_fold_num}')
checkpoint_manager.load_best_model(model, metric_name='accuracy')

# Create training engine for evaluation
engine = TrainingEngine(model=model, device=device)

# Evaluate on test set with verbose output
test_loss, test_acc, predictions, true_labels = engine.evaluate(
    test_loader, 
    criterion, 
    verbose=True  # Show detailed progress during evaluation
)

print(f"\nFinal Test Set Evaluation (Best Fold {best_fold_num}):")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
```

---

### **Step 9: Calculate All Metrics**

```python
# Cell 9: Calculate Comprehensive Metrics
# Get prediction probabilities for AUC
model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

import numpy as np
y_pred_proba = np.vstack(all_probs)
y_true = np.concatenate(all_labels)

# Calculate all research-standard metrics
metrics = calculate_metrics(
    y_true=true_labels,
    y_pred=predictions,
    y_pred_proba=y_pred_proba,
    class_names=classes,
    average='macro'
)

# Print comprehensive metrics
print_metrics(metrics, title="ResNet50 Test Set Evaluation")

# Metrics include:
# - Accuracy
# - Precision (PPV): TP / (TP + FP)
# - Recall/Sensitivity (TPR): TP / (TP + FN)
# - Specificity (TNR): TN / (TN + FP)
# - F1-Score: 2PR / (P + R)
# - MCC: Matthews Correlation Coefficient
# - AUC-ROC: Area Under Curve
```

---

### **Step 10: Visualize Results**

```python
# Cell 10: Plot Confusion Matrix
plot_confusion_matrix(
    y_true=true_labels,
    y_pred=predictions,
    class_names=classes,
    normalize=True,
    save_path='results/resnet50_confusion_matrix.png'
)
```

```python
# Cell 11: Plot ROC Curve
plot_roc_curve(
    y_true=true_labels,
    y_pred_proba=y_pred_proba,
    class_names=classes,
    save_path='results/resnet50_roc_curve.png'
)
```

---

## 🔧 Alternative Workflow: Manual Training Loop

If you want more control, use utilities piece by piece:

```python
# Manual training with MetricsTracker
tracker = MetricsTracker()

for epoch in range(30):
    # Train one epoch with verbose output
    train_loss, train_acc = engine.train_epoch(
        train_loader, 
        criterion, 
        optimizer, 
        verbose=True  # Show progress bar during training
    )
    
    # Validate with verbose output
    val_loss, val_acc, _, _ = engine.evaluate(
        val_loader, 
        criterion, 
        verbose=True  # Show progress bar during validation
    )
    
    # Update scheduler
    scheduler.step()
    
    # Track metrics
    tracker.update(
        train_loss=train_loss,
        train_acc=train_acc,
        val_loss=val_loss,
        val_acc=val_acc,
        epoch=epoch+1
    )
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch+1,
            metrics={'val_acc': val_acc, 'val_loss': val_loss}
        )
    
    print(f"Epoch {epoch+1}/30 - Train Loss: {train_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

# Print summary
tracker.summary()

# Plot metrics
tracker.plot_metrics(save_path='results/training_metrics.png')
```

---

## 🔄 Optional: Simple Train/Val/Test Split (No Cross-Validation)

**Note:** The default workflow uses 5-fold cross-validation for robust evaluation.
Use this simpler approach only if you want faster training without cross-validation.

```python
# Alternative: Simple Split (No Cross-Validation)
# Load splits separately
X_train, y_train = loader.load_split_paths('train', shuffle=True)
X_val, y_val = loader.load_split_paths('valid')
X_test, y_test = loader.load_split_paths('test')

# Create datasets
train_dataset = DFUDataset(X_train, y_train, transform=train_transform)
val_dataset = DFUDataset(X_val, y_val, transform=val_test_transform)
test_dataset = DFUDataset(X_test, y_test, transform=val_test_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Setup training
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
checkpoint_manager = CheckpointManager(checkpoint_dir='checkpoints/resnet50')
engine = TrainingEngine(model=model, device=device)

# Train for 30 epochs (single model, no cross-validation)
history = engine.train(
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=30,
    scheduler=scheduler,
    checkpoint_manager=checkpoint_manager,
    verbose=True  # Show detailed progress for each epoch
)

print(f"Best validation accuracy: {max(history['val_acc'])*100:.2f}%")

# Evaluate on test set
checkpoint_manager.load_best_model(model, metric_name='accuracy')
test_loss, test_acc, predictions, true_labels = engine.evaluate(test_loader, criterion)
print(f"Test Accuracy: {test_acc*100:.2f}%")
```

---

## 💾 Save/Load Model Examples

### **Save Model**

```python
# Option 1: Save best model (automatic during training)
# Already handled by TrainingEngine if checkpoint_manager is provided

# Option 2: Save manually
checkpoint_manager.save_best_model(
    model=model,
    metrics={'accuracy': test_acc, 'loss': test_loss}
)

# Option 3: Quick save (just model weights)
from utils.checkpoint_manager import save_model
save_model(model, 'final_resnet50.pth')
```

### **Load Model**

```python
# Option 1: Load best model
checkpoint_manager.load_best_model(model, metric_name='accuracy')

# Option 2: Load specific checkpoint
checkpoint = checkpoint_manager.load_checkpoint(model, optimizer, filename='checkpoint_epoch_20.pth')

# Option 3: Quick load
from utils.checkpoint_manager import load_model
load_model(model, 'final_resnet50.pth')
```

---

## 📊 Export Metrics to File

```python
# Save metrics as JSON
import json

with open('results/resnet50_metrics.json', 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in metrics.items()}
    json.dump(metrics_json, f, indent=4)

print("Metrics saved to results/resnet50_metrics.json")
```

---

## 🎯 Quick Reference Summary

| **Task** | **Module** | **Function/Class** |
|----------|------------|-------------------|
| Load dataset | `dataset_loader` | `SplitFolderDatasetLoader` |
| Train model | `training_engine` | `TrainingEngine.train()` |
| Save/load checkpoints | `checkpoint_manager` | `CheckpointManager` |
| Calculate metrics | `metrics_evaluator` | `calculate_metrics()` |
| Print metrics | `metrics_evaluator` | `print_metrics()` |
| Plot confusion matrix | `metrics_evaluator` | `plot_confusion_matrix()` |
| Plot ROC curve | `metrics_evaluator` | `plot_roc_curve()` |
| Track training | `metrics_evaluator` | `MetricsTracker` |

---

## 📝 Tips

1. **Always check GPU availability** before training
2. **Save checkpoints regularly** to avoid losing progress
3. **Use MetricsTracker** to monitor training in real-time
4. **Calculate all metrics** for comprehensive evaluation
5. **Save plots** for your research paper/presentation
6. **Use batch_size=32** (default) as per the paper
7. **Train for 30 epochs** (default) as per the paper
8. **Set momentum=0.8** for optimizer as per the paper
9. **Adjust num_workers** based on your CPU cores (4-8 is typical, 0 if issues occur)
10. **Number of classes = 4** (Grade 1, 2, 3, 4) - Don't confuse with num_workers!
11. **Default is 5-fold cross-validation** - More robust evaluation, better for research
12. **Use simple split if needed** - For faster training without CV (see optional simple split section)
13. **Best fold model is used for test** - Automatically selects best performing fold for final evaluation
14. **verbose=True is set by default** - Shows detailed progress bars and epoch information during training/evaluation

---

## 🚀 Ready to Start!

1. Open `models/classification/resnet50.ipynb`
2. Follow the workflow above
3. Copy-paste cells and adapt as needed
4. Run and train your model!

Good luck with your training! 🎉
