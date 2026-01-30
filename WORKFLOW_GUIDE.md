# KHOTAA Utilities Workflow Guide

Complete guide on how to use the utility modules in your model notebooks.

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

# Load splits
X_train, y_train = loader.load_split_paths('train', shuffle=True)
X_val, y_val = loader.load_split_paths('valid')
X_test, y_test = loader.load_split_paths('test')

# Create datasets
train_dataset = DFUDataset(X_train, y_train, transform=train_transform)
val_dataset = DFUDataset(X_val, y_val, transform=val_test_transform)
test_dataset = DFUDataset(X_test, y_test, transform=val_test_transform)

# Create dataloaders (batch_size=32 is default)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
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

### **Step 6: Train the Model**

```python
# Cell 6: Train Model (30 epochs is default)
# Train for 30 epochs
history = engine.train(
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=30,  # Default
    scheduler=scheduler,
    checkpoint_manager=checkpoint_manager
)

print("\nTraining completed!")
print(f"Best validation accuracy: {max(history['val_acc'])*100:.2f}%")
```

---

### **Step 7: Plot Training History**

```python
# Cell 7: Visualize Training
# Plot training curves
plot_training_history(history, save_path='results/resnet50_training_history.png')
```

---

### **Step 8: Evaluate on Test Set**

```python
# Cell 8: Test Evaluation
# Load best model
checkpoint_manager.load_best_model(model, metric_name='accuracy')

# Evaluate on test set
test_loss, test_acc, predictions, true_labels = engine.evaluate(test_loader, criterion)

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
    # Train one epoch
    train_loss, train_acc = engine.train_epoch(train_loader, criterion, optimizer)
    
    # Validate
    val_loss, val_acc, _, _ = engine.evaluate(val_loader, criterion)
    
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

---

## 🚀 Ready to Start!

1. Open `models/classification/resnet50.ipynb`
2. Follow the workflow above
3. Copy-paste cells and adapt as needed
4. Run and train your model!

Good luck with your training! 🎉
