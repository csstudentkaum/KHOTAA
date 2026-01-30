# KHOTAA Training Workflow Guide

**Training Method:** 5-Fold Cross-Validation (for robust model evaluation and selection)

**Why Cross-Validation?**
- Estimates generalization performance on medium-sized datasets (~10K images)
- Prevents overfitting to a single validation split
- Ensures model robustness across different data splits
- Provides statistical confidence (mean ± std) for research reporting
- Essential for fair model comparison (ResNet50 vs DenseNet vs MobileNet, etc.)

---

## Training Steps

### **Step 1: Imports**

```python
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.append('../')
sys.path.append('./')

from dataset_loader import SplitFolderDatasetLoader
from utils.checkpoint_manager import CheckpointManager
from utils.training_engine import TrainingEngine
from utils.metrics_evaluator import (
    calculate_metrics, print_metrics, plot_confusion_matrix,
    plot_roc_curve, plot_training_history
)
```

---

### **Step 2: Load Dataset**

```python
loader = SplitFolderDatasetLoader(root_dir='../../dataset')
classes = loader.get_classes()
num_classes = loader.get_num_classes()
```

---

### **Step 3: Create Dataset Class**

```python
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
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# Transformations
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
```

---

### **Step 4: Prepare Data for Cross-Validation**

```python
# Combine train + validation for 5-fold CV
X_train, y_train = loader.load_split_paths('train', shuffle=True)
X_val, y_val = loader.load_split_paths('valid')
X_all = X_train + X_val
y_all = np.concatenate([y_train, y_val])

# Test set (untouched until final evaluation)
X_test, y_test = loader.load_split_paths('test')
test_dataset = DFUDataset(X_test, y_test, transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize 5-fold stratified cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

### **Step 5: Setup Device & Loss Function**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
```

---

### **Step 6: 5-Fold Cross-Validation Training**

```python
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all, y_all), 1):
    print(f"\n{'='*60}\nFOLD {fold}/5\n{'='*60}")
    
    # Prepare fold data
    X_train_fold = [X_all[i] for i in train_idx]
    y_train_fold = y_all[train_idx]
    X_val_fold = [X_all[i] for i in val_idx]
    y_val_fold = y_all[val_idx]
    
    train_dataset = DFUDataset(X_train_fold, y_train_fold, transform=train_transform)
    val_dataset = DFUDataset(X_val_fold, y_val_fold, transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Setup training
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    checkpoint_manager = CheckpointManager(checkpoint_dir=f'checkpoints/resnet50_fold{fold}')
    engine = TrainingEngine(model=model, device=device)
    
    # Train
    history = engine.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=30,
        scheduler=scheduler,
        checkpoint_manager=checkpoint_manager,
        verbose=True
    )
    
    # Store results
    best_val_acc = max(history['val_acc'])
    fold_results.append({
        'fold': fold,
        'best_val_acc': best_val_acc,
        'history': history
    })
    print(f"Fold {fold} Best Accuracy: {best_val_acc*100:.2f}%")

# Cross-validation summary
avg_acc = np.mean([r['best_val_acc'] for r in fold_results])
std_acc = np.std([r['best_val_acc'] for r in fold_results])
print(f"\n5-FOLD CV RESULTS: {avg_acc*100:.2f}% ± {std_acc*100:.2f}%")
```

---

### **Step 7: Test Set Evaluation**

```python
# Load best fold model
best_fold_idx = np.argmax([r['best_val_acc'] for r in fold_results])
best_fold_num = fold_results[best_fold_idx]['fold']

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

checkpoint_manager = CheckpointManager(checkpoint_dir=f'checkpoints/resnet50_fold{best_fold_num}')
checkpoint_manager.load_best_model(model, metric_name='accuracy')

engine = TrainingEngine(model=model, device=device)
test_loss, test_acc, predictions, true_labels = engine.evaluate(test_loader, criterion, verbose=True)

print(f"\nTest Accuracy: {test_acc*100:.2f}%")
```

---

### **Step 8: Calculate Metrics**

```python
# Get probabilities for AUC
model.eval()
all_probs = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.to(device))
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())

y_pred_proba = np.vstack(all_probs)

# Calculate all metrics
metrics = calculate_metrics(
    y_true=true_labels,
    y_pred=predictions,
    y_pred_proba=y_pred_proba,
    class_names=classes,
    average='macro'
)

print_metrics(metrics, title="ResNet50 Results")
```

---

### **Step 9: Visualizations**

```python
# Confusion Matrix
plot_confusion_matrix(
    y_true=true_labels,
    y_pred=predictions,
    class_names=classes,
    normalize=True,
    save_path='results/resnet50_confusion_matrix.png'
)

# ROC Curve
plot_roc_curve(
    y_true=true_labels,
    y_pred_proba=y_pred_proba,
    class_names=classes,
    save_path='results/resnet50_roc_curve.png'
)

# Training History
plot_training_history(
    fold_results[best_fold_idx]['history'],
    save_path='results/resnet50_training_history.png'
)
```

---

## Key Parameters

- **batch_size**: 32 (default, as per paper)
- **epochs**: 30 (default, as per paper)
- **momentum**: 0.8 (for SGD optimizer)
- **learning_rate**: 0.001 with StepLR scheduler
- **num_workers**: 4 (CPU processes for data loading, NOT number of classes)
- **num_classes**: 4 (Grade 1, 2, 3, 4)
- **cv_folds**: 5 (stratified cross-validation)

---

## Metrics Calculated

All research-standard metrics are automatically calculated:
- **Accuracy**
- **Precision** (PPV)
- **Recall/Sensitivity** (TPR)
- **Specificity** (TNR)
- **F1-Score**
- **AUC-ROC**
- **MCC** (Matthews Correlation Coefficient)

---

## Quick Tips

✓ GPU will be used automatically if available  
✓ Best model from best fold is automatically saved  
✓ Checkpoints saved every 5 epochs  
✓ Progress bars show training progress  
✓ Test set untouched until final evaluation  
✓ Results include mean ± std for research reporting
