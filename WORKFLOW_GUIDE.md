# KHOTAA Training Workflow Guide

**Training Method:** 5-Fold Cross-Validation (for robust model evaluation and selection)

**Training Configuration:**
- **Folds:** 5-fold stratified cross-validation
- **Max Epochs:** 30 per fold
- **Early Stopping:** Enabled (patience=7 epochs)
- **Performance Reporting:** Mean ± Std across all folds

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
from dataset_preprocessing import DFUPreprocessing
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

### **Step 3: Initialize Preprocessing**

```python
# Initialize DFU preprocessing with data augmentation
preprocessor = DFUPreprocessing()

# Get transforms
train_transform = preprocessor.get_train_transforms()
val_test_transform = preprocessor.get_valid_test_transforms()
```

**Preprocessing Features:**
- **Training Augmentation:**
  - Resize to 224×224
  - Random horizontal flip (50%)
  - Random vertical flip (30%)
  - Random rotation (±15°)
  - Color jitter (brightness, contrast, saturation, hue)
  - Gaussian blur (simulates focus variance)
  - ImageNet normalization

- **Validation/Test:**
  - Resize to 224×224
  - ImageNet normalization only (no augmentation)

---

### **Step 4: Create Dataset Class**

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
```

---

### **Step 5: Prepare Data for Cross-Validation**

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

### **Step 6: Setup Device, Loss Function & Optimizer**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

# Use helper function to create optimizer with recommended defaults
from models.utils.training_engine import create_optimizer

# Option 1: Use defaults (SGD, lr=0.001, momentum=0.8)
optimizer = create_optimizer(model)

# Option 2: Customize learning rate
optimizer = create_optimizer(model, lr=0.01)

# Option 3: Use Adam instead
optimizer = create_optimizer(model, optimizer_type='adam', lr=0.001)

# Option 4: Full customization
optimizer = create_optimizer(model, optimizer_type='sgd', lr=0.01, momentum=0.9, weight_decay=5e-4)
```

**Note:** The helper function provides recommended defaults based on best practices. For SGD, momentum=0.8 and lr=0.001 work well for most models. You can customize as needed for specific architectures.

---

### **Step 7: 5-Fold Cross-Validation Training**

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
    
    # Setup optimizer using helper function
    optimizer = create_optimizer(model, lr=0.001)  # Uses SGD with momentum=0.8 by default
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
        early_stopping_patience=7,
        use_early_stopping=True,
        verbose=True
    )
    
    # Store results
    best_val_acc = max(history['val_acc'])
    fold_results.append({
        'fold': fold,
        'best_val_acc': best_val_acc,
        'final_val_acc': history['val_acc'][-1],
        'stopped_epoch': history['stopped_epoch'],
        'history': history
    })
    print(f"Fold {fold} - Best Acc: {best_val_acc*100:.2f}% (stopped at epoch {history['stopped_epoch']})")

# Cross-validation summary
avg_acc = np.mean([r['best_val_acc'] for r in fold_results])
std_acc = np.std([r['best_val_acc'] for r in fold_results])
avg_epochs = np.mean([r['stopped_epoch'] for r in fold_results])

print(f"\n{'='*60}")
print(f"5-FOLD CROSS-VALIDATION RESULTS")
print(f"{'='*60}")
print(f"Mean Accuracy: {avg_acc*100:.2f}% ± {std_acc*100:.2f}%")
print(f"Average Epochs: {avg_epochs:.1f}")
print(f"\nIndividual Fold Results:")
for r in fold_results:
    print(f"  Fold {r['fold']}: {r['best_val_acc']*100:.2f}% (epoch {r['stopped_epoch']})")
print(f"{'='*60}")
```

---

### **Step 8: Test Set Evaluation**

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

# Evaluate with inference time tracking
test_loss, test_acc, predictions, true_labels, inference_time = engine.evaluate(
    test_loader, 
    criterion, 
    measure_inference_time=True
)

print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"\nInference Time Statistics:")
print(f"  Total Time: {inference_time['total_time']:.4f}s")
print(f"  Avg Time/Batch: {inference_time['avg_time_per_batch']*1000:.2f}ms ± {inference_time['std_time_per_batch']*1000:.2f}ms")
print(f"  Avg Time/Image: {inference_time['avg_time_per_image']*1000:.2f}ms")
print(f"  Throughput: {inference_time['images_per_second']:.1f} images/second")
```

---

### **Step 9: Calculate Metrics**

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

### **Step 10: Visualizations**

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
- **max_epochs**: 30 per fold (with early stopping)
- **early_stopping_patience**: 7 epochs (stops if no improvement)
- **momentum**: 0.8 (for SGD optimizer)
- **learning_rate**: 0.001 with StepLR scheduler
- **num_workers**: 4 (CPU processes for data loading, NOT number of classes)
- **num_classes**: 4 (Grade 1, 2, 3, 4)
- **cv_folds**: 5 (stratified cross-validation)

---

## Performance Reporting

Results are reported as **mean ± standard deviation across 5 folds**:

```python
# Example output format for research paper:
# Model: ResNet50
# Cross-Validation Accuracy: 94.23% ± 1.15%
# Test Accuracy: 93.78%
# Average Training Epochs: 24.2 (with early stopping)
# Inference Time: 3.2ms per image (312 images/second)
```

Each fold may stop at different epochs due to early stopping, ensuring optimal performance without overfitting.

**Inference Time Metrics:**
- **Total Time**: Total time to process all test images
- **Time per Batch**: Average time to process one batch (with std)
- **Time per Image**: Average time to classify a single image
- **Throughput**: Number of images processed per second

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
- **Inference Time** (ms per image, images per second)

---

## Quick Tips

✓ GPU will be used automatically if available  
✓ Early stopping prevents overfitting (patience=7 epochs)  
✓ Best model from best fold is automatically saved  
✓ Checkpoints saved every 5 epochs  
✓ Progress bars show training progress  
✓ Test set untouched until final evaluation  
✓ Results reported as mean ± std for research papers  
✓ Each fold may stop at different epochs (optimal training)  
✓ Inference time measured on test set (important for deployment)  
✓ Throughput reported in images/second

---

## 🏆 Comparing All 7 Models (Final Model Selection)

After training all 7 models (ResNet50, ResNet101, DenseNet, MobileNet, GoogLeNet, EfficientNetV2S, PFCNN+DRNN) with 5-fold cross-validation, use the **ModelComparison** module to select the best model:

### **Step 11: Model Comparison Setup**

```python
from utils.model_comparison import ModelComparison

# Initialize comparison
comparison = ModelComparison()

# Add ResNet50 results
comparison.add_model_result(
    model_name='ResNet50',
    cv_results={
        'val_accuracy': {'mean': avg_acc_resnet50, 'std': std_acc_resnet50},
        'val_loss': {'mean': avg_loss_resnet50, 'std': std_loss_resnet50}
    },
    test_results={
        'test_accuracy': test_acc_resnet50,
        'test_loss': test_loss_resnet50,
        'precision': metrics_resnet50['precision'],
        'recall': metrics_resnet50['recall'],
        'f1_score': metrics_resnet50['f1_score'],
        'specificity': metrics_resnet50['specificity'],
        'sensitivity': metrics_resnet50['sensitivity'],
        'mcc': metrics_resnet50['mcc'],
        'auc': metrics_resnet50['auc']
    },
    inference_time=inference_time_resnet50
)

# Add DenseNet results
comparison.add_model_result(
    model_name='DenseNet',
    cv_results={
        'val_accuracy': {'mean': avg_acc_densenet, 'std': std_acc_densenet},
        'val_loss': {'mean': avg_loss_densenet, 'std': std_loss_densenet}
    },
    test_results={
        'test_accuracy': test_acc_densenet,
        'test_loss': test_loss_densenet,
        'precision': metrics_densenet['precision'],
        'recall': metrics_densenet['recall'],
        'f1_score': metrics_densenet['f1_score'],
        'specificity': metrics_densenet['specificity'],
        'sensitivity': metrics_densenet['sensitivity'],
        'mcc': metrics_densenet['mcc'],
        'auc': metrics_densenet['auc']
    },
    inference_time=inference_time_densenet
)

# Add MobileNet results
comparison.add_model_result(
    model_name='MobileNet',
    cv_results={
        'val_accuracy': {'mean': avg_acc_mobilenet, 'std': std_acc_mobilenet},
        'val_loss': {'mean': avg_loss_mobilenet, 'std': std_loss_mobilenet}
    },
    test_results={
        'test_accuracy': test_acc_mobilenet,
        'test_loss': test_loss_mobilenet,
        'precision': metrics_mobilenet['precision'],
        'recall': metrics_mobilenet['recall'],
        'f1_score': metrics_mobilenet['f1_score'],
        'specificity': metrics_mobilenet['specificity'],
        'sensitivity': metrics_mobilenet['sensitivity'],
        'mcc': metrics_mobilenet['mcc'],
        'auc': metrics_mobilenet['auc']
    },
    inference_time=inference_time_mobilenet
)

# Add GoogLeNet results
comparison.add_model_result(
    model_name='GoogLeNet',
    cv_results={
        'val_accuracy': {'mean': avg_acc_googlenet, 'std': std_acc_googlenet},
        'val_loss': {'mean': avg_loss_googlenet, 'std': std_loss_googlenet}
    },
    test_results={
        'test_accuracy': test_acc_googlenet,
        'test_loss': test_loss_googlenet,
        'precision': metrics_googlenet['precision'],
        'recall': metrics_googlenet['recall'],
        'f1_score': metrics_googlenet['f1_score'],
        'specificity': metrics_googlenet['specificity'],
        'sensitivity': metrics_googlenet['sensitivity'],
        'mcc': metrics_googlenet['mcc'],
        'auc': metrics_googlenet['auc']
    },
    inference_time=inference_time_googlenet
)

# Add ResNet101 results
comparison.add_model_result(
    model_name='ResNet101',
    cv_results={
        'val_accuracy': {'mean': avg_acc_resnet101, 'std': std_acc_resnet101},
        'val_loss': {'mean': avg_loss_resnet101, 'std': std_loss_resnet101}
    },
    test_results={
        'test_accuracy': test_acc_resnet101,
        'test_loss': test_loss_resnet101,
        'precision': metrics_resnet101['precision'],
        'recall': metrics_resnet101['recall'],
        'f1_score': metrics_resnet101['f1_score'],
        'specificity': metrics_resnet101['specificity'],
        'sensitivity': metrics_resnet101['sensitivity'],
        'mcc': metrics_resnet101['mcc'],
        'auc': metrics_resnet101['auc']
    },
    inference_time=inference_time_resnet101
)

# Add EfficientNetV2S results
comparison.add_model_result(
    model_name='EfficientNetV2S',
    cv_results={
        'val_accuracy': {'mean': avg_acc_efficientnet, 'std': std_acc_efficientnet},
        'val_loss': {'mean': avg_loss_efficientnet, 'std': std_loss_efficientnet}
    },
    test_results={
        'test_accuracy': test_acc_efficientnet,
        'test_loss': test_loss_efficientnet,
        'precision': metrics_efficientnet['precision'],
        'recall': metrics_efficientnet['recall'],
        'f1_score': metrics_efficientnet['f1_score'],
        'specificity': metrics_efficientnet['specificity'],
        'sensitivity': metrics_efficientnet['sensitivity'],
        'mcc': metrics_efficientnet['mcc'],
        'auc': metrics_efficientnet['auc']
    },
    inference_time=inference_time_efficientnet
)

# Add PFCNN+DRNN results
comparison.add_model_result(
    model_name='PFCNN+DRNN',
    cv_results={
        'val_accuracy': {'mean': avg_acc_pfcnn, 'std': std_acc_pfcnn},
        'val_loss': {'mean': avg_loss_pfcnn, 'std': std_loss_pfcnn}
    },
    test_results={
        'test_accuracy': test_acc_pfcnn,
        'test_loss': test_loss_pfcnn,
        'precision': metrics_pfcnn['precision'],
        'recall': metrics_pfcnn['recall'],
        'f1_score': metrics_pfcnn['f1_score'],
        'specificity': metrics_pfcnn['specificity'],
        'sensitivity': metrics_pfcnn['sensitivity'],
        'mcc': metrics_pfcnn['mcc'],
        'auc': metrics_pfcnn['auc']
    },
    inference_time=inference_time_pfcnn
)
```

---

### **Step 12: Print Comparison & Select Best Model**

```python
# Print comprehensive comparison table
comparison.print_comparison_table()

# Select best model by different criteria:

# 1. Best by Cross-Validation Accuracy (most reliable for generalization)
best_model = comparison.select_best_model(criterion='cv_val_accuracy_mean')

# 2. Best by Test Accuracy
best_by_test = comparison.select_best_model(criterion='test_accuracy')

# 3. Best by F1 Score (balanced precision/recall)
best_by_f1 = comparison.select_best_model(criterion='test_f1')

# 4. Best by MCC (Matthews Correlation Coefficient - handles class imbalance)
best_by_mcc = comparison.select_best_model(criterion='test_mcc')

# 5. Fastest model (lowest inference time - important for deployment)
fastest = comparison.select_best_model(criterion='inference_time_ms', minimize=True)

# Get top 3 models for potential ensemble
top_3_models = comparison.select_top_k_models(k=3, criterion='test_accuracy')

# Get performance-based weights for ensemble (if needed)
ensemble_weights = comparison.get_ensemble_weights_by_performance(criterion='cv_val_accuracy_mean')
```

**Example Output:**
```
====================================================================================================
MODEL COMPARISON RESULTS
====================================================================================================

1. CROSS-VALIDATION RESULTS (5-Fold)
----------------------------------------------------------------------------------------------------
Model            Val Acc (Mean)  Val Acc (Std)  Val Loss (Mean)
DenseNet         0.9512          0.0098         0.198
ResNet101        0.9489          0.0102         0.205
ResNet50         0.9423          0.0115         0.234
EfficientNetV2S  0.9401          0.0120         0.241
GoogLeNet        0.9378          0.0125         0.248
MobileNet        0.9245          0.0145         0.289
PFCNN+DRNN       0.9198          0.0156         0.301

2. TEST SET RESULTS (Hold-out)
----------------------------------------------------------------------------------------------------
Model            test_accuracy  test_precision  test_recall  test_f1  test_specificity  test_sensitivity  test_mcc  test_auc
DenseNet         0.9467         0.95            0.94         0.945    0.99              0.94              0.93      0.98
ResNet101        0.9445         0.94            0.94         0.942    0.98              0.94              0.92      0.97
ResNet50         0.9378         0.94            0.93         0.935    0.98              0.93              0.91      0.97
EfficientNetV2S  0.9345         0.93            0.93         0.930    0.98              0.93              0.90      0.96
GoogLeNet        0.9312         0.93            0.92         0.925    0.97              0.92              0.89      0.96
MobileNet        0.9178         0.92            0.91         0.915    0.97              0.91              0.87      0.95
PFCNN+DRNN       0.9134         0.91            0.91         0.910    0.96              0.91              0.86      0.94

3. INFERENCE TIME
----------------------------------------------------------------------------------------------------
Model            Time per Image (ms)  Throughput (FPS)
MobileNet        2.15                 465.1
ResNet50         2.83                 357.1
GoogLeNet        3.21                 311.5
EfficientNetV2S  3.67                 272.5
ResNet101        4.12                 242.7
DenseNet         4.56                 219.3
PFCNN+DRNN       5.89                 169.8

====================================================================================================

============================================================
BEST MODEL SELECTED BY: test_f1 (maximize)
============================================================
Model: DenseNet
test_f1: 0.9450
============================================================

============================================================
BEST MODEL SELECTED BY: inference_time_ms (minimize)
============================================================
Model: MobileNet
inference_time_ms: 2.1500
============================================================
```

---

### **Step 13: Generate Visualizations & Export Results**

```python
# Create comprehensive comparison plots
comparison.plot_comparison(save_path='results/all_models_comparison.png')

# Save comparison results to JSON
comparison.save_results(save_path='results/all_models_comparison.json')

# Generate LaTeX table for research paper
latex_table = comparison.generate_latex_table()

# Save LaTeX table to file
with open('results/models_comparison_table.tex', 'w') as f:
    f.write(latex_table)

print("\nModel comparison complete!")
print("Best model for deployment:", best_by_f1['model_name'])
print("Fastest model:", fastest['model_name'])
```

---

## Final Model Selection Criteria

**For Medical Applications (DFU Classification):**

1. **Primary:** Highest F1-Score or MCC (balanced performance across all classes)
2. **Secondary:** High Specificity (minimize false positives - avoid unnecessary interventions)
3. **Tertiary:** High Sensitivity (minimize false negatives - catch all serious cases)
4. **Consideration:** Inference time (real-time capability for clinical use)

**Recommended Selection Strategy:**
- Use **Cross-Validation Accuracy** for initial ranking
- Verify with **Test F1-Score** and **MCC** for final selection
- Check **inference time** for deployment feasibility
- If top 2-3 models have similar performance, consider ensemble

---