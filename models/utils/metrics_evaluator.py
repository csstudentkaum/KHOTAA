"""
Metrics & Evaluation Utilities
Provides comprehensive evaluation metrics and visualization tools
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, y_pred_proba=None, class_names=None, average='macro'):
    """
    Calculate comprehensive classification metrics as per research standards.
    
    Evaluation measures include:
    - Sensitivity (Recall/True Positive Rate): TP / (TP + FN)
    - Specificity (True Negative Rate): TN / (TN + FP)
    - Precision (Positive Predictive Value): TP / (TP + FP)
    - F-measure (F1-Score): 2PR / (P + R)
    - Accuracy: (TP + TN) / (TP + TN + FP + FN)
    - Area Under Curve (AUC-ROC)
    - Matthews Correlation Coefficient (MCC): correlation between predicted and actual
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities for AUC calculation (optional)
        class_names: List of class names (optional)
        average: Averaging method for multi-class ('macro', 'micro', 'weighted')
    
    Returns:
        dict: Dictionary of comprehensive metrics
    """
    # Confusion matrix for specificity calculation
    cm = confusion_matrix(y_true, y_pred)
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),  # Same as sensitivity
        'sensitivity': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),  # F-measure
        'mcc': matthews_corrcoef(y_true, y_pred)  # Matthews Correlation Coefficient
    }
    
    # Specificity calculation (for multi-class, average across classes)
    if len(cm) == 2:  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['specificity'] = specificity
    else:  # Multi-class: calculate per-class specificity and average
        specificities = []
        for i in range(len(cm)):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(specificity)
        metrics['specificity'] = np.mean(specificities)
    
    # AUC-ROC (if probabilities provided)
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:  # Multi-class
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba, 
                                              multi_class='ovr', average=average)
        except Exception as e:
            metrics['auc'] = None
            print(f"Warning: Could not calculate AUC - {str(e)}")
    
    # Per-class metrics
    if class_names is not None:
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'sensitivity_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
            
            # Per-class specificity
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics[f'specificity_{class_name}'] = specificity
    
    return metrics


def print_metrics(metrics, title="Evaluation Metrics"):
    """
    Pretty print comprehensive metrics including all research-standard measures
    
    Args:
        metrics (dict): Dictionary of metrics
        title (str): Title for the metrics display
    """
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    
    # Overall metrics
    print(f"\nOverall Performance Metrics:")
    print(f"   Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"   Precision:   {metrics['precision']*100:.2f}% (PPV - Positive Predictive Value)")
    print(f"   Recall:      {metrics['recall']*100:.2f}% (Sensitivity/TPR)")
    print(f"   Sensitivity: {metrics['sensitivity']*100:.2f}% (True Positive Rate)")
    print(f"   Specificity: {metrics.get('specificity', 0)*100:.2f}% (True Negative Rate)")
    print(f"   F1-Score:    {metrics['f1_score']*100:.2f}% (F-Measure)")
    print(f"   MCC:         {metrics['mcc']:.4f} (Matthews Correlation Coefficient)")
    
    if 'auc' in metrics and metrics['auc'] is not None:
        print(f"   AUC-ROC:     {metrics['auc']:.4f} (Area Under Curve)")
    
    # Formulas explanation
    print(f"\nMetric Formulas:")
    print(f"   Precision (PPV) = TP / (TP + FP)")
    print(f"   Recall (Sensitivity) = TP / (TP + FN)")
    print(f"   Specificity = TN / (TN + FP)")
    print(f"   F1-Score = 2PR / (P + R)")
    print(f"   MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))")
    
    # Per-class metrics (if available)
    per_class_metrics = {k: v for k, v in metrics.items() 
                        if any(metric in k for metric in ['precision_', 'recall_', 'f1_', 
                                                          'sensitivity_', 'specificity_'])}
    
    if per_class_metrics:
        print(f"\nPer-Class Detailed Metrics:")
        classes = set([k.split('_', 1)[1] for k in per_class_metrics.keys()])
        for class_name in sorted(classes):
            print(f"\n   {class_name}:")
            if f'precision_{class_name}' in metrics:
                print(f"      Precision (PPV):  {metrics[f'precision_{class_name}']*100:.2f}%")
            if f'recall_{class_name}' in metrics:
                print(f"      Recall:           {metrics[f'recall_{class_name}']*100:.2f}%")
            if f'sensitivity_{class_name}' in metrics:
                print(f"      Sensitivity (TPR): {metrics[f'sensitivity_{class_name}']*100:.2f}%")
            if f'specificity_{class_name}' in metrics:
                print(f"      Specificity (TNR): {metrics[f'specificity_{class_name}']*100:.2f}%")
            if f'f1_{class_name}' in metrics:
                print(f"      F1-Score:         {metrics[f'f1_{class_name}']*100:.2f}%")
    
    print(f"\n{'='*70}\n")


def plot_confusion_matrix(y_true, y_pred, class_names, figsize=(10, 8), 
                          normalize=False, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SUCCESS: Confusion matrix saved: {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, class_names=None, save_path=None):
    """
    Plot ROC curve and calculate AUC (Area Under Curve)
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        class_names: List of class names (optional)
        save_path: Path to save the plot (optional)
    """
    n_classes = y_pred_proba.shape[1] if len(y_pred_proba.shape) > 1 else 2
    
    plt.figure(figsize=(10, 8))
    
    if n_classes == 2:  # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
    else:  # Multi-class classification
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and AUC for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            class_label = class_names[i] if class_names else f'Class {i}'
            plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SUCCESS: ROC curve saved: {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history (dict): Training history with keys: train_loss, train_acc, val_loss, val_acc
        save_path: Path to save the plot (optional)
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, [acc*100 for acc in history['train_acc']], 
             'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, [acc*100 for acc in history['val_acc']], 
             'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SUCCESS: Training history plot saved: {save_path}")
    
    plt.show()


def print_classification_report(y_true, y_pred, class_names):
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT".center(60))
    print("="*60 + "\n")
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)


class MetricsTracker:
    """Track metrics during training including loss and accuracy"""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch': []
        }
        self.current_epoch = 0
    
    def update(self, train_loss=None, train_acc=None, val_loss=None, val_acc=None, epoch=None):
        """
        Update metrics for current epoch
        
        Args:
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            epoch: Current epoch number (optional)
        """
        if epoch is not None:
            self.current_epoch = epoch
            self.metrics['epoch'].append(epoch)
        else:
            self.current_epoch += 1
            self.metrics['epoch'].append(self.current_epoch)
            
        if train_loss is not None:
            self.metrics['train_loss'].append(train_loss)
        if train_acc is not None:
            self.metrics['train_acc'].append(train_acc)
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        if val_acc is not None:
            self.metrics['val_acc'].append(val_acc)
    
    def get_best_epoch(self, metric='val_acc'):
        """Get epoch with best metric value"""
        if not self.metrics[metric]:
            return None
        
        if 'loss' in metric:
            best_idx = np.argmin(self.metrics[metric])
        else:
            best_idx = np.argmax(self.metrics[metric])
        
        return best_idx + 1
    
    def get_best_value(self, metric='val_acc'):
        """Get best metric value"""
        if not self.metrics[metric]:
            return None
        
        if 'loss' in metric:
            return min(self.metrics[metric])
        else:
            return max(self.metrics[metric])
    
    def summary(self):
        """Print comprehensive summary of tracked metrics"""
        print("\n" + "="*70)
        print("TRAINING SUMMARY".center(70))
        print("="*70)
        
        if not self.metrics['train_loss']:
            print("\nWARNING: No metrics tracked yet!")
            print("="*70 + "\n")
            return
        
        print(f"\nTotal Epochs Trained: {len(self.metrics['train_loss'])}")
        
        print(f"\nBest Results:")
        if self.metrics['val_acc']:
            best_val_acc_idx = self.get_best_epoch('val_acc') - 1
            print(f"   Best Val Accuracy: {self.get_best_value('val_acc')*100:.2f}% "
                  f"(Epoch {self.get_best_epoch('val_acc')})")
            print(f"      └─ Val Loss at best accuracy: {self.metrics['val_loss'][best_val_acc_idx]:.4f}")
        
        if self.metrics['val_loss']:
            best_val_loss_idx = self.get_best_epoch('val_loss') - 1
            print(f"   Best Val Loss: {self.get_best_value('val_loss'):.4f} "
                  f"(Epoch {self.get_best_epoch('val_loss')})")
            print(f"      └─ Val Accuracy at best loss: {self.metrics['val_acc'][best_val_loss_idx]*100:.2f}%")
        
        if self.metrics['train_acc']:
            print(f"   Best Train Accuracy: {self.get_best_value('train_acc')*100:.2f}% "
                  f"(Epoch {self.get_best_epoch('train_acc')})")
        
        if self.metrics['train_loss']:
            print(f"   Best Train Loss: {self.get_best_value('train_loss'):.4f} "
                  f"(Epoch {self.get_best_epoch('train_loss')})")
        
        print(f"\nFinal Results (Epoch {len(self.metrics['train_loss'])}):")
        if self.metrics['train_acc']:
            print(f"   Final Train Accuracy: {self.metrics['train_acc'][-1]*100:.2f}%")
        if self.metrics['val_acc']:
            print(f"   Final Val Accuracy:   {self.metrics['val_acc'][-1]*100:.2f}%")
        if self.metrics['train_loss']:
            print(f"   Final Train Loss:     {self.metrics['train_loss'][-1]:.4f}")
        if self.metrics['val_loss']:
            print(f"   Final Val Loss:       {self.metrics['val_loss'][-1]:.4f}")
        
        # Learning progression
        if len(self.metrics['train_loss']) > 1:
            train_loss_improvement = self.metrics['train_loss'][0] - self.metrics['train_loss'][-1]
            val_loss_improvement = self.metrics['val_loss'][0] - self.metrics['val_loss'][-1] if self.metrics['val_loss'] else 0
            
            print(f"\nLoss Improvement:")
            print(f"   Train Loss: {self.metrics['train_loss'][0]:.4f} → {self.metrics['train_loss'][-1]:.4f} "
                  f"(Δ {train_loss_improvement:+.4f})")
            if self.metrics['val_loss']:
                print(f"   Val Loss:   {self.metrics['val_loss'][0]:.4f} → {self.metrics['val_loss'][-1]:.4f} "
                      f"(Δ {val_loss_improvement:+.4f})")
        
        print("\n" + "="*70 + "\n")
    
    def plot_metrics(self, save_path=None):
        """
        Plot all tracked metrics (loss and accuracy)
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.metrics['train_loss']:
            print("WARNING: No metrics to plot!")
            return
        
        epochs = self.metrics.get('epoch', range(1, len(self.metrics['train_loss']) + 1))
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(epochs, self.metrics['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
        if self.metrics['val_loss']:
            axes[0].plot(epochs, self.metrics['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
        axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Mark best epoch
        if self.metrics['val_loss']:
            best_epoch = self.get_best_epoch('val_loss')
            best_value = self.get_best_value('val_loss')
            axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (Epoch {best_epoch})')
            axes[0].scatter([best_epoch], [best_value], color='gold', s=100, zorder=5, edgecolors='black')
        
        # Accuracy plot
        if self.metrics['train_acc']:
            axes[1].plot(epochs, [acc*100 for acc in self.metrics['train_acc']], 
                        'b-o', label='Training Accuracy', linewidth=2, markersize=4)
        if self.metrics['val_acc']:
            axes[1].plot(epochs, [acc*100 for acc in self.metrics['val_acc']], 
                        'r-s', label='Validation Accuracy', linewidth=2, markersize=4)
        axes[1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # Mark best epoch
        if self.metrics['val_acc']:
            best_epoch = self.get_best_epoch('val_acc')
            best_value = self.get_best_value('val_acc') * 100
            axes[1].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (Epoch {best_epoch})')
            axes[1].scatter([best_epoch], [best_value], color='gold', s=100, zorder=5, edgecolors='black')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SUCCESS: Metrics plot saved: {save_path}")
        
        plt.show()


def plot_class_distribution(labels, class_names, title="Class Distribution"):
    """
    Plot class distribution bar chart
    
    Args:
        labels: Array of labels
        class_names: List of class names
        title (str): Plot title
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[class_names[u] for u in unique], y=counts, palette='viridis')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def show_sample_predictions(images, true_labels, pred_labels, class_names, rows=3, cols=3):
    """
    Display sample predictions with true and predicted labels
    
    Args:
        images: Array of images
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        rows (int): Number of rows
        cols (int): Number of columns
    """
    plt.figure(figsize=(cols * 4, rows * 4))
    
    for i in range(rows * cols):
        if i >= len(images):
            break
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        
        plt.title(
            f"True: {true_class}\nPred: {pred_class}",
            fontsize=10,
            color=color
        )
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_cv_metric(fold_metrics, metric_name, title=None):
    """
    Plot metric values across cross-validation folds
    
    Args:
        fold_metrics: List of metric dictionaries from each fold
        metric_name (str): Name of metric to plot
        title (str): Plot title (optional)
    """
    if title is None:
        title = f"{metric_name} per Fold"
    
    values = [m[metric_name] for m in fold_metrics]
    folds = np.arange(1, len(fold_metrics) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(folds, values, marker='o', linewidth=2, markersize=8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(folds)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_metrics_to_file(metrics_dict, filepath):
    """
    Save metrics dictionary to JSON file
    
    Args:
        metrics_dict (dict): Dictionary of metrics
        filepath (str): Path to save file
    """
    import json
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"SUCCESS: Metrics saved to: {filepath}")


def load_metrics_from_file(filepath):
    """
    Load metrics dictionary from JSON file
    
    Args:
        filepath (str): Path to metrics file
    
    Returns:
        dict: Metrics dictionary
    """
    import json
    with open(filepath, 'r') as f:
        return json.load(f)


# Example usage (commented out)
"""
# Calculate comprehensive metrics (all research-standard measures)
metrics = calculate_metrics(
    y_true, 
    y_pred, 
    y_pred_proba=model_probabilities,  # Include probabilities for AUC
    class_names=['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
)

# Print all metrics with formulas
print_metrics(metrics, title="Test Set Evaluation")

# Metrics include:
# - Accuracy
# - Precision (PPV - Positive Predictive Value): TP / (TP + FP)
# - Recall/Sensitivity (TPR - True Positive Rate): TP / (TP + FN)
# - Specificity (TNR - True Negative Rate): TN / (TN + FP)
# - F1-Score (F-Measure): 2PR / (P + R)
# - MCC (Matthews Correlation Coefficient)
# - AUC-ROC (Area Under Curve)

# Plot confusion matrix
plot_confusion_matrix(
    y_true, 
    y_pred, 
    class_names=['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
)

# Plot ROC curve with AUC
plot_roc_curve(
    y_true, 
    y_pred_proba, 
    class_names=['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
    save_path='roc_curve.png'
)

# Print classification report
print_classification_report(
    y_true, 
    y_pred, 
    class_names=['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
)

# Plot training history (from history dict)
plot_training_history(history, save_path='training_curves.png')

# Track metrics during training with MetricsTracker
tracker = MetricsTracker()

# During training loop:
for epoch in range(num_epochs):
    # ... training code ...
    tracker.update(
        train_loss=epoch_train_loss, 
        train_acc=epoch_train_acc, 
        val_loss=epoch_val_loss, 
        val_acc=epoch_val_acc,
        epoch=epoch+1
    )

# After training - view summary
tracker.summary()

# Plot all tracked metrics (loss and accuracy curves)
tracker.plot_metrics(save_path='metrics_plot.png')
"""
