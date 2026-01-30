"""
Metrics & Evaluation Utilities
Provides comprehensive evaluation metrics and visualization tools
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, class_names=None, average='macro'):
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        average: Averaging method for multi-class ('macro', 'micro', 'weighted')
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Per-class metrics
    if class_names is not None:
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
    
    return metrics


def print_metrics(metrics, title="Evaluation Metrics"):
    """
    Pretty print metrics
    
    Args:
        metrics (dict): Dictionary of metrics
        title (str): Title for the metrics display
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Overall metrics
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   Precision: {metrics['precision']*100:.2f}%")
    print(f"   Recall:    {metrics['recall']*100:.2f}%")
    print(f"   F1-Score:  {metrics['f1_score']*100:.2f}%")
    
    # Per-class metrics (if available)
    per_class_metrics = {k: v for k, v in metrics.items() 
                        if any(metric in k for metric in ['precision_', 'recall_', 'f1_'])}
    
    if per_class_metrics:
        print(f"\n📈 Per-Class Metrics:")
        classes = set([k.split('_', 1)[1] for k in per_class_metrics.keys()])
        for class_name in classes:
            print(f"\n   {class_name}:")
            if f'precision_{class_name}' in metrics:
                print(f"      Precision: {metrics[f'precision_{class_name}']*100:.2f}%")
            if f'recall_{class_name}' in metrics:
                print(f"      Recall:    {metrics[f'recall_{class_name}']*100:.2f}%")
            if f'f1_{class_name}' in metrics:
                print(f"      F1-Score:  {metrics[f'f1_{class_name}']*100:.2f}%")
    
    print(f"\n{'='*60}\n")


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
        print(f"✅ Confusion matrix saved: {save_path}")
    
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
        print(f"✅ Training history plot saved: {save_path}")
    
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
    """Track metrics during training"""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def update(self, train_loss=None, train_acc=None, val_loss=None, val_acc=None):
        """Update metrics"""
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
        """Print summary of tracked metrics"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY".center(60))
        print("="*60)
        
        print(f"\n📊 Best Results:")
        print(f"   Best Val Accuracy: {self.get_best_value('val_acc')*100:.2f}% "
              f"(Epoch {self.get_best_epoch('val_acc')})")
        print(f"   Best Val Loss: {self.get_best_value('val_loss'):.4f} "
              f"(Epoch {self.get_best_epoch('val_loss')})")
        print(f"   Best Train Accuracy: {self.get_best_value('train_acc')*100:.2f}% "
              f"(Epoch {self.get_best_epoch('train_acc')})")
        print(f"   Best Train Loss: {self.get_best_value('train_loss'):.4f} "
              f"(Epoch {self.get_best_epoch('train_loss')})")
        
        print(f"\n📈 Final Results:")
        print(f"   Final Train Accuracy: {self.metrics['train_acc'][-1]*100:.2f}%")
        print(f"   Final Val Accuracy: {self.metrics['val_acc'][-1]*100:.2f}%")
        print(f"   Final Train Loss: {self.metrics['train_loss'][-1]:.4f}")
        print(f"   Final Val Loss: {self.metrics['val_loss'][-1]:.4f}")
        
        print("\n" + "="*60 + "\n")


# Example usage (commented out)
"""
# Calculate metrics
metrics = calculate_metrics(y_true, y_pred, class_names=['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'])
print_metrics(metrics, title="Test Set Evaluation")

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, class_names=['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'])

# Print classification report
print_classification_report(y_true, y_pred, class_names=['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'])

# Plot training history
plot_training_history(history, save_path='training_curves.png')

# Track metrics during training
tracker = MetricsTracker()
tracker.update(train_loss=0.5, train_acc=0.85, val_loss=0.6, val_acc=0.82)
tracker.summary()
"""
