"""
Checkpoint Manager - Model Saving & Loading Utilities
Handles saving and loading PyTorch model checkpoints with cross-validation support
"""

import torch
import os
import json
from pathlib import Path


class CheckpointManager:
    """
    Universal checkpoint manager for PyTorch models
    Supports fold-based organization for cross-validation experiments
    """
    
    def __init__(self, base_dir='runs', experiment_name='experiment'):
        """
        Initialize checkpoint manager
        
        Args:
            base_dir (str): Base directory for all experiments
            experiment_name (str): Name of this experiment
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        self.exp_dir = self.base_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
    
    def get_fold_dir(self, fold_index):
        """
        Get directory for specific fold
        
        Args:
            fold_index (int): Fold index (0-based)
        
        Returns:
            Path: Directory path for the fold
        """
        fold_dir = self.exp_dir / f"fold_{fold_index + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        return fold_dir
    
    def get_model_path(self, fold_index, filename):
        """
        Get path for model file in specific fold
        
        Args:
            fold_index (int): Fold index
            filename (str): Base filename (without extension)
        
        Returns:
            Path: Full path to model file
        """
        fold_dir = self.get_fold_dir(fold_index)
        return fold_dir / filename
    
    def save_checkpoint(self, fold_index, model, optimizer, epoch, metrics, filename=None):
        """
        Save model checkpoint for a specific fold
        
        Args:
            fold_index (int): Fold index (0-based)
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch (int): Current epoch number
            metrics (dict): Training metrics (loss, accuracy, etc.)
            filename (str, optional): Custom filename for checkpoint
        
        Returns:
            str: Path to saved checkpoint
        """
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint_path = self.get_model_path(fold_index, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"SUCCESS: Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, fold_index, model, optimizer=None, filename=None):
        """
        Load model checkpoint from specific fold
        
        Args:
            fold_index (int): Fold index (0-based)
            model: PyTorch model
            optimizer: PyTorch optimizer (optional)
            filename (str): Checkpoint filename
        
        Returns:
            dict: Checkpoint data (epoch, metrics)
        """
        if filename is None:
            # Load latest checkpoint in fold
            fold_dir = self.get_fold_dir(fold_index)
            checkpoints = list(fold_dir.glob('*.pth'))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in fold {fold_index + 1}")
            checkpoint_path = max(checkpoints, key=os.path.getctime)
        else:
            checkpoint_path = self.get_model_path(fold_index, filename)
        
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"SUCCESS: Checkpoint loaded: {checkpoint_path}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Metrics: {checkpoint['metrics']}")
        
        return checkpoint
    
    def save_metrics(self, fold_index, metrics_dict):
        """
        Save metrics for specific fold
        
        Args:
            fold_index (int): Fold index (0-based)
            metrics_dict (dict): Dictionary of metrics to save
        """
        path = self.get_model_path(fold_index, 'metrics.json')
        with open(path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"SUCCESS: Metrics saved: {path}")
    
    def load_metrics(self, fold_index):
        """
        Load metrics from specific fold
        
        Args:
            fold_index (int): Fold index (0-based)
        
        Returns:
            dict: Metrics dictionary
        """
        path = self.get_model_path(fold_index, 'metrics.json')
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_history(self, fold_index, history):
        """
        Save training history for specific fold
        
        Args:
            fold_index (int): Fold index (0-based)
            history (dict): Training history (losses, accuracies per epoch)
        """
        path = self.get_model_path(fold_index, 'history.json')
        with open(path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"SUCCESS: History saved: {path}")
    
    def load_history(self, fold_index):
        """
        Load training history from specific fold
        
        Args:
            fold_index (int): Fold index (0-based)
        
        Returns:
            dict: Training history
        """
        path = self.get_model_path(fold_index, 'history.json')
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_model(self, fold_index, model, filename='model'):
        """
        Save PyTorch model for specific fold
        
        Args:
            fold_index (int): Fold index (0-based)
            model: PyTorch model
            filename (str): Base filename (without extension)
        
        Returns:
            str: Path to saved model
        """
        path = self.get_model_path(fold_index, filename + '.pt')
        torch.save(model.state_dict(), path)
        print(f"SUCCESS: Model saved: {path}")
        return str(path)
    
    def load_model(self, fold_index, create_model_fn, filename='model'):
        """
        Load PyTorch model from specific fold
        
        Args:
            fold_index (int): Fold index (0-based)
            create_model_fn (callable): Function that creates model architecture
            filename (str): Base filename (without extension)
        
        Returns:
            model: Loaded PyTorch model
        """
        path = self.get_model_path(fold_index, filename + '.pt')
        model = create_model_fn()
        model.load_state_dict(torch.load(path))
        print(f"SUCCESS: Model loaded: {path}")
        return model
    
    def save_best_model(self, fold_index, model, metrics, metric_name='accuracy'):
        """
        Save best model for specific fold based on metric
        
        Args:
            fold_index (int): Fold index (0-based)
            model: PyTorch model
            metrics (dict): Current metrics
            metric_name (str): Metric to track for best model
        """
        filename = f'best_{metric_name}'
        path = self.get_model_path(fold_index, filename + '.pt')
        score_path = self.get_model_path(fold_index, filename + '_score.json')
        
        # Save model
        torch.save(model.state_dict(), path)
        
        # Save score
        with open(score_path, 'w') as f:
            json.dump({'score': metrics.get(metric_name, 0)}, f)
        
        print(f"SUCCESS: Best model saved: {path}")
    
    def load_best_model(self, fold_index, create_model_fn, metric_name='accuracy'):
        """
        Load best saved model for specific fold
        
        Args:
            fold_index (int): Fold index (0-based)
            create_model_fn (callable): Function that creates model architecture
            metric_name (str): Metric name used for best model
        
        Returns:
            model: Loaded PyTorch model
        """
        filename = f'best_{metric_name}'
        path = self.get_model_path(fold_index, filename + '.pt')
        score_path = self.get_model_path(fold_index, filename + '_score.json')
        
        model = create_model_fn()
        model.load_state_dict(torch.load(path))
        
        # Load score
        with open(score_path, 'r') as f:
            score = json.load(f)['score']
        
        print(f"SUCCESS: Best model loaded: {path}")
        print(f"   {metric_name}: {score}")
        
        return model
    
    def load_best_score(self, fold_index, metric_name='accuracy'):
        """
        Load best score for specific fold
        
        Args:
            fold_index (int): Fold index (0-based)
            metric_name (str): Metric name used for best model
        
        Returns:
            float: Best score value
        """
        filename = f'best_{metric_name}'
        score_path = self.get_model_path(fold_index, filename + '_score.json')
        with open(score_path, 'r') as f:
            return json.load(f)['score']


def save_model(model, filepath):
    """
    Quick save function for model only (no fold structure)
    
    Args:
        model: PyTorch model
        filepath (str): Path to save model
    """
    torch.save(model.state_dict(), filepath)
    print(f"SUCCESS: Model saved: {filepath}")


def load_model(create_model_fn, filepath):
    """
    Quick load function for model only (no fold structure)
    
    Args:
        create_model_fn (callable): Function that creates model architecture
        filepath (str): Path to load model from
    
    Returns:
        model: Loaded PyTorch model
    """
    model = create_model_fn()
    model.load_state_dict(torch.load(filepath))
    print(f"SUCCESS: Model loaded: {filepath}")
    return model


# Example usage (commented out)
"""
# Initialize checkpoint manager for cross-validation
manager = CheckpointManager(base_dir='runs', experiment_name='resnet50_cv')

# Save checkpoint during training (fold 0, epoch 10)
manager.save_checkpoint(
    fold_index=0,
    model=model,
    optimizer=optimizer,
    epoch=10,
    metrics={'loss': 0.25, 'accuracy': 0.92}
)

# Save metrics for fold
manager.save_metrics(fold_index=0, metrics_dict={'val_acc': 0.95, 'val_loss': 0.15})

# Save training history for fold
manager.save_history(fold_index=0, history={'train_loss': [...], 'val_loss': [...]})

# Save best model for fold
manager.save_best_model(
    fold_index=0,
    model=model,
    metrics={'accuracy': 0.95, 'loss': 0.15},
    metric_name='accuracy'
)

# Load best model from fold
def create_model():
    return ResNet50(num_classes=4)

model = manager.load_best_model(fold_index=0, create_model_fn=create_model)

# Load checkpoint
checkpoint = manager.load_checkpoint(fold_index=0, model=model, optimizer=optimizer)

# Quick save/load (without fold structure)
save_model(model, 'my_model.pt')
model = load_model(create_model, 'my_model.pt')
"""
