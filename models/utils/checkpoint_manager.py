"""
Checkpoint Manager - Model Saving & Loading Utilities
Handles saving and loading PyTorch model checkpoints
"""

import torch
import os
from pathlib import Path


class CheckpointManager:
    """Universal checkpoint manager for PyTorch models"""
    
    def __init__(self, checkpoint_dir='checkpoints'):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir (str): Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, filename=None):
        """
        Save model checkpoint
        
        Args:
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
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, model, optimizer=None, filename=None):
        """
        Load model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer (optional)
            filename (str): Checkpoint filename
        
        Returns:
            dict: Checkpoint data (epoch, metrics)
        """
        if filename is None:
            # Load latest checkpoint
            checkpoints = list(self.checkpoint_dir.glob('*.pth'))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = max(checkpoints, key=os.path.getctime)
        else:
            checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✅ Checkpoint loaded: {checkpoint_path}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Metrics: {checkpoint['metrics']}")
        
        return checkpoint
    
    def save_best_model(self, model, metrics, metric_name='accuracy'):
        """
        Save best model based on metric
        
        Args:
            model: PyTorch model
            metrics (dict): Current metrics
            metric_name (str): Metric to track for best model
        """
        filename = f'best_model_{metric_name}.pth'
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }, checkpoint_path)
        
        print(f"✅ Best model saved: {checkpoint_path}")
        print(f"   {metric_name}: {metrics.get(metric_name, 'N/A')}")
    
    def load_best_model(self, model, metric_name='accuracy'):
        """
        Load best saved model
        
        Args:
            model: PyTorch model
            metric_name (str): Metric name used for best model
        
        Returns:
            dict: Model metrics
        """
        filename = f'best_model_{metric_name}.pth'
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Best model loaded: {checkpoint_path}")
        print(f"   Metrics: {checkpoint['metrics']}")
        
        return checkpoint['metrics']


def save_model(model, filepath):
    """
    Quick save function for model only
    
    Args:
        model: PyTorch model
        filepath (str): Path to save model
    """
    torch.save(model.state_dict(), filepath)
    print(f"✅ Model saved: {filepath}")


def load_model(model, filepath):
    """
    Quick load function for model only
    
    Args:
        model: PyTorch model
        filepath (str): Path to load model from
    """
    model.load_state_dict(torch.load(filepath))
    print(f"✅ Model loaded: {filepath}")
    return model


# Example usage (commented out)
"""
# Initialize checkpoint manager
manager = CheckpointManager(checkpoint_dir='checkpoints')

# Save checkpoint during training
manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=10,
    metrics={'loss': 0.25, 'accuracy': 0.92}
)

# Save best model
manager.save_best_model(
    model=model,
    metrics={'accuracy': 0.95, 'loss': 0.15},
    metric_name='accuracy'
)

# Load checkpoint
checkpoint = manager.load_checkpoint(model, optimizer)

# Load best model
metrics = manager.load_best_model(model, metric_name='accuracy')
"""
