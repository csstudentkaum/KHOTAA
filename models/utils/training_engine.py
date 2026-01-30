"""
Training Engine - PyTorch Training & Evaluation Runner
Handles the complete training loop, evaluation, and cross-validation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=7, min_delta=0.0, verbose=True):
        """
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change to qualify as improvement
            verbose (bool): Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        """
        Check if should stop training
        
        Args:
            val_loss (float): Current validation loss
            epoch (int): Current epoch number
        
        Returns:
            bool: True if should stop training
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'WARNING: EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'SUCCESS: Early stopping triggered at epoch {epoch}')
                    print(f'Best validation loss was {self.best_loss:.4f} at epoch {self.best_epoch}')
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        
        return self.early_stop


class TrainingEngine:
    """Complete training and evaluation engine for PyTorch models"""
    
    def __init__(self, model, device='cuda'):
        """
        Initialize training engine
        
        Args:
            model: PyTorch model
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Training on: {self.device}")
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc='Training')
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, val_loader, criterion):
        """
        Evaluate model on validation/test set
        
        Args:
            val_loader: DataLoader for validation data
            criterion: Loss function
        
        Returns:
            tuple: (average_loss, accuracy, predictions, labels)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Evaluating')
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc, np.array(all_predictions), np.array(all_labels)
    
    def train(self, train_loader, val_loader, criterion, optimizer, 
              num_epochs, scheduler=None, checkpoint_manager=None,
              early_stopping_patience=7, use_early_stopping=True, verbose=True):
        """
        Complete training loop with early stopping support
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            criterion: Loss function
            optimizer: Optimizer
            num_epochs (int): Maximum number of epochs to train
            scheduler: Learning rate scheduler (optional)
            checkpoint_manager: CheckpointManager instance (optional)
            early_stopping_patience (int): Epochs to wait before early stopping (default: 7)
            use_early_stopping (bool): Whether to use early stopping (default: True)
            verbose (bool): Show detailed progress (default: True)
        
        Returns:
            dict: Training history with actual epochs trained
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'stopped_epoch': num_epochs
        }
        
        best_val_acc = 0.0
        
        # Initialize early stopping
        early_stopper = None
        if use_early_stopping:
            early_stopper = EarlyStopping(patience=early_stopping_patience, verbose=verbose)
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc, _, _ = self.evaluate(val_loader, criterion)
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Learning Rate: {current_lr:.6f}")
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print epoch results
            print(f"\nEpoch {epoch+1} Results:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
            
            # Save checkpoint
            if checkpoint_manager is not None:
                checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    epoch=epoch+1,
                    metrics={
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }
                )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if checkpoint_manager is not None:
                    checkpoint_manager.save_best_model(
                        model=self.model,
                        metrics={'val_acc': val_acc, 'val_loss': val_loss}
                    )
                if verbose:
                    print(f"⭐ New best model! Val Acc: {val_acc*100:.2f}%")
            
            # Check early stopping
            if early_stopper is not None:
                if early_stopper(val_loss, epoch+1):
                    history['stopped_epoch'] = epoch + 1
                    print(f"\n{'='*60}")
                    print(f"SUCCESS: Training stopped early at epoch {epoch+1}/{num_epochs}")
                    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
                    print(f"{'='*60}\n")
                    return history
        
        # Training completed without early stopping
        history['stopped_epoch'] = num_epochs
        if verbose:
            print(f"\n{'='*60}")
            print(f"SUCCESS: Training Complete!")
            print(f"Completed all {num_epochs} epochs")
            print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
            print(f"{'='*60}\n")
        
        return history


# Example usage (commented out)
"""
# Initialize training engine
engine = TrainingEngine(model=model, device='cuda')

# Train model
history = engine.train(
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    num_epochs=30,
    scheduler=scheduler,
    checkpoint_manager=checkpoint_manager
)

# Evaluate on test set
test_loss, test_acc, predictions, labels = engine.evaluate(
    test_loader, 
    criterion=nn.CrossEntropyLoss()
)
print(f"Test Accuracy: {test_acc*100:.2f}%")
"""


class CrossValidationHelper:
    """
    Stratified K-Fold cross-validation helper for classification tasks
    Automates fold splitting and model evaluation across folds
    """
    
    def __init__(self, n_splits=5, random_state=42, shuffle=True):
        """
        Initialize cross-validation helper
        
        Args:
            n_splits (int): Number of folds
            random_state (int): Random seed for reproducibility
            shuffle (bool): Whether to shuffle data before splitting
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
    
    def run(self, X, y, create_model_fn, train_eval_fn, verbose=True, start_fold=1):
        """
        Run cross-validation
        
        Args:
            X: Feature data (array-like, can be indices or actual data)
            y: Labels (array-like)
            create_model_fn: Function that creates and returns a new model instance
                           Signature: create_model_fn(fold_index) -> model
            train_eval_fn: Function that trains and evaluates the model
                          Signature: train_eval_fn(model, X_train, y_train, X_val, y_val, fold_index) -> metrics_dict
            verbose (bool): Whether to print progress
            start_fold (int): Starting fold number (1-based, useful for resuming)
        
        Returns:
            tuple: (fold_metrics, summary)
                - fold_metrics: List of metric dictionaries, one per fold
                - summary: Dictionary with mean and std for each metric
        """
        X = np.array(X)
        y = np.array(y)
        
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            fold_no = fold_idx + 1
            
            # Skip folds before start_fold (useful for resuming)
            if fold_no < start_fold:
                if verbose:
                    print("=" * 60)
                    print(f"Skipping Fold {fold_no}/{self.n_splits} (start_fold={start_fold})")
                continue
            
            if verbose:
                print("\n" + "=" * 60)
                print(f"Fold {fold_no}/{self.n_splits}")
                print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)}")
                print("=" * 60)
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Create new model for this fold
            model = create_model_fn(fold_idx)
            
            # Train and evaluate
            metrics = train_eval_fn(
                model, X_train, y_train, X_val, y_val, fold_idx
            )
            
            if verbose:
                print(f"\nFold {fold_no} metrics: {metrics}")
            
            fold_metrics.append(metrics)
        
        # Calculate summary statistics
        summary = {}
        if fold_metrics:
            metric_names = fold_metrics[0].keys()
            for name in metric_names:
                values = [m[name] for m in fold_metrics]
                summary[name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }
        
        if verbose:
            print("\n" + "=" * 60)
            print("Cross-Validation Summary:")
            print("=" * 60)
            for name, stats in summary.items():
                print(f"{name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print("=" * 60 + "\n")
        
        return fold_metrics, summary


# Example cross-validation usage (commented out)
"""
# Prepare data for CV
X_indices = np.arange(len(train_dataset))  # Use indices
y_labels = np.array([label for _, label in train_dataset])  # Extract labels

# Define model creation function
def create_model(fold_index):
    model = ResNet50(num_classes=4)
    return model

# Define train/eval function
def train_eval_fold(model, X_train, y_train, X_val, y_val, fold_index):
    # X_train, y_train are indices and labels
    # Create data loaders using these indices
    from torch.utils.data import Subset
    
    train_subset = Subset(train_dataset, X_train)
    val_subset = Subset(train_dataset, X_val)
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32)
    
    # Train the model
    engine = TrainingEngine(model, device='cuda')
    history = engine.train(
        train_loader, val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters()),
        num_epochs=30
    )
    
    # Return metrics
    return {
        'val_accuracy': history['val_acc'][-1],
        'val_loss': history['val_loss'][-1],
        'train_accuracy': history['train_acc'][-1]
    }

# Run cross-validation
cv_helper = CrossValidationHelper(n_splits=5, random_state=42)
fold_metrics, summary = cv_helper.run(
    X=X_indices,
    y=y_labels,
    create_model_fn=create_model,
    train_eval_fn=train_eval_fold
)

print(f"Mean Val Accuracy: {summary['val_accuracy']['mean']:.4f}")
"""
