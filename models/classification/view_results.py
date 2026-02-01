#!/usr/bin/env python3
"""
View Training Results - Quick Summary of All Model Checkpoints
Usage: python view_results.py [model_name]
Example: python view_results.py mobilenet
"""

import torch
import glob
import numpy as np
import sys
import os

def view_fold_results(model_name='mobilenet'):
    """Display training results for all folds of a model"""
    
    print("\n" + "=" * 80)
    print(f"{model_name.upper()} - CROSS-VALIDATION RESULTS")
    print("=" * 80)
    
    # Find all fold directories
    pattern = f'checkpoints/{model_name}_fold*/fold_*/'
    fold_dirs = sorted(glob.glob(pattern))
    
    if not fold_dirs:
        print(f"\n❌ No results found for '{model_name}'")
        print(f"   Searched in: {pattern}")
        return
    
    fold_results = []
    
    # Collect results from each fold
    for fold_dir in fold_dirs:
        fold_name = fold_dir.split('/')[-3].replace(f'{model_name}_fold', 'Fold ')
        checkpoints = glob.glob(f'{fold_dir}checkpoint_epoch_*.pth')
        
        if not checkpoints:
            continue
        
        # Find best accuracy and final epoch
        best_acc = 0
        best_epoch = 0
        final_epoch = 0
        best_train_acc = 0
        
        for ckpt_file in checkpoints:
            ckpt = torch.load(ckpt_file, map_location='cpu')
            metrics = ckpt['metrics']
            val_acc = metrics.get('val_acc', 0)
            train_acc = metrics.get('train_acc', 0)
            epoch = ckpt['epoch']
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_train_acc = train_acc
            
            if epoch > final_epoch:
                final_epoch = epoch
        
        fold_results.append({
            'name': fold_name,
            'best_val_acc': best_acc,
            'best_train_acc': best_train_acc,
            'best_epoch': best_epoch,
            'total_epochs': final_epoch
        })
    
    # Display results table
    print(f"\n{'Fold':<10} {'Best Val Acc':<15} {'Train Acc':<12} {'Best Epoch':<12} {'Total Epochs':<15}")
    print("-" * 80)
    
    for result in fold_results:
        print(f"{result['name']:<10} "
              f"{result['best_val_acc']*100:>13.2f}% "
              f"{result['best_train_acc']*100:>10.2f}% "
              f"{result['best_epoch']:>11} "
              f"{result['total_epochs']:>14}")
    
    # Calculate statistics
    if fold_results:
        val_accs = [r['best_val_acc'] for r in fold_results]
        mean_val_acc = np.mean(val_accs) * 100
        std_val_acc = np.std(val_accs) * 100
        
        print("-" * 80)
        print(f"\n📊 STATISTICS:")
        print(f"   Mean Validation Accuracy: {mean_val_acc:.2f}% ± {std_val_acc:.2f}%")
        print(f"   Best Single Fold:         {max(val_accs)*100:.2f}%")
        print(f"   Worst Single Fold:        {min(val_accs)*100:.2f}%")
        print(f"   Completed Folds:          {len(fold_results)}/5")
        
        if len(fold_results) < 5:
            print(f"   Status:                   ⏳ Training in progress...")
        else:
            print(f"   Status:                   ✅ All folds complete!")
    
    print("\n" + "=" * 80 + "\n")

def list_available_models():
    """List all models with available checkpoints"""
    checkpoint_dirs = glob.glob('checkpoints/*_fold*/')
    models = set()
    
    for dir_path in checkpoint_dirs:
        model_name = os.path.basename(dir_path.rstrip('/')).rsplit('_fold', 1)[0]
        models.add(model_name)
    
    return sorted(models)

if __name__ == '__main__':
    # Get model name from command line or use default
    if len(sys.argv) > 1:
        model_name = sys.argv[1].lower()
    else:
        # List available models
        available = list_available_models()
        if available:
            print("\n📁 Available models with results:")
            for model in available:
                print(f"   - {model}")
            print(f"\nShowing results for: {available[0]}")
            model_name = available[0]
        else:
            print("\n❌ No training results found in checkpoints/")
            sys.exit(1)
    
    view_fold_results(model_name)
