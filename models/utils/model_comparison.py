"""
Model Comparison - Compare Multiple Models Across All Metrics
Helps select the best performing model after training all architectures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


class ModelComparison:
    """
    Compare multiple trained models and select the best one
    Supports comparison across accuracy, F1, inference time, and other metrics
    """
    
    def __init__(self):
        """Initialize model comparison"""
        self.model_results = []
        self.model_names = []
    
    def add_model_result(self, model_name, cv_results, test_results=None, inference_time=None):
        """
        Add results from one trained model
        
        Args:
            model_name (str): Name of the model (e.g., 'ResNet50', 'DenseNet')
            cv_results (dict): Cross-validation results with mean ± std
                Example: {
                    'val_accuracy': {'mean': 0.94, 'std': 0.015},
                    'val_loss': {'mean': 0.23, 'std': 0.05},
                    'f1_score': {'mean': 0.93, 'std': 0.012}
                }
            test_results (dict): Optional test set results
                Example: {
                    'test_accuracy': 0.945,
                    'test_loss': 0.22,
                    'precision': 0.94,
                    'recall': 0.93,
                    'f1_score': 0.935,
                    'specificity': 0.98,
                    'sensitivity': 0.93,
                    'mcc': 0.91,
                    'auc': 0.97
                }
            inference_time (dict): Optional inference time statistics
                Example: {
                    'avg_time_per_image': 0.0028,
                    'images_per_second': 357.1
                }
        """
        result = {
            'model_name': model_name,
            'cv_val_accuracy_mean': cv_results.get('val_accuracy', {}).get('mean', 0),
            'cv_val_accuracy_std': cv_results.get('val_accuracy', {}).get('std', 0),
            'cv_val_loss_mean': cv_results.get('val_loss', {}).get('mean', 0),
            'cv_val_loss_std': cv_results.get('val_loss', {}).get('std', 0),
        }
        
        # Add F1 if available
        if 'f1_score' in cv_results:
            result['cv_f1_mean'] = cv_results['f1_score']['mean']
            result['cv_f1_std'] = cv_results['f1_score']['std']
        
        # Add test results if provided
        if test_results:
            result['test_accuracy'] = test_results.get('test_accuracy', 0)
            result['test_loss'] = test_results.get('test_loss', 0)
            result['test_precision'] = test_results.get('precision', 0)
            result['test_recall'] = test_results.get('recall', 0)
            result['test_f1'] = test_results.get('f1_score', 0)
            result['test_specificity'] = test_results.get('specificity', 0)
            result['test_sensitivity'] = test_results.get('sensitivity', 0)
            result['test_mcc'] = test_results.get('mcc', 0)
            result['test_auc'] = test_results.get('auc', 0)
        
        # Add inference time if provided
        if inference_time:
            result['inference_time_ms'] = inference_time.get('avg_time_per_image', 0) * 1000
            result['throughput_fps'] = inference_time.get('images_per_second', 0)
        
        self.model_results.append(result)
        self.model_names.append(model_name)
    
    def get_comparison_table(self):
        """
        Get comparison table as pandas DataFrame
        
        Returns:
            pd.DataFrame: Comparison table with all models and metrics
        """
        df = pd.DataFrame(self.model_results)
        
        # Sort by CV validation accuracy (descending)
        if 'cv_val_accuracy_mean' in df.columns:
            df = df.sort_values('cv_val_accuracy_mean', ascending=False)
        
        return df
    
    def print_comparison_table(self):
        """Print formatted comparison table"""
        df = self.get_comparison_table()
        
        print("\n" + "=" * 100)
        print("MODEL COMPARISON RESULTS")
        print("=" * 100)
        
        # Cross-Validation Results
        print("\n1. CROSS-VALIDATION RESULTS (5-Fold)")
        print("-" * 100)
        cv_cols = ['model_name', 'cv_val_accuracy_mean', 'cv_val_accuracy_std', 'cv_val_loss_mean']
        if 'cv_f1_mean' in df.columns:
            cv_cols.append('cv_f1_mean')
        
        cv_df = df[cv_cols].copy()
        cv_df.columns = ['Model', 'Val Acc (Mean)', 'Val Acc (Std)', 'Val Loss (Mean)'] + \
                        (['F1 (Mean)'] if 'cv_f1_mean' in df.columns else [])
        print(cv_df.to_string(index=False))
        
        # Test Results (if available)
        if 'test_accuracy' in df.columns:
            print("\n2. TEST SET RESULTS (Hold-out)")
            print("-" * 100)
            test_cols = ['model_name', 'test_accuracy', 'test_precision', 'test_recall', 
                        'test_f1', 'test_specificity', 'test_sensitivity', 'test_mcc', 'test_auc']
            test_cols = [col for col in test_cols if col in df.columns]
            
            test_df = df[test_cols].copy()
            print(test_df.to_string(index=False))
        
        # Inference Time (if available)
        if 'inference_time_ms' in df.columns:
            print("\n3. INFERENCE TIME")
            print("-" * 100)
            inf_df = df[['model_name', 'inference_time_ms', 'throughput_fps']].copy()
            inf_df.columns = ['Model', 'Time per Image (ms)', 'Throughput (FPS)']
            print(inf_df.to_string(index=False))
        
        print("\n" + "=" * 100)
    
    def select_best_model(self, criterion='cv_val_accuracy_mean', minimize=False):
        """
        Select best model based on a criterion
        
        Args:
            criterion (str): Metric to use for selection
                Options: 'cv_val_accuracy_mean', 'test_accuracy', 'test_f1', 
                        'test_mcc', 'inference_time_ms', etc.
            minimize (bool): If True, select model with minimum value (for loss, time)
        
        Returns:
            dict: Best model result dictionary
        """
        df = self.get_comparison_table()
        
        if criterion not in df.columns:
            raise ValueError(f"Criterion '{criterion}' not found in results. Available: {list(df.columns)}")
        
        if minimize:
            best_idx = df[criterion].idxmin()
        else:
            best_idx = df[criterion].idxmax()
        
        best_model = df.loc[best_idx].to_dict()
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL SELECTED BY: {criterion} ({'minimize' if minimize else 'maximize'})")
        print(f"{'='*60}")
        print(f"Model: {best_model['model_name']}")
        print(f"{criterion}: {best_model[criterion]:.4f}")
        print(f"{'='*60}\n")
        
        return best_model
    
    def select_top_k_models(self, k=3, criterion='cv_val_accuracy_mean', minimize=False):
        """
        Select top K models
        
        Args:
            k (int): Number of top models to select
            criterion (str): Metric to rank by
            minimize (bool): If True, select models with minimum values
        
        Returns:
            pd.DataFrame: Top K models
        """
        df = self.get_comparison_table()
        
        if criterion not in df.columns:
            raise ValueError(f"Criterion '{criterion}' not found")
        
        df_sorted = df.sort_values(criterion, ascending=minimize)
        top_k = df_sorted.head(k)
        
        print(f"\n{'='*60}")
        print(f"TOP {k} MODELS BY: {criterion}")
        print(f"{'='*60}")
        for idx, row in top_k.iterrows():
            print(f"{row['model_name']:20s} - {criterion}: {row[criterion]:.4f}")
        print(f"{'='*60}\n")
        
        return top_k
    
    def get_ensemble_weights_by_performance(self, criterion='cv_val_accuracy_mean'):
        """
        Calculate weights for each model based on their performance
        Weights are normalized so they sum to 1.0
        
        Args:
            criterion (str): Metric to use for weighting
        
        Returns:
            list: Normalized weights for each fold
        """
        df = self.get_comparison_table()
        
        if criterion in ['val_loss', 'test_loss', 'inference_time_ms']:
            # Lower is better, so invert
            scores = [1.0 / (row[criterion] + 1e-8) for _, row in df.iterrows()]
        else:
            # Higher is better
            scores = [row[criterion] for _, row in df.iterrows()]
        
        # Normalize weights
        total = sum(scores)
        weights = [score / total for score in scores]
        
        print(f"Ensemble weights based on {criterion}:")
        for (_, row), weight in zip(df.iterrows(), weights):
            print(f"  {row['model_name']:20s}: weight={weight:.4f}")
        
        return weights
    
    def plot_comparison(self, save_path='results/model_comparison.png'):
        """
        Create comprehensive comparison plots
        
        Args:
            save_path (str): Path to save the plot
        """
        df = self.get_comparison_table()
        
        # Determine number of subplots
        n_plots = 0
        if 'cv_val_accuracy_mean' in df.columns:
            n_plots += 1
        if 'test_accuracy' in df.columns:
            n_plots += 1
        if 'test_f1' in df.columns:
            n_plots += 1
        if 'inference_time_ms' in df.columns:
            n_plots += 1
        
        if n_plots == 0:
            print("No metrics available for plotting")
            return
        
        # Create figure
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot 1: CV Validation Accuracy
        if 'cv_val_accuracy_mean' in df.columns:
            ax = axes[plot_idx]
            models = df['model_name']
            means = df['cv_val_accuracy_mean'] * 100
            stds = df['cv_val_accuracy_std'] * 100
            
            bars = ax.bar(range(len(models)), means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_xlabel('Model')
            ax.set_ylabel('Validation Accuracy (%)')
            ax.set_title('Cross-Validation Accuracy (5-Fold)')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Highlight best
            best_idx = means.idxmax()
            bars[best_idx].set_color('green')
            bars[best_idx].set_alpha(0.9)
            
            plot_idx += 1
        
        # Plot 2: Test Accuracy
        if 'test_accuracy' in df.columns:
            ax = axes[plot_idx]
            models = df['model_name']
            test_acc = df['test_accuracy'] * 100
            
            bars = ax.bar(range(len(models)), test_acc, alpha=0.7, color='orange')
            ax.set_xlabel('Model')
            ax.set_ylabel('Test Accuracy (%)')
            ax.set_title('Test Set Accuracy')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Highlight best
            best_idx = test_acc.idxmax()
            bars[best_idx].set_color('green')
            
            plot_idx += 1
        
        # Plot 3: F1 Score
        if 'test_f1' in df.columns:
            ax = axes[plot_idx]
            models = df['model_name']
            f1_scores = df['test_f1'] * 100
            
            bars = ax.bar(range(len(models)), f1_scores, alpha=0.7, color='blue')
            ax.set_xlabel('Model')
            ax.set_ylabel('F1 Score (%)')
            ax.set_title('Test Set F1 Score')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Highlight best
            best_idx = f1_scores.idxmax()
            bars[best_idx].set_color('green')
            
            plot_idx += 1
        
        # Plot 4: Inference Time
        if 'inference_time_ms' in df.columns:
            ax = axes[plot_idx]
            models = df['model_name']
            inf_times = df['inference_time_ms']
            
            bars = ax.bar(range(len(models)), inf_times, alpha=0.7, color='red')
            ax.set_xlabel('Model')
            ax.set_ylabel('Inference Time (ms)')
            ax.set_title('Inference Time per Image')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Highlight fastest (minimum)
            best_idx = inf_times.idxmin()
            bars[best_idx].set_color('green')
            
            plot_idx += 1
        
        plt.tight_layout()
        
        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
        plt.close()
    
    def save_results(self, save_path='results/model_comparison.json'):
        """
        Save comparison results to JSON file
        
        Args:
            save_path (str): Path to save JSON file
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.model_results, f, indent=4)
        
        print(f"Comparison results saved to: {save_path}")
    
    def generate_latex_table(self):
        """
        Generate LaTeX table for research paper
        
        Returns:
            str: LaTeX table code
        """
        df = self.get_comparison_table()
        
        latex_str = "\\begin{table}[h]\n\\centering\n\\caption{Model Comparison Results}\n"
        latex_str += "\\begin{tabular}{|l|c|c|c|c|c|}\n\\hline\n"
        latex_str += "Model & Val Acc (\\%) & Test Acc (\\%) & F1 Score & MCC & Inference (ms) \\\\\n\\hline\n"
        
        for _, row in df.iterrows():
            model_name = row['model_name']
            cv_acc = f"{row.get('cv_val_accuracy_mean', 0)*100:.2f} $\\pm$ {row.get('cv_val_accuracy_std', 0)*100:.2f}"
            test_acc = f"{row.get('test_accuracy', 0)*100:.2f}" if 'test_accuracy' in row else "-"
            f1 = f"{row.get('test_f1', 0):.4f}" if 'test_f1' in row else "-"
            mcc = f"{row.get('test_mcc', 0):.4f}" if 'test_mcc' in row else "-"
            inf_time = f"{row.get('inference_time_ms', 0):.2f}" if 'inference_time_ms' in row else "-"
            
            latex_str += f"{model_name} & {cv_acc} & {test_acc} & {f1} & {mcc} & {inf_time} \\\\\n"
        
        latex_str += "\\hline\n\\end{tabular}\n\\end{table}"
        
        print("\nLaTeX Table Generated:")
        print("=" * 80)
        print(latex_str)
        print("=" * 80)
        
        return latex_str


# Example usage (commented out)
"""
# After training all 7 models with 5-fold CV, compare them

from model_comparison import ModelComparison

comparison = ModelComparison()

# Add ResNet50 results
comparison.add_model_result(
    model_name='ResNet50',
    cv_results={
        'val_accuracy': {'mean': 0.9423, 'std': 0.0115},
        'val_loss': {'mean': 0.234, 'std': 0.045}
    },
    test_results={
        'test_accuracy': 0.9378,
        'precision': 0.94,
        'recall': 0.93,
        'f1_score': 0.935,
        'specificity': 0.98,
        'sensitivity': 0.93,
        'mcc': 0.91,
        'auc': 0.97
    },
    inference_time={
        'avg_time_per_image': 0.0028,
        'images_per_second': 357.1
    }
)

# Add DenseNet results
comparison.add_model_result(
    model_name='DenseNet',
    cv_results={
        'val_accuracy': {'mean': 0.9512, 'std': 0.0098},
        'val_loss': {'mean': 0.198, 'std': 0.038}
    },
    test_results={
        'test_accuracy': 0.9467,
        'precision': 0.95,
        'recall': 0.94,
        'f1_score': 0.945,
        'specificity': 0.99,
        'sensitivity': 0.94,
        'mcc': 0.93,
        'auc': 0.98
    },
    inference_time={
        'avg_time_per_image': 0.0045,
        'images_per_second': 222.2
    }
)

# ... Add all other 5 models (MobileNet, GoogLeNet, ResNet101, EfficientNetV2S, PFCNN+DRNN) ...

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

# Get top 3 models for ensemble
top_3_models = comparison.select_top_k_models(k=3, criterion='test_accuracy')

# Get performance-based weights for ensemble
ensemble_weights = comparison.get_ensemble_weights_by_performance(criterion='cv_val_accuracy_mean')

# Create comparison plots
comparison.plot_comparison(save_path='results/all_models_comparison.png')

# Save results to JSON for later analysis
comparison.save_results(save_path='results/all_models_comparison.json')

# Generate LaTeX table for research paper
latex_table = comparison.generate_latex_table()

# Save LaTeX table to file
with open('results/models_table.tex', 'w') as f:
    f.write(latex_table)

print("Model comparison complete!")
"""
