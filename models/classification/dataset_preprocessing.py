"""
Preprocessing for DFU classification.
Unified across all models.

Augmentation strategy aligned with DenseNet121 DFU notebook:
  - RandomHorizontalFlip + RandomVerticalFlip (both 50%)
  - RandomRotation ±20°
  - RandomAffine (zoom / scale ±20%)
  - ColorJitter  brightness=0.1, contrast=0.1
  - ImageNet normalization
"""

import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


class DFUPreprocessing:
    """
    Preprocessing for DFU dataset.
    Provides train and valid/test transforms, class-distribution plots,
    and sample preprocessing visualisation.
    """

    # ── ImageNet normalization constants ──────────────────────────────────
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)
    IMAGE_SIZE = (224, 224)
    SEED = 42

    def __init__(self, image_size=None):
        """Initialize with default DFU preprocessing configuration."""
        if image_size is not None:
            self.IMAGE_SIZE = image_size

        self.mean = self.MEAN
        self.std  = self.STD

        # Build transforms
        self.train_transforms      = self._build_train()
        self.valid_test_transforms = self._build_valid_test()

        print("[DFUPreprocessing] Initialized")
        print(f"[DFUPreprocessing] Image size: {self.IMAGE_SIZE[0]}x{self.IMAGE_SIZE[1]}")
        print(f"[DFUPreprocessing] Train: with augmentation")
        print(f"[DFUPreprocessing] Valid/Test: no augmentation")

    # ─── Transform builders ───────────────────────────────────────────────
    def _build_train(self):
        """
        Build training transforms with augmentation.

        Augmentation steps (aligned with DenseNet121 DFU notebook):
        1. Resize to 224×224
        2. RandomHorizontalFlip (p=0.5)
        3. RandomVerticalFlip   (p=0.5)
        4. RandomRotation       ±20°
        5. RandomAffine         scale 0.8–1.2  (equivalent to RandomZoom ±20%)
        6. ColorJitter          brightness=0.1, contrast=0.1
        7. ToTensor             → [0, 1]
        8. Normalize            ImageNet mean / std
        """
        return transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),   # zoom ±20 %
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def _build_valid_test(self):
        """
        Build validation / test transforms (no augmentation).

        Steps:
        1. Resize to 224×224
        2. ToTensor → [0, 1]
        3. Normalize  ImageNet mean / std
        """
        return transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    # ─── Public accessors ─────────────────────────────────────────────────
    def get_train_transforms(self):
        """Get training transforms (with augmentation)."""
        return self.train_transforms

    def get_valid_test_transforms(self):
        """Get validation/test transforms (no augmentation)."""
        return self.valid_test_transforms

    # ─── Inverse normalisation (for display) ──────────────────────────────
    @staticmethod
    def inverse_normalize(tensor, mean=MEAN, std=STD):
        """Convert a normalised tensor back to [0, 1] for display."""
        inv = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1.0 / s for s in std],
        )
        return torch.clamp(inv(tensor), 0, 1)

    # ─── Visualise augmentation on a single image ─────────────────────────
    def show_augmentation_samples(self, image_path, n_samples=8, title='Data Augmentation Samples'):
        """
        Apply the training augmentation pipeline to a single image
        multiple times and display the results in a grid.

        Args:
            image_path (str): Path to the source image.
            n_samples  (int): Number of augmented versions to show.
            title      (str): Figure title.
        """
        img = Image.open(image_path).convert('RGB')
        n_cols = min(4, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        axes = np.array(axes).ravel()

        for i in range(n_samples):
            aug_tensor = self.train_transforms(img)
            aug_img = self.inverse_normalize(aug_tensor).permute(1, 2, 0).numpy()
            axes[i].imshow(aug_img)
            axes[i].set_title(f'Aug {i + 1}', fontsize=9)
            axes[i].axis('off')

        # Hide unused axes
        for j in range(n_samples, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    # ─── Show preprocessing results (before / after) ─────────────────────
    def show_preprocessing_samples(self, image_paths, labels, class_names,
                                   n_samples=4, title='Preprocessing Results'):
        """
        Display original vs. preprocessed (augmented + normalised) images
        side by side for inspection before training.

        Args:
            image_paths (array): Array of image file paths.
            labels      (array): Corresponding integer labels.
            class_names (list) : Class name strings.
            n_samples   (int)  : Number of images to display.
            title       (str)  : Figure title.
        """
        rng = np.random.RandomState(self.SEED)
        indices = rng.choice(len(image_paths), size=min(n_samples, len(image_paths)), replace=False)

        fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 3))
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

        col_titles = ['Original', 'Resized (224×224)', 'Augmented + Normalised']
        for ax, ct in zip(axes[0] if n_samples > 1 else [axes], col_titles):
            ax.set_title(ct, fontsize=11, fontweight='bold')

        resize_tf = transforms.Resize(self.IMAGE_SIZE)

        for row, idx in enumerate(indices):
            img = Image.open(image_paths[idx]).convert('RGB')
            lbl = class_names[labels[idx]]
            ax_row = axes[row] if n_samples > 1 else axes

            # Original
            ax_row[0].imshow(img)
            ax_row[0].set_ylabel(lbl, fontsize=10, fontweight='bold', rotation=0, labelpad=60)
            ax_row[0].set_yticks([])
            ax_row[0].set_xticks([])

            # Resized only
            resized = resize_tf(img)
            ax_row[1].imshow(resized)
            ax_row[1].axis('off')

            # Augmented + normalised (inverted for display)
            aug_tensor = self.train_transforms(img)
            aug_img = self.inverse_normalize(aug_tensor).permute(1, 2, 0).numpy()
            ax_row[2].imshow(aug_img)
            ax_row[2].axis('off')

        plt.tight_layout()
        plt.show()

    # ─── Class distribution bar chart ─────────────────────────────────────
    @staticmethod
    def plot_class_distribution(labels, class_names, title='Class Distribution',
                                figsize=None, save_path=None):
        """
        Plot a bar chart of class counts.

        Args:
            labels      : 1-D array of integer labels.
            class_names : List of class name strings.
            title       : Figure title.
            figsize     : Optional (w, h) tuple.
            save_path   : If provided, save figure to this path.
        """
        from collections import Counter
        counts = Counter(labels)
        names  = [class_names[i] for i in range(len(class_names))]
        values = [counts.get(i, 0) for i in range(len(class_names))]
        total  = sum(values)

        if figsize is None:
            figsize = (max(6, len(class_names) * 1.5), 4)

        palette = sns.color_palette('tab10', len(class_names))
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(names, values, color=palette)
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')

        for bar, val in zip(bars, values):
            pct = val / total * 100 if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[DFUPreprocessing] Distribution plot saved → {save_path}")
        plt.show()

    # ─── Before / after augmentation class distribution comparison ────────
    @staticmethod
    def plot_class_distribution_comparison(labels_before, labels_after,
                                           class_names,
                                           title='Class Distribution — Before vs After Augmentation',
                                           figsize=None, save_path=None):
        """
        Side-by-side grouped bar chart comparing class distribution
        before and after augmentation.

        Args:
            labels_before : 1-D integer labels (original).
            labels_after  : 1-D integer labels (after augmentation / oversampling).
            class_names   : List of class name strings.
            title         : Figure title.
            figsize       : Optional (w, h).
            save_path     : Optional path to save figure.
        """
        from collections import Counter
        c_before = Counter(labels_before)
        c_after  = Counter(labels_after)

        n = len(class_names)
        before_vals = [c_before.get(i, 0) for i in range(n)]
        after_vals  = [c_after.get(i, 0)  for i in range(n)]

        if figsize is None:
            figsize = (max(8, n * 2), 5)

        x = np.arange(n)
        width = 0.35

        fig, ax = plt.subplots(figsize=figsize)
        bars1 = ax.bar(x - width / 2, before_vals, width, label='Before Augmentation', color='steelblue')
        bars2 = ax.bar(x + width / 2, after_vals,  width, label='After Augmentation',  color='darkorange')

        for bar in list(bars1) + list(bars2):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)

        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[DFUPreprocessing] Comparison plot saved → {save_path}")
        plt.show()