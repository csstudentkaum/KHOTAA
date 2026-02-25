import os
import glob
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class SplitFolderDatasetLoader:
    """
    Dataset loader for Diabetic Foot Ulcer dataset.
    Supports pre-split folders: train / {valid|val} / test.
    Auto-detects whether the validation folder is named 'valid' or 'val'.
    """

    def __init__(
        self,
        root_dir,
        splits=None,
        allowed_exts=(".jpg", ".jpeg", ".png", ".bmp"),
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.allowed_exts = allowed_exts

        # ── Auto-detect available splits ──────────────────────────────────
        if splits is not None:
            self.splits = list(splits)
        else:
            self.splits = self._detect_splits()

        # ── Determine the training directory for class discovery ──────────
        train_dir = os.path.join(self.root_dir, "train")
        if not os.path.isdir(train_dir):
            raise ValueError(f"Expected 'train' directory at: {train_dir}")

        class_names = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ])

        if not class_names:
            raise ValueError(f"No class folders found inside: {train_dir}")

        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        print(f"[DatasetLoader] Root: {self.root_dir}")
        print(f"[DatasetLoader] Splits: {self.splits}")
        print(f"[DatasetLoader] Classes ({len(self.class_names)}): {self.class_names}")

    # ─── Private helpers ──────────────────────────────────────────────────
    def _detect_splits(self):
        """Auto-detect split directories (handles 'valid' or 'val')."""
        subdirs = set(
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        )

        splits = []
        if "train" in subdirs:
            splits.append("train")
        # Accept either 'valid' or 'val'
        if "valid" in subdirs:
            splits.append("valid")
        elif "val" in subdirs:
            splits.append("val")
        if "test" in subdirs:
            splits.append("test")

        if not splits:
            raise ValueError(
                f"No recognised split folders (train/valid/val/test) found in {self.root_dir}. "
                f"Found: {subdirs}"
            )
        return splits

    def _scan_split(self, split):
        # Allow the caller to use the canonical name 'valid' even if the
        # folder on disk is 'val' (and vice-versa).
        actual_split = split
        split_dir = os.path.join(self.root_dir, split)
        if not os.path.isdir(split_dir):
            # Try the alternative name
            alt = {"valid": "val", "val": "valid"}.get(split)
            if alt:
                alt_dir = os.path.join(self.root_dir, alt)
                if os.path.isdir(alt_dir):
                    actual_split = alt
                    split_dir = alt_dir
            if not os.path.isdir(split_dir) and not os.path.isdir(alt_dir if alt else ""):
                raise ValueError(f"Split directory does not exist: {split_dir}")

        X_paths = []
        y_labels = []

        for class_name in self.class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"[WARN] Class folder missing in {actual_split}: {class_dir}")
                continue

            for ext in self.allowed_exts:
                pattern = os.path.join(class_dir, f"*{ext}")
                for fpath in glob.glob(pattern):
                    X_paths.append(fpath)
                    y_labels.append(self.class_to_idx[class_name])

        X_paths  = np.array(X_paths)
        y_labels = np.array(y_labels, dtype=np.int64)

        print(f"[DatasetLoader] Split '{actual_split}': {len(X_paths)} images")
        return X_paths, y_labels

    # ─── Public loading methods ───────────────────────────────────────────
    def load_split_paths(self, split, shuffle=False):
        """Load image paths and integer labels."""
        X_paths, y_labels = self._scan_split(split)

        if shuffle:
            idx = np.random.permutation(len(X_paths))
            X_paths  = X_paths[idx]
            y_labels = y_labels[idx]

        return X_paths, y_labels

    def load_split_images(self, split, to_rgb=True, shuffle=False):
        """Load actual images and labels."""
        X_paths, y_labels = self.load_split_paths(split, shuffle=shuffle)

        images = []
        for path in X_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"[WARN] Could not load: {path}")
                continue
            if to_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

        return images, y_labels

    # ─── Class / metadata accessors ───────────────────────────────────────
    def get_classes(self):
        return self.class_names

    def get_num_classes(self):
        return len(self.class_names)

    def get_class_counts(self, split):
        _, y = self._scan_split(split)
        counter = Counter(y)
        return {self.class_names[k]: v for k, v in sorted(counter.items())}

    # ─── Dataset directory structure overview ─────────────────────────────
    def print_structure(self):
        """Print a summary of the dataset directory structure."""
        print(f"\n{'='*50}")
        print(f"Dataset root: {self.root_dir}")
        print(f"{'='*50}")
        for entry in sorted(os.listdir(self.root_dir)):
            full = os.path.join(self.root_dir, entry)
            kind = 'DIR' if os.path.isdir(full) else 'FILE'
            print(f"  [{kind}] {entry}")
        print()

        for split in self.splits:
            counts = self.get_class_counts(split)
            total = sum(counts.values())
            print(f"  Split '{split}' ({total} images):")
            for cls, cnt in counts.items():
                pct = cnt / total * 100 if total else 0
                print(f"    {cls}: {cnt}  ({pct:.1f}%)")
        print(f"{'='*50}\n")

    # ─── Class distribution bar chart ─────────────────────────────────────
    def plot_class_distribution(self, split, title=None, figsize=None, save_path=None):
        """
        Plot a bar chart showing the class distribution for a given split.

        Args:
            split     : One of 'train', 'valid'/'val', 'test'.
            title     : Optional custom title.
            figsize   : Optional (w, h) tuple.
            save_path : If set, save the figure to this path.
        """
        counts = self.get_class_counts(split)
        names  = list(counts.keys())
        values = list(counts.values())
        total  = sum(values)

        if title is None:
            title = f'Class Distribution — {split.capitalize()} Set'
        if figsize is None:
            figsize = (max(6, len(names) * 1.5), 4)

        palette = sns.color_palette('tab10', len(names))
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
            print(f"[DatasetLoader] Plot saved → {save_path}")
        plt.show()

    def plot_all_splits_distribution(self, figsize=None, save_path=None):
        """
        Plot class distributions for all available splits side-by-side.
        """
        n = len(self.splits)
        if figsize is None:
            figsize = (max(6, len(self.class_names) * 1.5) * n, 4)

        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]

        palette = sns.color_palette('tab10', len(self.class_names))

        for ax, split in zip(axes, self.splits):
            counts = self.get_class_counts(split)
            names  = list(counts.keys())
            values = list(counts.values())
            total  = sum(values)

            bars = ax.bar(names, values, color=palette)
            ax.set_title(f'{split.capitalize()} ({total})', fontweight='bold')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')

            for bar, val in zip(bars, values):
                pct = val / total * 100 if total > 0 else 0
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)

        fig.suptitle('Class Distribution — All Splits', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[DatasetLoader] Plot saved → {save_path}")
        plt.show()

    # ─── Show sample images from a split ──────────────────────────────────
    def show_samples(self, split, n_cols=4, n_rows=2, title=None):
        """Display a grid of sample images from the given split."""
        X, y = self.load_split_paths(split, shuffle=True)
        if title is None:
            title = f'{split.capitalize()} Set — Sample Images'

        total = min(n_cols * n_rows, len(X))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        axes = np.array(axes).ravel()

        for i in range(total):
            img = cv2.imread(X[i])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
            axes[i].set_title(self.class_names[y[i]], fontsize=9)
            axes[i].axis('off')

        for j in range(total, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
