import os
import glob
import cv2
import numpy as np
from collections import Counter


class SplitFolderDatasetLoader:
    """
    Dataset loader for Diabetic Foot Ulcer dataset.
    """

    def __init__(
        self,
        root_dir,
        splits=("train", "valid", "test"),
        allowed_exts=(".jpg", ".jpeg", ".png", ".bmp"),
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.splits = list(splits)
        self.allowed_exts = allowed_exts

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

    def _scan_split(self, split):
        if split not in self.splits:
            raise ValueError(f"Unknown split '{split}'. Available: {self.splits}")

        split_dir = os.path.join(self.root_dir, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split directory does not exist: {split_dir}")

        X_paths = []
        y_labels = []

        for class_name in self.class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"[WARN] Class folder missing in {split}: {class_dir}")
                continue

            for ext in self.allowed_exts:
                pattern = os.path.join(class_dir, f"*{ext}")
                for fpath in glob.glob(pattern):
                    X_paths.append(fpath)
                    y_labels.append(self.class_to_idx[class_name])

        X_paths = np.array(X_paths)
        y_labels = np.array(y_labels, dtype=np.int64)

        print(f"[DatasetLoader] Split '{split}': {len(X_paths)} images")
        return X_paths, y_labels

    def load_split_paths(self, split, shuffle=False):
        """Load image paths and integer labels."""
        X_paths, y_labels = self._scan_split(split)

        if shuffle:
            idx = np.random.permutation(len(X_paths))
            X_paths = X_paths[idx]
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

    def get_classes(self):
        return self.class_names

    def get_num_classes(self):
        return len(self.class_names)

    def get_class_counts(self, split):
        _, y = self._scan_split(split)
        counter = Counter(y)
        return {self.class_names[k]: v for k, v in sorted(counter.items())}
