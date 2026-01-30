"""
Test script for Dataset Loader
Tests if the DFU dataset is properly loaded
"""

import sys
import os

# Add the classification folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models', 'classification'))

from dataset_loader import SplitFolderDatasetLoader


def test_dataset_loader():
    """Test the dataset loader with DFU dataset"""
    
    print("=" * 70)
    print("TESTING DATASET LOADER")
    print("=" * 70)
    
    # Path to your dataset (UPDATE THIS PATH!)
    # After downloading from Kaggle, extract to one of these locations:
    dataset_paths = [
        "./dataset",  # Same level as this script
        "./data/dfu-dataset",  # Inside data folder
        "/Users/manaralharbi/Desktop/KHOTAA/dataset",  # Absolute path
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f"\n✅ Found dataset at: {dataset_path}")
            break
    
    if dataset_path is None:
        print("\n❌ DATASET NOT FOUND!")
        print("\n📥 Please download the dataset from:")
        print("   https://www.kaggle.com/datasets/khalidsiddiqui2003/dfu-dataset-annotated-into-4-classes")
        print("\n📂 Extract it to one of these locations:")
        for path in dataset_paths:
            print(f"   - {os.path.abspath(path)}")
        print("\n📋 Expected folder structure:")
        print("   dataset/")
        print("   ├── train/")
        print("   │   ├── class1/")
        print("   │   ├── class2/")
        print("   │   ├── class3/")
        print("   │   └── class4/")
        print("   ├── valid/  (or val/)")
        print("   └── test/")
        return False
    
    print("\n" + "=" * 70)
    print("INITIALIZING DATASET LOADER")
    print("=" * 70)
    
    try:
        # Initialize loader
        loader = SplitFolderDatasetLoader(root_dir=dataset_path)
        
        print("\n✅ Dataset loader initialized successfully!")
        
        # Get class information
        print("\n" + "=" * 70)
        print("CLASS INFORMATION")
        print("=" * 70)
        classes = loader.get_classes()
        num_classes = loader.get_num_classes()
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {classes}")
        print(f"Class to index mapping: {loader.class_to_idx}")
        
        # Test each split
        print("\n" + "=" * 70)
        print("TESTING SPLITS")
        print("=" * 70)
        
        for split in ['train', 'valid', 'test']:
            try:
                print(f"\n🔍 Testing '{split}' split...")
                X_paths, y_labels = loader.load_split_paths(split)
                
                print(f"✅ {split.upper()} split loaded:")
                print(f"   - Total images: {len(X_paths)}")
                print(f"   - Labels shape: {y_labels.shape}")
                print(f"   - Unique labels: {np.unique(y_labels)}")
                
                # Get class distribution
                class_counts = loader.get_class_counts(split)
                print(f"   - Class distribution:")
                for class_name, count in class_counts.items():
                    print(f"     • {class_name}: {count} images")
                
                # Show sample paths
                print(f"   - Sample image paths:")
                for i in range(min(3, len(X_paths))):
                    print(f"     • {X_paths[i]}")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not load '{split}' split")
                print(f"   Error: {str(e)}")
        
        # Test loading actual images (just a few)
        print("\n" + "=" * 70)
        print("TESTING IMAGE LOADING")
        print("=" * 70)
        
        try:
            print("\n🖼️  Loading 5 sample images from train split...")
            X_train, y_train = loader.load_split_paths('train')
            
            # Load just 5 images
            sample_paths = X_train[:5]
            sample_labels = y_train[:5]
            
            import cv2
            for i, (path, label) in enumerate(zip(sample_paths, sample_labels)):
                img = cv2.imread(path)
                if img is not None:
                    print(f"✅ Image {i+1}: {os.path.basename(path)}")
                    print(f"   - Label: {label} ({classes[label]})")
                    print(f"   - Shape: {img.shape}")
                    print(f"   - Size: {img.shape[1]}x{img.shape[0]}")
                else:
                    print(f"❌ Failed to load: {path}")
            
            print("\n✅ Image loading test passed!")
            
        except Exception as e:
            print(f"❌ Image loading test failed: {str(e)}")
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n🎉 Your dataset loader is working correctly!")
        print("   You can now use it in your model notebooks.\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import numpy as np
    success = test_dataset_loader()
    sys.exit(0 if success else 1)
