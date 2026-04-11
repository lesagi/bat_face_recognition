#!/usr/bin/env python3
"""
Test script for SiameseNetworkTrainingDataSplitter
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from siamese_data.data_splitter import SiameseNetworkTrainingDataSplitter
from utils.filename_parser import parse_filename_class, group_files_by_class

def test_filename_parsing():
    """Test the filename parsing functionality."""
    print("🔍 Testing filename parsing...")
    
    # Test cases
    test_filenames = [
        "bat--rous1--img001.jpg",
        "bat--rous2--img002--aug001.jpg",
        "img--rous3--photo003.png",
        "bat--rous4--img004--aug123.jpg",
        "invalid_filename.txt",
        "bat--rous5--img005--aug999.jpg"
    ]
    
    for filename in test_filenames:
        result = parse_filename_class(filename)
        if result:
            file_type, class_name, file_id, aug_suffix = result
            print(f"✅ {filename} -> type: {file_type}, class: {class_name}, id: {file_id}, aug: {aug_suffix}")
        else:
            print(f"❌ {filename} -> Failed to parse")

def test_file_grouping(data_path):
    """Test the file grouping functionality."""
    print(f"\n🔍 Testing file grouping for: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Data path does not exist: {data_path}")
        return
    
    # Get all files from the directory
    all_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if not file.startswith('.') and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, file))
    
    print(f"Found {len(all_files)} image files")
    
    if all_files:
        # Show first few files
        print("Sample files:")
        for i, file_path in enumerate(all_files[:5]):
            filename = os.path.basename(file_path)
            print(f"  {i+1}. {filename}")
        
        # Test grouping
        class_files = group_files_by_class(all_files)
        print(f"\nGrouped into {len(class_files)} classes:")
        for class_name, ids in class_files.items():
            print(f"  Class '{class_name}': {len(ids)} files")
            if ids:
                id_key = list(ids.keys())[0]
                sample_filename = os.path.basename(ids[id_key][0])
                print(f"    Sample: {sample_filename}")

def test_data_splitter(data_path, skip_preprocessing=False):
    """Test the complete data splitter."""
    print(f"\n🔍 Testing complete data splitter for: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Data path does not exist: {data_path}")
        return
    
    try:
        # Initialize the data splitter
        splitter = SiameseNetworkTrainingDataSplitter(
            images_dirs_paths_list=[data_path],
            training_portion=0.7,
            mode="combination",
            skip_preprocessing=skip_preprocessing
        )
        
        print(f"✅ Data splitter initialized successfully!")
        print(f"   Found {len(splitter.class_names)} classes: {splitter.class_names}")
        
        # Check if we have training and test data
        if splitter.train_data is not None:
            train_size = splitter.train_data.cardinality().numpy()
            print(f"   Training data: {train_size} pairs")
        else:
            print("   ❌ No training data created")
        
        if splitter.test_data is not None:
            test_size = splitter.test_data.cardinality().numpy()
            print(f"   Test data: {test_size} pairs")
        else:
            print("   ❌ No test data created")
        
        # Test a few samples from training data
        if splitter.train_data is not None and train_size > 0:
            print(f"\n🔍 Testing training data samples...")
            sample_batch = splitter.train_data.take(3)
            
            for i, (v1, v2, label, class_info) in enumerate(sample_batch):
                print(f"  Sample {i+1}:")
                print(f"    Value 1 type: {type(v1.numpy() if hasattr(v1, 'numpy') else v1)}")
                print(f"    Value 2 type: {type(v2.numpy() if hasattr(v2, 'numpy') else v2)}")
                print(f"    Label: {label.numpy()}")
                print(f"    Class: {class_info.numpy().decode('utf-8')}")
                if not skip_preprocessing:
                    # Expect tensors when preprocessing is applied
                    assert isinstance(v1, tf.Tensor) and isinstance(v2, tf.Tensor)
                    print(f"    Tensor shapes: {v1.shape}, {v2.shape}")
                else:
                    # Expect file path strings when skipping preprocessing
                    assert isinstance(v1.numpy().decode('utf-8'), str)
                    assert isinstance(v2.numpy().decode('utf-8'), str)
                    print(f"    Paths: {v1.numpy().decode('utf-8')[:40]}..., {v2.numpy().decode('utf-8')[:40]}...")
        
        return splitter
        
    except Exception as e:
        print(f"❌ Error testing data splitter: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_permutation_mode(data_path):
    """Test the permutation mode."""
    print(f"\n🔍 Testing permutation mode for: {data_path}")
    
    try:
        # Initialize with permutation mode
        splitter = SiameseNetworkTrainingDataSplitter(
            images_dirs_paths_list=[data_path],
            training_portion=0.7,
            mode="permutation"
        )
        
        print(f"✅ Permutation mode data splitter initialized!")
        
        if splitter.train_data is not None:
            train_size = splitter.train_data.cardinality().numpy()
            print(f"   Training data: {train_size} pairs (permutation mode)")
        
        return splitter
        
    except Exception as e:
        print(f"❌ Error testing permutation mode: {e}")
        import traceback
        traceback.print_exc()
        return None

def _create_tiny_dataset(tmp_dir: str, num_classes=3):
    """Create a tiny synthetic dataset with valid filenames and PNG images.
    Structure: flat folder with filenames matching pattern type--class--id[--aug###].png
    """
    rng = np.random.default_rng(0)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    if num_classes <= 3:
        samples = [
            ("r", "A", "img001", None),
            ("r", "A", "img001", "--aug001"),
            ("r", "A", "img002", None),
            ("r", "B", "img010", None),
            ("r", "B", "img011", None),
            ("r", "C", "img100", None),
        ]
    else:
        # Larger dataset for bat_split tests (need 5+ classes)
        samples = []
        for cls_idx in range(num_classes):
            cls_name = chr(ord("A") + cls_idx)
            for img_idx in range(3):
                fid = f"img{cls_idx:02d}{img_idx:02d}"
                samples.append(("r", cls_name, fid, None))
            # Add one augmented image per class
            samples.append(("r", cls_name, f"img{cls_idx:02d}00", "--aug001"))

    for t, cls, fid, aug in samples:
        name = f"{t}--{cls}--{fid}{aug or ''}.png"
        img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        tf.keras.utils.save_img(os.path.join(tmp_dir, name), img)

    return tmp_dir

def test_image_split_all_classes_in_both(data_path):
    """Verify image_split mode puts every class in both train and test."""
    print(f"\n🔍 Testing image_split: every class in both train and test...")

    try:
        splitter = SiameseNetworkTrainingDataSplitter(
            images_dirs_paths_list=[data_path],
            training_portion=0.7,
            mode="combination",
            skip_preprocessing=True,
            split_mode="image_split",
            split_seed=42,
        )

        train_classes = set(splitter.train_class_files.keys())
        test_classes = set(splitter.test_class_files.keys())
        all_classes = set(splitter.class_names)

        assert train_classes == all_classes, (
            f"image_split: not all classes in train. Missing: {all_classes - train_classes}"
        )
        assert test_classes == all_classes, (
            f"image_split: not all classes in test. Missing: {all_classes - test_classes}"
        )
        print("✅ image_split: every class appears in both train and test")
        return True
    except Exception as e:
        print(f"❌ image_split class check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bat_split_mode(data_path):
    """Test bat_split mode with 6 verification checks."""
    print(f"\n🔍 Testing bat_split mode...")

    try:
        splitter = SiameseNetworkTrainingDataSplitter(
            images_dirs_paths_list=[data_path],
            training_portion=0.7,
            mode="combination",
            skip_preprocessing=True,
            split_mode="bat_split",
            split_seed=42,
        )

        train_classes = set(splitter.train_class_files.keys())
        test_classes = set(splitter.test_class_files.keys())
        all_classes = set(splitter.class_names)
        num_classes = len(all_classes)

        # 1. Disjoint classes
        overlap = train_classes & test_classes
        assert len(overlap) == 0, f"Classes overlap between train and test: {overlap}"
        print("  ✅ Check 1: Train and test classes are disjoint")

        # 2. Complete coverage
        assert train_classes | test_classes == all_classes, (
            f"Not all classes assigned. Missing: {all_classes - (train_classes | test_classes)}"
        )
        print("  ✅ Check 2: All classes assigned to train or test")

        # 3. No file leakage -- pair images reference only their split's classes
        if splitter.train_data is not None:
            for v1, v2, label, ci in splitter.train_data.take(20):
                p1 = v1.numpy().decode("utf-8")
                p2 = v2.numpy().decode("utf-8")
                fn1 = os.path.basename(p1)
                fn2 = os.path.basename(p2)
                parsed1 = parse_filename_class(fn1)
                parsed2 = parse_filename_class(fn2)
                assert parsed1 is not None and parsed2 is not None
                assert parsed1[1] in train_classes, f"Train pair has test-class file: {fn1}"
                assert parsed2[1] in train_classes, f"Train pair has test-class file: {fn2}"

        if splitter.test_data is not None:
            for v1, v2, label, ci in splitter.test_data.take(20):
                p1 = v1.numpy().decode("utf-8")
                p2 = v2.numpy().decode("utf-8")
                fn1 = os.path.basename(p1)
                fn2 = os.path.basename(p2)
                parsed1 = parse_filename_class(fn1)
                parsed2 = parse_filename_class(fn2)
                assert parsed1 is not None and parsed2 is not None
                assert parsed1[1] in test_classes, f"Test pair has train-class file: {fn1}"
                assert parsed2[1] in test_classes, f"Test pair has train-class file: {fn2}"
        print("  ✅ Check 3: No file leakage between train and test pairs")

        # 4. Correct split ratio
        expected_train = max(1, round(num_classes * 0.7))
        assert len(train_classes) == expected_train, (
            f"Expected {expected_train} train classes, got {len(train_classes)}"
        )
        print(f"  ✅ Check 4: Correct split ratio ({len(train_classes)} train, {len(test_classes)} test)")

        # 5. Seed determinism
        splitter2 = SiameseNetworkTrainingDataSplitter(
            images_dirs_paths_list=[data_path],
            training_portion=0.7,
            mode="combination",
            skip_preprocessing=True,
            split_mode="bat_split",
            split_seed=42,
        )
        assert set(splitter2.train_class_files.keys()) == train_classes, "Seed determinism failed"
        assert set(splitter2.test_class_files.keys()) == test_classes, "Seed determinism failed"
        print("  ✅ Check 5: Same seed produces identical splits")

        # 6. Datasets are non-empty
        train_size = splitter.train_data.cardinality().numpy() if splitter.train_data else 0
        test_size = splitter.test_data.cardinality().numpy() if splitter.test_data else 0
        assert train_size > 0, "Train dataset is empty"
        assert test_size > 0, "Test dataset is empty"
        print(f"  ✅ Check 6: Non-empty datasets (train={train_size}, test={test_size})")

        print("✅ bat_split: all 6 checks passed")
        return True
    except Exception as e:
        print(f"❌ bat_split test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    with tempfile.TemporaryDirectory() as td:
        data_path = _create_tiny_dataset(td)
        
        print("🧪 Testing SiameseNetworkTrainingDataSplitter")
        print("=" * 60)
        
        # Test 1: Filename parsing
        test_filename_parsing()
        
        # Test 2: File grouping
        test_file_grouping(data_path)
        
        # Test 3: Complete data splitter (combination mode)
        splitter = test_data_splitter(data_path, skip_preprocessing=False)
        
        # Test 4: Permutation mode
        permutation_splitter = test_permutation_mode(data_path)

        # Test: skip_preprocessing=True
        print("\n🔍 Testing with skip_preprocessing=True...")
        splitter_skip = test_data_splitter(data_path, skip_preprocessing=True)
        
        print("\n" + "=" * 60)
        print("🎯 Test Summary:")
        
        if splitter:
            print("✅ Combination mode: SUCCESS")
            if splitter.train_data is not None:
                print(f"   Training pairs: {splitter.train_data.cardinality().numpy()}")
            if splitter.test_data is not None:
                print(f"   Test pairs: {splitter.test_data.cardinality().numpy()}")
        else:
            print("❌ Combination mode: FAILED")
        
        if permutation_splitter:
            print("✅ Permutation mode: SUCCESS")
            if permutation_splitter.train_data is not None:
                print(f"   Training pairs: {permutation_splitter.train_data.cardinality().numpy()}")
            if permutation_splitter.test_data is not None:
                print(f"   Test pairs: {permutation_splitter.test_data.cardinality().numpy()}")
        else:
            print("❌ Permutation mode: FAILED")

    # Split mode tests (need 6+ classes for bat_split)
    print("\n" + "=" * 60)
    print("🧪 Testing split modes")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        data_path = _create_tiny_dataset(td, num_classes=6)

        img_split_ok = test_image_split_all_classes_in_both(data_path)
        bat_split_ok = test_bat_split_mode(data_path)

    print("\n" + "=" * 60)
    print("🎯 Split Mode Test Summary:")
    print(f"  image_split: {'✅ PASS' if img_split_ok else '❌ FAIL'}")
    print(f"  bat_split:   {'✅ PASS' if bat_split_ok else '❌ FAIL'}")


if __name__ == "__main__":
    main()
