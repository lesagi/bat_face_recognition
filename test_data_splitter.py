#!/usr/bin/env python3
"""
Test script for SiameseNetworkTrainingDataSplitter
"""

import os
import sys
import tensorflow as tf

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from siamese_network.data_splitter import SiameseNetworkTrainingDataSplitter, parse_filename_class, group_files_by_class

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
        for class_name, files in class_files.items():
            print(f"  Class '{class_name}': {len(files)} files")
            if files:
                sample_filename = os.path.basename(files[0])
                print(f"    Sample: {sample_filename}")

def test_data_splitter(data_path):
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
            mode="combination"
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
            
            for i, (img1, img2, label) in enumerate(sample_batch):
                print(f"  Sample {i+1}:")
                print(f"    Image 1 shape: {img1.shape}, dtype: {img1.dtype}")
                print(f"    Image 2 shape: {img2.shape}, dtype: {img2.dtype}")
                print(f"    Label: {label.numpy()}")
                print(f"    Value range img1: [{img1.numpy().min():.3f}, {img1.numpy().max():.3f}]")
                print(f"    Value range img2: [{img2.numpy().min():.3f}, {img2.numpy().max():.3f}]")
        
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

def main():
    """Main test function."""
    data_path = "/Users/MAC/Documents/bat_face_rec/data/processed/rous_siamese_input/augmented_cropped_picsum"
    
    print("🧪 Testing SiameseNetworkTrainingDataSplitter")
    print("=" * 60)
    
    # Test 1: Filename parsing
    test_filename_parsing()
    
    # Test 2: File grouping
    test_file_grouping(data_path)
    
    # Test 3: Complete data splitter (combination mode)
    splitter = test_data_splitter(data_path)
    
    # Test 4: Permutation mode
    permutation_splitter = test_permutation_mode(data_path)
    
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

if __name__ == "__main__":
    main()
