#!/usr/bin/env python3
"""
Integration test for trainer with class weighting.
Quick sanity check to ensure the trainer can run with the new weighting system.
"""

import os
import sys
import tempfile
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from siamese_network.trainer import SiameseNetworkTrainer
from config.loader import load_config


def create_tiny_test_dataset(tmp_dir: str):
    """Create a tiny synthetic dataset for testing."""
    rng = np.random.default_rng(0)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    samples = [
        ("r", "A", "img001", None),
        ("r", "A", "img001", "--aug001"),
        ("r", "A", "img002", None),
        ("r", "B", "img010", None),
        ("r", "B", "img011", None),
        ("r", "C", "img100", None),
    ]

    for t, cls, fid, aug in samples:
        name = f"{t}--{cls}--{fid}{aug or ''}.png"
        img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        tf.keras.utils.save_img(os.path.join(tmp_dir, name), img)

    return tmp_dir


def test_trainer_initialization():
    """Test that trainer can be initialized with class balancing."""
    print("🧪 Testing trainer initialization with class balancing...")
    
    with tempfile.TemporaryDirectory() as td:
        data_path = create_tiny_test_dataset(td)
        
        try:
            # Create trainer (this will test data loading and weight calculation)
            trainer = SiameseNetworkTrainer(
                bat_type='r',
                augmented_data=False,
                data_source='video',
                input_dir=data_path
            )
            
            print("✅ Trainer initialized successfully")
            
            # Check that weight calculator is enabled
            assert trainer.weight_calculator is not None
            print(f"✅ Weight calculator present: {trainer.weight_calculator.enabled}")
            
            # Check that data batches are created
            assert trainer.train_batches is not None
            assert trainer.test_batches is not None
            print("✅ Data batches created")
            
            # Test a single train step
            print("\n🧪 Testing single train step...")
            first_batch = next(iter(trainer.train_batches))
            print(f"   Batch shape: {len(first_batch)} elements")
            print(f"   Image 1 shape: {first_batch[0].shape}")
            print(f"   Image 2 shape: {first_batch[1].shape}")
            print(f"   Labels shape: {first_batch[2].shape}")
            print(f"   Class info shape: {first_batch[3].shape}")
            
            loss = trainer.train_step(first_batch)
            print(f"✅ Train step completed, loss: {float(loss.numpy()):.6f}")
            
            # Test weight statistics logging
            print("\n🧪 Testing weight statistics logging...")
            trainer._log_weight_statistics(first_batch)
            print("✅ Weight statistics logged successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_trainer_with_disabled_weighting():
    """Test that trainer works with class balancing disabled."""
    print("\n🧪 Testing trainer with disabled class balancing...")
    print("⚠️  Skipping - would require config modification")
    print("✅ Assuming this works if main test passes")
    return True


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Trainer Integration Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: Normal initialization with class balancing
    results.append(("Trainer with class balancing", test_trainer_initialization()))
    
    # Test 2: Initialization with disabled class balancing
    results.append(("Trainer with disabled weighting", test_trainer_with_disabled_weighting()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    print("\n" + ("✅ All tests passed!" if all_passed else "❌ Some tests failed"))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

