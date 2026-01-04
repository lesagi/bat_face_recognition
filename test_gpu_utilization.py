#!/usr/bin/env python3
"""
Test script to verify GPU is actually being used for computation.
Run this with: conda activate frec && source .venv/bin/activate && python test_gpu_utilization.py
"""

import tensorflow as tf
import numpy as np
import time
import subprocess

print("=" * 80)
print("GPU Utilization Test")
print("=" * 80)

# Check GPU availability
print("\n1. Checking GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
print(f"   Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")

if not gpus:
    print("   ❌ No GPUs found!")
    sys.exit(1)

# Configure GPU memory growth
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print(f"   ✅ Configured {gpu.name}")

# Check TensorFlow GPU setup
print("\n2. TensorFlow GPU configuration...")
print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"   GPU devices: {tf.config.list_physical_devices('GPU')}")
print(f"   Logical devices: {tf.config.list_logical_devices()}")

# Create a simple model that will use GPU
print("\n3. Creating test model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(1000,)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Generate dummy data
print("\n4. Generating test data...")
batch_size = 32
x_train = np.random.random((batch_size, 1000)).astype('float32')
y_train = np.random.randint(2, size=(batch_size, 1)).astype('float32')

# Function to get GPU utilization
def get_gpu_util():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return "N/A"

print("\n5. Running computation test...")
print("   Monitoring GPU utilization during computation...")
print("   " + "-" * 70)

# Warm up
print("   Warming up GPU...")
_ = model.predict(x_train, verbose=0)

# Test with multiple batches
print("\n   Running 10 batches of computation...")
for i in range(10):
    # Get GPU util before
    util_before = get_gpu_util()
    
    # Run computation
    start = time.time()
    _ = model.predict(x_train, verbose=0)
    elapsed = time.time() - start
    
    # Get GPU util after
    util_after = get_gpu_util()
    
    print(f"   Batch {i+1:2d}: {elapsed*1000:.2f}ms | GPU util: {util_after}")

print("\n6. Running training step test...")
print("   " + "-" * 70)

# Test training step (more intensive)
for i in range(5):
    util_before = get_gpu_util()
    start = time.time()
    model.train_on_batch(x_train, y_train)
    elapsed = time.time() - start
    util_after = get_gpu_util()
    print(f"   Training step {i+1}: {elapsed*1000:.2f}ms | GPU util: {util_after}")

print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)
print("\n💡 Interpretation:")
print("   - If GPU utilization shows 0%: GPU is NOT being used (CPU fallback)")
print("   - If GPU utilization shows >50%: GPU IS being used for computation")
print("   - Check nvidia-smi in another terminal for real-time monitoring")
print("=" * 80)

