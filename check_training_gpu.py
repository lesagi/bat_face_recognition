#!/usr/bin/env python3
"""
Check if the currently running training process is using GPU.
Run with: conda activate frec && source .venv/bin/activate && python check_training_gpu.py
"""

import tensorflow as tf
import subprocess
import sys

print("=" * 80)
print("Training GPU Usage Check")
print("=" * 80)

# Check GPU availability
print("\n1. TensorFlow GPU Status:")
gpus = tf.config.list_physical_devices('GPU')
print(f"   Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")

# Check current GPU processes
print("\n2. Current GPU Processes:")
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True,
        check=True
    )
    processes = result.stdout.strip().split('\n') if result.stdout.strip() else []
    if processes:
        print(f"   {'PID':<10} {'Process':<20} {'Memory (MB)':<15}")
        print("   " + "-" * 45)
        for line in processes:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                pid, proc, mem = parts
                print(f"   {pid:<10} {proc:<20} {mem:<15}")
    else:
        print("   No processes currently using GPU")
except Exception as e:
    print(f"   Error: {e}")

# Check GPU utilization
print("\n3. Current GPU Utilization:")
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True,
        check=True
    )
    gpu_info = result.stdout.strip().split('\n')
    print(f"   {'GPU':<5} {'Name':<20} {'Util %':<10} {'Mem Util %':<12} {'Mem Used (MB)':<15} {'Mem Total (MB)':<15}")
    print("   " + "-" * 80)
    for line in gpu_info:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 6:
            gpu_idx, name, util_gpu, util_mem, mem_used, mem_total = parts
            print(f"   {gpu_idx:<5} {name:<20} {util_gpu:<10} {util_mem:<12} {mem_used:<15} {mem_total:<15}")
            
            # Warning
            if int(util_gpu) < 5 and int(mem_used) > 1000:
                print(f"      ⚠️  WARNING: High memory ({mem_used} MB) but low utilization ({util_gpu}%)")
                print(f"         This suggests GPU is loaded but not actively computing!")
except Exception as e:
    print(f"   Error: {e}")

# Test device placement
print("\n4. Testing Device Placement:")
print("   Creating a simple operation to verify GPU usage...")

# Create a simple computation
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    result = c.numpy()

print(f"   ✅ Computation completed: {result}")
print(f"   Device placement test passed")

# Check if operations are actually on GPU
print("\n5. Verifying TensorFlow Operations:")
try:
    # Create a test operation
    with tf.device('/GPU:0'):
        x = tf.random.normal((1000, 1000))
        y = tf.random.normal((1000, 1000))
        z = tf.matmul(x, y)
    
    # Check where the operation executed
    print(f"   Operation shape: {z.shape}")
    print(f"   Operation device: {z.device}")
    print(f"   Operation dtype: {z.dtype}")
    
    if '/GPU' in str(z.device):
        print("   ✅ Operations are being placed on GPU")
    else:
        print("   ⚠️  Operations are NOT on GPU (check device placement)")
        
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 80)
print("Recommendations:")
print("=" * 80)
print("1. If GPU utilization is 0% but memory is high:")
print("   - Check if data loading is the bottleneck")
print("   - Verify operations are not falling back to CPU")
print("   - Check for .numpy() calls that force CPU execution")
print("\n2. To monitor GPU in real-time:")
print("   watch -n 1 nvidia-smi")
print("\n3. To check if training is using GPU:")
print("   Run this script while training is active")
print("=" * 80)

