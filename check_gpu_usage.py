#!/usr/bin/env python3
"""
Script to check if GPU is actually being used during training.
Run this while training is active to monitor GPU utilization.
"""

import subprocess
import time
import sys

def get_gpu_utilization():
    """Get current GPU utilization from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return []

def get_gpu_processes():
    """Get processes using GPU."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        return []

def main():
    print("=" * 80)
    print("GPU Usage Monitor")
    print("=" * 80)
    print("\nPress Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Clear screen (optional, comment out if you want to see history)
            # print("\033[2J\033[H", end="")
            
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
            print("-" * 80)
            
            # GPU utilization
            gpu_info = get_gpu_utilization()
            if gpu_info:
                print("\n📊 GPU Status:")
                print(f"{'GPU':<5} {'Name':<20} {'Util %':<10} {'Mem Util %':<12} {'Mem Used':<12} {'Mem Total':<12} {'Power (W)':<12}")
                print("-" * 80)
                for line in gpu_info:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 7:
                        gpu_idx, name, util_gpu, util_mem, mem_used, mem_total, power = parts
                        print(f"{gpu_idx:<5} {name:<20} {util_gpu:<10} {util_mem:<12} {mem_used:<12} {mem_total:<12} {power:<12}")
                        
                        # Warning if utilization is low but memory is high
                        if int(util_gpu) < 5 and int(mem_used) > 1000:
                            print(f"    ⚠️  WARNING: GPU {gpu_idx} has high memory usage ({mem_used} MB) but low utilization ({util_gpu}%)")
                            print(f"       This suggests the model is loaded but not actively computing!")
            
            # GPU processes
            processes = get_gpu_processes()
            if processes:
                print("\n🔧 Processes Using GPU:")
                print(f"{'PID':<10} {'Process':<20} {'Memory (MB)':<15}")
                print("-" * 80)
                for line in processes:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        pid, proc, mem = parts
                        print(f"{pid:<10} {proc:<20} {mem:<15}")
            
            print("\n" + "=" * 80)
            print("💡 Interpretation:")
            print("   - GPU Utilization > 50%: GPU is actively computing")
            print("   - GPU Utilization < 5% with high memory: Model loaded but idle (possible bottleneck)")
            print("   - High memory usage: Model is on GPU")
            print("   - Low memory usage: Model might be on CPU")
            print("=" * 80)
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()

