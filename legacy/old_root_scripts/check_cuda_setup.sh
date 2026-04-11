#!/bin/bash

OUTPUT_FILE="cuda_setup_info.txt"

echo "Starting CUDA setup check..."
echo "Output will be written to: $OUTPUT_FILE"
echo ""

echo "=== CUDA Setup Information ===" > $OUTPUT_FILE
echo "Generated on: $(date)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "[1/20] Checking CUDA compiler version..."
echo "=== 1. CUDA Compiler Version ===" >> $OUTPUT_FILE
nvcc --version >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "[2/20] Checking NVIDIA driver and CUDA version..."
echo "=== 2. NVIDIA Driver and CUDA Version ===" >> $OUTPUT_FILE
nvidia-smi >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "[3/20] Checking CUDA installations..."
echo "=== 3. CUDA Installations in /usr/local/ ===" >> $OUTPUT_FILE
ls -la /usr/local/ | grep cuda >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "[4/20] Checking CUDA libraries in /usr/local/cuda/lib64/..."
echo "=== 4. CUDA Libraries in /usr/local/cuda/lib64/ ===" >> $OUTPUT_FILE
ls -la /usr/local/cuda/lib64/ 2>/dev/null | grep -E "(libcudart|libcublas|libcudnn)" >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "[5/20] Checking system-wide CUDA libraries..."
echo "=== 5. System-wide CUDA Libraries ===" >> $OUTPUT_FILE
ldconfig -p | grep -E "(libcudart|libcublas|libcudnn)" >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "[6/20] Checking CUDA environment variables..."
echo "=== 6. CUDA Environment Variables ===" >> $OUTPUT_FILE
echo "CUDA_HOME: $CUDA_HOME" >> $OUTPUT_FILE
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" >> $OUTPUT_FILE
echo "PATH: $PATH" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "[7/20] Finding CUDA libraries in common locations (with 60s timeout)..."
echo "=== 7. Finding CUDA Libraries ===" >> $OUTPUT_FILE
echo "--- libcudart.so ---" >> $OUTPUT_FILE
timeout 60 find /usr/local /usr/lib /usr/lib64 -name "libcudart.so*" 2>/dev/null >> $OUTPUT_FILE || echo "Search timed out or interrupted" >> $OUTPUT_FILE
echo "--- libcublas.so ---" >> $OUTPUT_FILE
timeout 60 find /usr/local /usr/lib /usr/lib64 -name "libcublas.so*" 2>/dev/null >> $OUTPUT_FILE || echo "Search timed out or interrupted" >> $OUTPUT_FILE
echo "--- libcudnn.so ---" >> $OUTPUT_FILE
timeout 60 find /usr/local /usr/lib /usr/lib64 -name "libcudnn.so*" 2>/dev/null >> $OUTPUT_FILE || echo "Search timed out or interrupted" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "[8/20] Checking cuDNN version..."
echo "=== 8. cuDNN Version ===" >> $OUTPUT_FILE
echo "--- From /usr/local/cuda/include/cudnn_version.h ---" >> $OUTPUT_FILE
cat /usr/local/cuda/include/cudnn_version.h 2>/dev/null | grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" >> $OUTPUT_FILE 2>&1
echo "--- From /usr/include/cudnn_version.h ---" >> $OUTPUT_FILE
cat /usr/include/cudnn_version.h 2>/dev/null | grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "[9/20] Checking Python environment..."
echo "=== 9. Python Environment ===" >> $OUTPUT_FILE
echo "Python location: $(which python)" >> $OUTPUT_FILE
echo "Python version: $(python --version)" >> $OUTPUT_FILE
echo "--- Conda packages (tensorflow, cuda, cudnn) ---" >> $OUTPUT_FILE
conda list | grep -E "(tensorflow|cuda|cudnn)" >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "[10/20] Testing TensorFlow CUDA detection..."
echo "=== 10. TensorFlow CUDA Detection ===" >> $OUTPUT_FILE
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Built with CUDA:', tf.test.is_built_with_cuda()); print('GPU devices:', tf.config.list_physical_devices('GPU'))" >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "[11/20] Checking CUDA symlinks..."
echo "=== 11. All CUDA-related Symlinks ===" >> $OUTPUT_FILE
ls -la /usr/local/cuda* 2>/dev/null >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "[12/20] Checking CUDA toolkit packages..."
echo "=== 12. CUDA Toolkit Files ===" >> $OUTPUT_FILE
dpkg -l | grep -i cuda >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "[13/20] Checking NVIDIA driver packages..."
echo "=== 13. NVIDIA Driver Packages ===" >> $OUTPUT_FILE
dpkg -l | grep -i nvidia >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "[14/20] Checking for cuDNN libraries..."
echo "=== 14. Check for cuDNN Libraries ===" >> $OUTPUT_FILE
echo "--- In /usr/local/cuda-12.4/lib64/ ---" >> $OUTPUT_FILE
ls -la /usr/local/cuda-12.4/lib64/ | grep cudnn >> $OUTPUT_FILE 2>&1
echo "--- System-wide cuDNN (with 60s timeout) ---" >> $OUTPUT_FILE
timeout 60 find /usr/local /usr/lib /usr/lib64 -name "libcudnn.so*" 2>/dev/null >> $OUTPUT_FILE || echo "Search timed out or interrupted" >> $OUTPUT_FILE
echo "--- In conda environment ---" >> $OUTPUT_FILE
ls -la $CONDA_PREFIX/lib/ 2>/dev/null | grep cudnn >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "[15/20] Checking conda environment libraries..."
echo "=== 15. Check Conda Environment Libraries ===" >> $OUTPUT_FILE
echo "CONDA_PREFIX: $CONDA_PREFIX" >> $OUTPUT_FILE
ls -la $CONDA_PREFIX/lib/ 2>/dev/null | grep -E "(libcudart|libcublas|libcudnn)" >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "[16/20] Verifying LD_LIBRARY_PATH components..."
echo "=== 16. Verify LD_LIBRARY_PATH Components ===" >> $OUTPUT_FILE
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" >> $OUTPUT_FILE
echo "--- Checking each path in LD_LIBRARY_PATH ---" >> $OUTPUT_FILE
IFS=':' read -ra PATHS <<< "$LD_LIBRARY_PATH"
for path in "${PATHS[@]}"; do
    echo "Path: $path" >> $OUTPUT_FILE
    ls -la "$path" 2>/dev/null | grep -E "(libcudart|libcublas|libcudnn)" >> $OUTPUT_FILE 2>&1
done
echo "" >> $OUTPUT_FILE

echo "[17/20] Searching Python site packages for CUDA libraries (with 60s timeout)..."
echo "=== 17. Python Site Packages CUDA Libraries ===" >> $OUTPUT_FILE
python -c "import site; print('\n'.join(site.getsitepackages()))" > /tmp/site_packages.txt 2>&1
while IFS= read -r site_pkg; do
    echo "--- $site_pkg ---" >> $OUTPUT_FILE
    timeout 60 find "$site_pkg" -name "*cuda*" -o -name "*cudnn*" 2>/dev/null | head -20 >> $OUTPUT_FILE || echo "Search timed out or interrupted" >> $OUTPUT_FILE
done < /tmp/site_packages.txt
rm /tmp/site_packages.txt
echo "" >> $OUTPUT_FILE

echo "[18/20] Checking TensorFlow version and dependencies..."
echo "=== 18. TensorFlow Version and Dependencies ===" >> $OUTPUT_FILE
pip show tensorflow >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "[19/20] Running detailed TensorFlow GPU check..."
echo "=== 19. Detailed TensorFlow GPU Check ===" >> $OUTPUT_FILE
python -c "
import tensorflow as tf
import sys
print('Python version:', sys.version)
print('TensorFlow version:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
print('GPU devices:', tf.config.list_physical_devices('GPU'))
print('Available devices:', tf.config.list_physical_devices())
" >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

echo "[20/20] Checking for missing CUDA libraries..."
echo "=== 20. Check for Missing CUDA Libraries ===" >> $OUTPUT_FILE
echo "Checking for required libraries..." >> $OUTPUT_FILE
for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcufft.so.11 libcurand.so.10 libcusolver.so.11 libcusparse.so.12 libcudnn.so.8; do
    echo "--- $lib ---" >> $OUTPUT_FILE
    ldconfig -p | grep "$lib" >> $OUTPUT_FILE 2>&1
    if [ $? -ne 0 ]; then
        echo "MISSING: $lib not found in ldconfig cache" >> $OUTPUT_FILE
    fi
done
echo "" >> $OUTPUT_FILE

echo "=== END OF CUDA SETUP INFORMATION ===" >> $OUTPUT_FILE

echo ""
echo "✅ Done! CUDA setup information has been written to: $OUTPUT_FILE"
echo "You can now share this file."

