#!/bin/bash
echo "=== Environment Check ==="
echo "Active Python: $(which python)"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""
echo "=== Checking cuDNN in conda environment ==="
if [ -n "$CONDA_PREFIX" ]; then
    ls -la $CONDA_PREFIX/lib/libcudnn* 2>/dev/null || echo "No cuDNN found in conda environment"
else
    echo "No conda environment active"
fi
echo ""
echo "=== Checking cuDNN in system ==="
ldconfig -p 2>/dev/null | grep libcudnn || echo "No cuDNN found in system linker cache"
echo ""
echo "=== TensorFlow GPU Test ==="
python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPU devices:', tf.config.list_physical_devices('GPU'))" 2>&1 | tail -n 5
