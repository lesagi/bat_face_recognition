# Saliency Maps Model Loading Issue - Planning Document

## Issue Description

The saliency maps generation script (`run_saliency_maps.py`) is failing to load the trained Siamese model due to a **TensorFlow SavedModel format compatibility issue** with Keras 3.

**Error Message:**
```
File format not supported: filepath=/home/sagilevi1/bat_face_rec_project/output_siamese_20250406_014341/train/model/siamesemodelv2_v40. 
Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). 
Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. 
In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, 
use `keras.layers.TFSMLayer(/home/sagilevi1/bat_face_rec_project/output_siamese_20250406_014341/train/model/siamesemodelv2_v40, call_endpoint='serving_default')`
```

## Root Cause Analysis

### 1. Model Format Mismatch
- **Current Model**: SavedModel format (TensorFlow 2.x legacy format)
  - Contains: `saved_model.pb`, `keras_metadata.pb`, `variables/` directory
  - This is the standard format when saving models with `model.save()` in TensorFlow 2.x
- **Required Format**: Keras 3 compatible format
  - `.keras` files (new V3 format)
  - `.h5` files (legacy HDF5 format)

### 2. Keras Version Compatibility
- **Keras 3**: Dropped support for TensorFlow SavedModel format
- **TensorFlow 2.x**: Still supports SavedModel format
- **Migration Issue**: Models saved in TensorFlow 2.x SavedModel format cannot be loaded directly in Keras 3

### 3. Why This Happened
1. **Model Training**: The Siamese model was trained and saved using TensorFlow 2.x
2. **Environment Update**: The current environment uses Keras 3 (likely from a recent update)
3. **Format Incompatibility**: The old SavedModel format is no longer supported

## Solution: Keras Downgrade Approach

### Overview
Downgrade to Keras 2.15.0 which natively supports SavedModel format. This approach is faster to implement and provides immediate compatibility.

### Implementation Plan

#### Phase 1: Requirements Management and Git Preparation (Priority: HIGH)

##### 1.1 Update Requirements Files
```bash
# 1. Update main requirements.txt with current environment
pip freeze > requirements.txt

# 2. Update requirements-linux.txt based on latest pip freeze to requirements.txt
```

**Files to Update:**
- `requirements.txt` - Full current environment
- `requirements-linux.txt` - Full current environment compatible with linux

##### 1.2 Git Operations - Keras v3 Tag
```bash
# 1. Stage all requirements file changes
git add requirements*.txt

# 2. Commit with descriptive message
git commit -m "Update requirements: capture current Keras 3 environment state

- Update main requirements.txt with current pip freeze output
- Update requirements-linux.txt with minimum version constraints
- Prepare for Keras downgrade testing"

# 3. Tag the commit
git tag -a keras-v3 -m "Keras 3 environment baseline before downgrade testing"
```

#### Phase 2: Create Keras v2 Branch and Implement Downgrade (Priority: HIGH)

##### 2.1 Create and Switch to New Branch
```bash
# 1. Create new branch from current state
git checkout -b keras-v2-saliency

# 2. Verify branch creation
git branch
# Should show: * keras-v2-saliency
```

##### 2.2 Update All Requirements Files for Keras 2
```bash
# 1. Update main requirements.txt
# Change any keras>=3.0.0 to keras==2.15.0
# Ensure tensorflow compatibility

# 2. Update requirements-linux.txt
# Change keras>=3.0.0 to keras==2.15.0
# Update tensorflow to compatible version (2.15.x)
```

**Specific Changes:**
```diff
# requirements.txt
- keras>=3.0.0
+ keras==2.15.0

# requirements-linux.txt  
- keras>=3.0.0
+ keras==2.15.0
- tensorflow>=2.15.0
+ tensorflow==2.15.1
```

##### 2.3 Update Relevant Code (if needed)
**Files to Check for Compatibility:**
- `app/visualization/run_saliency_maps.py` - Main saliency script
- `app/siamese_network/trainer.py` - Uses tf.keras.metrics
- `app/siamese_network/network.py` - Uses tf.keras.models
- `app/siamese_network/utils.py` - Uses tf.keras.models.load_model
- `app/generate_predictions.py` - Uses tf.keras.metrics

**Expected Changes:**
- Most files should work without changes (they use `tensorflow.keras`)
- May need to update import statements if any use standalone `keras`
- Verify `tf.keras.models.load_model()` works with SavedModel format

##### 2.4 Environment Downgrade
```bash
# 1. Activate conda environment
conda activate bat_face_rec

# 2. Uninstall current Keras
pip uninstall keras -y

# 3. Install Keras 2.15.0
pip install keras==2.15.0

# 4. Verify installation
python -c "import keras; print(f'Keras version: {keras.__version__}')"
python -c "import tensorflow.keras; print('TensorFlow Keras works')"
```

#### Phase 3: Sanity Testing (Priority: HIGH)

##### 3.1 Basic Import Tests
```bash
# Test core imports
python -c "
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
print('✅ All imports successful')
print(f'TensorFlow: {tf.__version__}')
print(f'Keras: {keras.__version__}')
"
```

##### 3.2 Model Loading Test
```bash
# Test SavedModel loading (don't save output)
python -c "
import tensorflow as tf
from tensorflow.keras.models import load_model
from siamese_network.network import L1Dist
from config.loader import load_config

# Get model path from config
config = load_config()
model_path = config.models.siamese.model_path

try:
    model = load_model(model_path, 
                      custom_objects={'L1Dist': L1Dist})
    print('✅ Model loaded successfully')
    print(f'Model type: {type(model)}')
    print(f'Input shape: {model.input_shape}')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"
```

##### 3.3 Saliency Script Test
```bash
# Test saliency script basic functionality (don't save output)
python app/visualization/run_saliency_maps.py \
    --input_dir /path/to/test/data \
    --model_path /path/to/siamesemodelv2_v40 \
    --sample_size 2 \
    --method standard \
    --output_dir /tmp/test_saliency
```

**Expected Results:**
- ✅ Model loads without SavedModel format errors
- ✅ Script runs without import errors
- ✅ Basic saliency map generation works
- ✅ No output files saved (test only)

#### Phase 4: Git Operations - Keras v2 Tag (Priority: HIGH)

##### 4.1 Commit Keras v2 Changes
```bash
# 1. Stage all changes
git add requirements*.txt
git add app/  # Any code changes made

# 2. Commit with descriptive message
git commit -m "Downgrade to Keras 2.15.0 for SavedModel compatibility

- Update all requirements files to use Keras 2.15.0
- Ensure TensorFlow 2.15.1 compatibility
- Test SavedModel loading functionality
- Verify saliency maps script works with Keras 2
- Resolves SavedModel format compatibility issues"

# 3. Tag the commit
git tag -a saliency-keras-v2 -m "Keras 2.15.0 environment for SavedModel compatibility"
```

##### 4.2 Verify Tags
```bash
# List all tags
git tag -l

# Should show:
# keras-v3
# saliency-keras-v2

# Show tag details
git show keras-v3
git show saliency-keras-v2
```

#### Phase 5: Validation and Cleanup (Priority: MEDIUM)

##### 5.1 Final Validation
```bash
# 1. Verify Keras 2 environment
python -c "import keras; assert keras.__version__.startswith('2.15'), 'Wrong Keras version'"

# 2. Test model loading one more time
python -c "
from tensorflow.keras.models import load_model
from siamese_network.network import L1Dist
from config.loader import load_config

config = load_config()
model_path = config.models.siamese.model_path

model = load_model(model_path, 
                  custom_objects={'L1Dist': L1Dist})
print('✅ Final validation: Model loads successfully')
"

# 3. Test saliency script with small sample
python app/visualization/run_saliency_maps.py \
    --input_dir /path/to/test/data \
    --model_path /path/to/siamesemodelv2_v40 \
    --sample_size 1 \
    --method standard \
    --output_dir /tmp/final_test
```

##### 5.2 Cleanup Test Files
```bash
# Remove any test output files
rm -rf /tmp/test_saliency
rm -rf /tmp/final_test
```

### Detailed File Changes

#### 1. `requirements.txt`
```diff
# Current: pip freeze output
# Add: keras==2.15.0
+ keras==2.15.0
```

#### 2. `requirements-linux.txt` (if exists)
```diff
- keras>=3.0.0
+ keras==2.15.0
- tensorflow>=2.15.0
+ tensorflow==2.15.1
```

### Rollback Plan

If issues arise during downgrade:

```bash
# 1. Switch back to main branch
git checkout main

# 2. Try a different approach (TFSMLayer implementation)
# This will be implemented as an alternative solution
```

### Success Criteria for Keras v2 Approach

1. ✅ **Requirements Updated**: All requirements files use Keras 2.15.0
2. ✅ **Environment Downgraded**: Keras 2.15.0 successfully installed
3. ✅ **Model Loading**: SavedModel loads without format errors
4. ✅ **Saliency Script**: Runs without import or compatibility issues
5. ✅ **Git Operations**: Both tags created successfully
6. ✅ **No Output Files**: Tests run without saving outputs
7. ✅ **Clean State**: Test files removed, environment stable

### Timeline for Keras v2 Approach

- **Phase 1 (Requirements & Git)**: 1-2 hours
- **Phase 2 (Downgrade & Code)**: 2-3 hours
- **Phase 3 (Testing)**: 1-2 hours
- **Phase 4 (Git Tagging)**: 30 minutes
- **Phase 5 (Validation)**: 1 hour

**Total Estimated Time**: 5-8 hours

### Next Action Items

1. **Review this plan** for accuracy and completeness
2. **Approve implementation** of Keras v2 approach
3. **Begin Phase 1** (Requirements management and git preparation)
4. **Execute phases sequentially** with validation at each step
5. **Document results** for future reference
