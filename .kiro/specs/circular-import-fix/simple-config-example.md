# Simple Configuration Class Example

## Current Problematic Implementation

### File: `config/model_config.py` (CURRENT - PROBLEMATIC)
```python
from typing import Union, Literal
from ultralytics import YOLO  # ❌ HEAVY IMPORT - CAUSES THE ISSUE

class ImageModelConfig:
    def __init__(
        self,
        model: Union[str, YOLO],  # ❌ Accepts model instances
        model_type: Literal["yolo_segmentation", "yolo_pose"],
        confidence_threshold: float = 0.3
    ):
        self.model = model  # ❌ Stores model instances
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
```

**Problems:**
- Imports heavy `ultralytics` library at module level
- Accepts and stores model instances
- Mixes configuration data with model objects
- Causes import chain bottleneck

## Proposed Simple Implementation

### File: `config/model_config.py` (PROPOSED - SIMPLE)
```python
from typing import Literal

class ImageModelConfig:
    """Simple data holder for model configuration from config.yml"""
    
    def __init__(
        self,
        model_path: str,
        model_type: Literal["yolo_segmentation", "yolo_pose"],
        confidence_threshold: float = 0.3
    ):
        self.model_path = model_path              # ✅ String path only
        self.model_type = model_type              # ✅ String type only  
        self.confidence_threshold = confidence_threshold  # ✅ Number only
```

**Benefits:**
- No heavy imports
- Pure data representation
- Fast to import and instantiate
- Matches config.yml structure exactly

## Configuration Data Mapping

### config.yml Structure:
```yaml
models:
  segmentation:
    model_path: "/path/to/model.pt"
    model_type: "yolo_segmentation"
    confidence_threshold: 0.5
```

### Configuration Class Properties:
```python
config = load_config()
seg_config = config.models.segmentation

# These should be simple data access:
seg_config.model_path           # → "/path/to/model.pt" (string)
seg_config.model_type           # → "yolo_segmentation" (string)
seg_config.confidence_threshold # → 0.5 (float)
```

## Model Creation Pattern

### Separate Model Factory (NEW)
```python
# File: models/model_factory.py
from typing import Union
from config.model_config import ImageModelConfig

def create_yolo_model(config: ImageModelConfig):
    """Create YOLO model from configuration - heavy import happens here"""
    from ultralytics import YOLO  # ✅ Import only when actually creating model
    
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model file not found: {config.model_path}")
    
    return YOLO(config.model_path)

# Usage:
config = load_config()
seg_config = config.models.segmentation  # ✅ Fast - no model loading
model = create_yolo_model(seg_config)    # ✅ Heavy work happens here, when needed
```

## Updated Usage Patterns

### Before (Problematic):
```python
from config import ImageModelConfig
from ultralytics import YOLO

# This would trigger heavy imports during config loading
model_config = ImageModelConfig(
    model=YOLO("/path/to/model.pt"),  # Heavy operation during config
    model_type="yolo_segmentation"
)
```

### After (Simple):
```python
from config.loader import load_config
from models.model_factory import create_yolo_model

# Fast configuration loading
config = load_config()
seg_config = config.models.segmentation  # Just data, no heavy operations

# Model creation only when needed
if need_to_use_model:
    model = create_yolo_model(seg_config)  # Heavy operation happens here
    results = model(image)
```

## All Configuration Classes Should Follow This Pattern

### Example: SiameseNetworkConfig
```python
class SiameseNetworkConfig:
    def __init__(self, config_data: dict):
        # Only data from config.yml
        self.input_size = config_data.get('input_size', 224)
        self.learning_rate = config_data.get('learning_rate', 0.001)
        self.batch_size = config_data.get('batch_size', 32)
        # No model instantiation, no heavy imports
```

### Example: YOLOSegmentationConfig  
```python
class YOLOSegmentationConfig:
    def __init__(self, config_data: dict):
        # Only data from config.yml
        self.default_epochs = config_data.get('default_epochs', 80)
        self.default_batch_size = config_data.get('default_batch_size', 16)
        self.default_confidence = config_data.get('default_confidence', 0.3)
        # No YOLO imports, no model creation
```

## Key Principles

1. **Configuration classes = Data holders only**
2. **No heavy imports in config modules**
3. **Model creation = Separate factory functions**
4. **Import heavy libraries only when actually using them**
5. **Configuration loading should be instant**

This approach completely eliminates the import chain bottleneck while keeping the code clean and maintainable.