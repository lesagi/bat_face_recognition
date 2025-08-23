# Config Class Refactoring - Design Document

## Overview

This design document outlines the refactoring of configuration classes from mixed data/model holders to pure data representations that mirror the `config.yml` structure. The design eliminates heavy imports from configuration modules and introduces a separate model factory pattern for creating model instances when needed.

## Architecture

### Current Architecture (Problematic)
```
config.yml → ConfigLoader → Config Classes (with heavy imports) → Model Instances
                                    ↓
                            Heavy imports during config loading
                            (ultralytics, torch, etc.)
```

### New Architecture (Clean Separation)
```
config.yml → ConfigLoader → Pure Data Config Classes (no heavy imports)
                                    ↓
                            Model Factory Functions → Model Instances
                                    ↓
                            Heavy imports only when creating models
```

## Components and Interfaces

### 1. Pure Data Configuration Classes

#### 1.1 ImageModelConfig (Refactored)
```python
# File: app/config/model_config.py
from typing import Literal

class ImageModelConfig:
    """Pure data holder for model configuration from config.yml"""
    
    def __init__(
        self,
        model_path: str,
        model_type: Literal["yolo_segmentation", "yolo_pose"],
        confidence_threshold: float
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
    
    def __repr__(self) -> str:
        return f"ImageModelConfig(model_type='{self.model_type}', model_path='{self.model_path}', confidence_threshold={self.confidence_threshold})"
```

**Key Changes:**
- No `ultralytics` import
- No model instance storage
- Only simple data types
- Constructor takes individual parameters, not model instances

#### 1.2 ModelsConfig (Updated)
```python
# File: app/config/models.py
from typing import Any, Dict, Optional
from .model_config import ImageModelConfig

class ModelsConfig:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        
        # Create ImageModelConfig instances from data
        self._segmentation = self._create_model_config("segmentation")
        self._pose = self._create_model_config("pose")
        self._siamese = self._create_model_config("siamese")
    
    def _create_model_config(self, model_name: str) -> Optional[ImageModelConfig]:
        """Create ImageModelConfig from config data"""
        model_data = self._config.get(model_name, {})
        if not model_data:
            return None
            
        return ImageModelConfig(
            model_path=model_data.get("model_path", ""),
            model_type=model_data.get("model_type", ""),
            confidence_threshold=model_data.get("confidence_threshold", 0.3)
        )
    
    @property
    def segmentation(self) -> Optional[ImageModelConfig]:
        return self._segmentation
    
    @property
    def pose(self) -> Optional[ImageModelConfig]:
        return self._pose
    
    @property
    def siamese(self) -> Optional[ImageModelConfig]:
        return self._siamese
```

**Key Changes:**
- Returns `ImageModelConfig` instances instead of raw dictionaries
- No heavy imports
- Validates and structures data from config.yml

### 2. Model Factory Pattern

#### 2.1 Model Factory Interface
```python
# File: app/models/__init__.py
from typing import Protocol, Any
from config.model_config import ImageModelConfig

class ModelFactory(Protocol):
    """Protocol for model factory functions"""
    
    def create_model(self, config: ImageModelConfig) -> Any:
        """Create a model instance from configuration"""
        ...
```

#### 2.2 YOLO Model Factory
```python
# File: app/models/yolo_factory.py
import os
from typing import Any, Optional
from config.model_config import ImageModelConfig

def create_yolo_model(config: ImageModelConfig) -> Any:
    """
    Create YOLO model from configuration.
    Heavy imports happen here, not during config loading.
    """
    # Validate configuration
    if not config.model_path:
        raise ValueError("Model path is required")
    
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model file not found: {config.model_path}")
    
    # Heavy import only when actually creating model
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(f"ultralytics not available: {e}")
    
    # Create and return model
    model = YOLO(config.model_path)
    return model

def create_segmentation_model(config: ImageModelConfig) -> Any:
    """Create segmentation model with validation"""
    if config.model_type != "yolo_segmentation":
        raise ValueError(f"Expected yolo_segmentation, got {config.model_type}")
    
    return create_yolo_model(config)

def create_pose_model(config: ImageModelConfig) -> Any:
    """Create pose model with validation"""
    if config.model_type != "yolo_pose":
        raise ValueError(f"Expected yolo_pose, got {config.model_type}")
    
    return create_yolo_model(config)
```



### 3. Updated Image Processing Integration

#### 3.1 ImageTransforms Updates
```python
# File: app/image_processor/transforms.py (relevant methods)
class ImageTransforms:
    @staticmethod
    def _get_model_from_config(model_type: str):
        """Get model configuration and create model instance"""
        try:
            from config.loader import load_config
            from models.yolo_factory import create_yolo_model
            
            config = load_config()
            
            if model_type == "segmentation":
                model_config = config.models.segmentation
            elif model_type == "pose":
                model_config = config.models.pose
            else:
                raise ValueError(f"Invalid model type: {model_type}")
            
            if not model_config:
                print(f"⚠️  No {model_type} model configuration found")
                return None
            
            # Create model using factory (heavy import happens here)
            model = create_yolo_model(model_config)
            return model
            
        except Exception as e:
            print(f"⚠️  Error creating model: {e}")
            return None
    
    @staticmethod
    def segment_image(original_image, debug=False, debug_dir=None, base_filename=None):
        """Segment image using YOLO model"""
        # Get model instance (heavy work happens here, not during import)
        model = ImageTransforms._get_model_from_config("segmentation")
        if model is None:
            return None
        
        # Use model for segmentation
        try:
            from config.loader import load_config
            config = load_config()
            confidence_threshold = config.models.segmentation.confidence_threshold
            
            results = model(original_image, conf=confidence_threshold)
            # ... rest of segmentation logic
            
        except Exception as e:
            print(f"⚠️  Segmentation failed: {e}")
            return None
```

## Data Models

### Configuration Data Flow
```
config.yml
    ↓
ConfigLoader._load_config() → Dict[str, Any]
    ↓
ModelsConfig(config_dict) → Creates ImageModelConfig instances
    ↓
ImageModelConfig(model_path, model_type, confidence_threshold)
    ↓
Pure data access: config.models.segmentation.model_path
```

### Model Creation Flow
```
ImageModelConfig (data only)
    ↓
create_yolo_model(config) → Heavy imports happen here
    ↓
YOLO model instance
    ↓
Model operations (predict, train, etc.)
```

## Error Handling

### Configuration Loading Errors
```python
class ConfigurationError(Exception):
    """Base exception for configuration errors"""
    pass

class ModelConfigurationError(ConfigurationError):
    """Exception for model configuration issues"""
    pass

class ModelCreationError(Exception):
    """Exception for model creation issues"""
    pass
```

### Error Handling Strategy
1. **Configuration Loading**: Fail fast with clear error messages
2. **Model Creation**: Graceful degradation with informative errors
3. **Model Operations**: Handle missing models gracefully

### Example Error Handling
```python
def create_yolo_model(config: ImageModelConfig) -> Any:
    try:
        # Validate configuration
        if not config.model_path:
            raise ModelConfigurationError("Model path is required")
        
        if not os.path.exists(config.model_path):
            raise ModelConfigurationError(f"Model file not found: {config.model_path}")
        
        # Import heavy dependency
        from ultralytics import YOLO
        
        # Create model
        model = YOLO(config.model_path)
        return model
        
    except ImportError as e:
        raise ModelCreationError(f"ultralytics not available: {e}")
    except Exception as e:
        raise ModelCreationError(f"Failed to create YOLO model: {e}")
```

## Testing Strategy

The testing approach will focus on validating that:
1. Configuration classes load quickly without heavy imports
2. Model factory functions work correctly when called
3. Existing functionality continues to work with the new architecture

Detailed test implementations will be created during the implementation phase based on the actual code structure.

## Migration Strategy

### Phase 1: Update Configuration Classes
1. Remove heavy imports from `config/model_config.py`
2. Update `ImageModelConfig` to be pure data
3. Update `ModelsConfig` to create `ImageModelConfig` instances

### Phase 2: Create Model Factory
1. Create `models/` package
2. Implement `yolo_factory.py` with model creation functions

### Phase 3: Update Consumers
1. Update `ImageTransforms` to use model factory
2. Update any other code that expects loaded models from config
3. Update tests to reflect new patterns

### Phase 4: Cleanup
1. Remove any remaining heavy imports from config modules
2. Update documentation
3. Add performance tests

This design ensures clean separation of concerns, eliminates the import bottleneck, and provides a maintainable architecture for future development.