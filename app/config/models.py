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
            
        # Extract configuration values with defaults
        model_path = model_data.get("model_path", "")
        model_type = model_data.get("model_type", "")
        confidence_threshold = model_data.get("confidence_threshold", 0.3)
        
        # Skip creating ImageModelConfig for siamese models since they have different model_type
        # that's not supported by ImageModelConfig's Literal type constraint
        if model_type == "siamese_network":
            return None
            
        # Validate that we have required data
        if not model_path or not model_type:
            return None
            
        return ImageModelConfig(
            model_path=model_path,
            model_type=model_type,
            confidence_threshold=confidence_threshold
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

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Legacy method for backward compatibility - returns raw config data"""
        return self._config.get(model_name, {})
