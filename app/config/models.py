from typing import Any, Dict


class ModelsConfig:
    def __init__(self, config: Dict[str, Any]):
        self._config = config

    @property
    def segmentation(self) -> Dict[str, Any]:
        return self._config.get("segmentation", {})

    @property
    def pose(self) -> Dict[str, Any]:
        return self._config.get("pose", {})

    @property
    def siamese(self) -> Dict[str, Any]:
        return self._config.get("siamese", {})

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        return self._config.get(model_name, {})
