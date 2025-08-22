from typing import Any, Dict, List


class YOLOSegmentationConfig:
    def __init__(self, config: Dict[str, Any]):
        self._config = config

    @property
    def training(self) -> Dict[str, Any]:
        return self._config.get("training", {})

    @property
    def supported_base_models(self) -> List[str]:
        return self._config.get("supported_base_models", [])

    @property
    def data(self) -> Dict[str, Any]:
        return self._config.get("data", {})

    @property
    def outputs(self) -> Dict[str, Any]:
        return self._config.get("outputs", {})

    @property
    def data_path(self) -> str:
        return self.training.get("data_path", "rousesttus/data/data.yaml")

    @property
    def name(self) -> str:
        return self.training.get("name", "bat_face_seg")

    @property
    def augmentation(self) -> Dict[str, Any]:
        return self._config.get("augmentation", {})
