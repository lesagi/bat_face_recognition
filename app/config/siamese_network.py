from typing import Any, Dict


class SiameseNetworkConfig:
    def __init__(self, config: Dict[str, Any]):
        self._config = config

    @property
    def model(self) -> Dict[str, Any]:
        return self._config.get("model", {})

    @property
    def training(self) -> Dict[str, Any]:
        return self._config.get("training", {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self._config.get("evaluation", {})

    @property
    def logging(self) -> Dict[str, Any]:
        return self._config.get("logging", {})

    @property
    def advanced(self) -> Dict[str, Any]:
        return self._config.get("advanced", {})

    @property
    def deployment(self) -> Dict[str, Any]:
        return self._config.get("deployment", {})
    
    @property
    def input_paths(self) -> Dict[str, Any]:
        return self._config.get("input_paths", {})
    
    @property
    def generate_predictions(self) -> Dict[str, Any]:
        return self._config.get("generate_predictions", {})
    
    @property
    def saliency_maps(self) -> Dict[str, Any]:
        return self._config.get("saliency_maps", {})