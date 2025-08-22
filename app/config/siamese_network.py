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
