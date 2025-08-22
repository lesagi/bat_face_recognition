from typing import Any, Dict


class VisualizationConfig:
    def __init__(self, config: Dict[str, Any]):
        self._config = config

    @property
    def paths(self) -> Dict[str, Any]:
        return self._config.get("paths", {})

    @property
    def display(self) -> Dict[str, Any]:
        return self._config.get("display", {})

    @property
    def plots(self) -> Dict[str, Any]:
        return self._config.get("plots", {})
