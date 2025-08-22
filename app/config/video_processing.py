from typing import Any, Dict


class VideoProcessingConfig:
    def __init__(self, config: Dict[str, Any]):
        self._config = config

    @property
    def processing_limit(self) -> int:
        return self._config.get("processing_limit", 10)

    @property
    def paths(self) -> Dict[str, Any]:
        return self._config.get("paths", {})

    @property
    def settings(self) -> Dict[str, Any]:
        return self._config.get("settings", {})
