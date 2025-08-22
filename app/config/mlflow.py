from typing import Any, Dict


class MLflowConfig:
    def __init__(self, config: Dict[str, Any]):
        self._config = config or {}

    @property
    def enabled(self) -> bool:
        return self._config.get("enabled", True)

    @property
    def experiment_name(self) -> str:
        return self._config.get("experiment_name", "bat_face_recognition")

    @property
    def tracking(self) -> Dict[str, Any]:
        return self._config.get("tracking", {})

    @property
    def tracking_host(self) -> str:
        return self.tracking.get("host", "localhost")

    @property
    def tracking_port(self) -> int:
        return self.tracking.get("port", 8080)

    @property
    def tracking_uri(self) -> str:
        return self.tracking.get("uri", "file:./mlruns")
