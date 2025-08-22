from typing import Any, Dict


class GeneralConfig:
    def __init__(self, config: Dict[str, Any]):
        self._config = config

    @property
    def gpu_enabled(self) -> bool:
        return self._config.get("gpu_enabled", True)

    @property
    def random_seed(self) -> int:
        return self._config.get("random_seed", 42)
