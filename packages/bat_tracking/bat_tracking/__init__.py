"""bat_tracking: thin MLflow facade for the bat-face-recognition workspace.

Public surface:

* :class:`MLflowTracker` --- implements :class:`bat_core.Tracker`.
* :func:`start_run` --- context manager wrapping :func:`mlflow.start_run`.
* :func:`register_model`, :func:`get_champion` --- model-registry helpers.
* :data:`KEEP`, :data:`KEEP_PREFIXES`, :data:`DROP_PREFIXES`,
  :func:`filter_params` --- HP audit allowlist.
"""

from bat_tracking.hp_audit import DROP_PREFIXES, KEEP, KEEP_PREFIXES, filter_params
from bat_tracking.mlflow_tracker import MLflowTracker
from bat_tracking.registry import get_champion, register_model
from bat_tracking.run_context import start_run

__all__ = [
    "DROP_PREFIXES",
    "KEEP",
    "KEEP_PREFIXES",
    "MLflowTracker",
    "filter_params",
    "get_champion",
    "register_model",
    "start_run",
]
