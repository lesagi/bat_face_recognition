"""``with start_run(...) as tracker`` context manager.

Wraps :func:`mlflow.start_run` and yields a :class:`MLflowTracker` already
bound to the new run id. On exit the run is closed regardless of success or
failure.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import mlflow

from bat_tracking.mlflow_tracker import MLflowTracker


@contextmanager
def start_run(
    experiment_name: str,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> Iterator[MLflowTracker]:
    """Open an MLflow run and yield a configured :class:`MLflowTracker`.

    Parameters
    ----------
    experiment_name:
        MLflow experiment to log under. Created if missing.
    run_name:
        Optional human-readable run name (logged as ``mlflow.runName`` tag).
    tags:
        Optional run tags applied at start time.
    tracking_uri:
        Optional MLflow tracking URI. Forwarded to :class:`MLflowTracker`.

    Yields
    ------
    MLflowTracker
        Tracker bound to the freshly-started run id; valid only inside the
        ``with`` block.
    """
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    # Build the tracker first so the experiment exists by the time start_run
    # tries to resolve experiment_id.
    tracker = MLflowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )

    with mlflow.start_run(
        experiment_id=tracker.experiment_id,
        run_name=run_name,
        tags=tags,
    ) as active_run:
        tracker.set_run_id(active_run.info.run_id)
        yield tracker


__all__ = ["start_run"]
