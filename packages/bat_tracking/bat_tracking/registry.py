"""MLflow Model Registry helpers.

Two thin wrappers around :class:`mlflow.MlflowClient` calls; kept separate
from :class:`MLflowTracker` so non-tracking code (e.g. CLI promote command)
can use them without instantiating a tracker.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, cast

import mlflow
from mlflow.tracking import MlflowClient

if TYPE_CHECKING:
    from mlflow.entities.model_registry import ModelVersion


_PROD_STAGE = "Production"


def register_model(
    run_id: str,
    artifact_path: str,
    name: str,
    tracking_uri: str | None = None,
    client: MlflowClient | None = None,
) -> int:
    """Register the model logged at ``runs:/{run_id}/{artifact_path}``.

    Returns the new model version (as ``int``). Idempotently creates the
    registered model entry if it does not already exist.
    """
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    c = client if client is not None else MlflowClient(tracking_uri=tracking_uri)

    # create_registered_model raises on duplicates; tolerate that explicitly.
    with contextlib.suppress(Exception):
        c.create_registered_model(name)

    source = f"runs:/{run_id}/{artifact_path}"
    mv = c.create_model_version(name=name, source=source, run_id=run_id)
    return int(mv.version)


def get_champion(
    name: str,
    tracking_uri: str | None = None,
    client: MlflowClient | None = None,
) -> ModelVersion | None:
    """Return the current Production-stage version of *name*, if any."""
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    c = client if client is not None else MlflowClient(tracking_uri=tracking_uri)
    try:
        versions = c.get_latest_versions(name, stages=[_PROD_STAGE])
    except Exception:
        return None
    if not versions:
        return None
    # Highest version number wins; usually there's exactly one Production.
    return cast("ModelVersion", max(versions, key=lambda mv: int(mv.version)))


__all__ = ["register_model", "get_champion"]
