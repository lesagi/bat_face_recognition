"""Tests for :mod:`bat_tracking.registry`."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from bat_tracking import get_champion, register_model


def test_register_model_creates_and_returns_int_version() -> None:
    client = MagicMock()
    client.create_model_version.return_value = SimpleNamespace(version="3")
    v = register_model(
        run_id="run-1",
        artifact_path="model",
        name="arcface",
        client=client,
    )
    assert v == 3
    client.create_registered_model.assert_called_once_with("arcface")
    client.create_model_version.assert_called_once_with(
        name="arcface",
        source="runs:/run-1/model",
        run_id="run-1",
    )


def test_register_model_tolerates_existing_registered_model() -> None:
    client = MagicMock()
    client.create_registered_model.side_effect = Exception("already exists")
    client.create_model_version.return_value = SimpleNamespace(version="7")
    v = register_model("run-2", "model", "arcface", client=client)
    assert v == 7


def test_get_champion_returns_production_version() -> None:
    client = MagicMock()
    mv4 = SimpleNamespace(name="arcface", version="4", run_id="r-A")
    mv5 = SimpleNamespace(name="arcface", version="5", run_id="r-B")
    client.get_latest_versions.return_value = [mv4, mv5]
    out = get_champion("arcface", client=client)
    assert out is mv5  # highest version wins
    client.get_latest_versions.assert_called_once_with("arcface", stages=["Production"])


def test_get_champion_returns_none_when_no_production() -> None:
    client = MagicMock()
    client.get_latest_versions.return_value = []
    assert get_champion("arcface", client=client) is None


def test_get_champion_returns_none_on_lookup_failure() -> None:
    client = MagicMock()
    client.get_latest_versions.side_effect = Exception("registry boom")
    assert get_champion("missing", client=client) is None
