"""Shared fixtures for bat_stats tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def _build_cfg() -> dict[str, Any]:
    """Hydra-style cfg with the required experiment-naming fields.

    Mirrors the shape ``configs/experiment/<name>.yaml`` would compose to
    after Hydra resolution.
    """

    return {
        "data": {
            "species": "rousettus",
            "source": "video",
            "background": "random",
        },
        "model": {"name": "arcface"},
        "loss": {"name": "arcface"},
        # Top-level shortcut keys are also allowed.
        "species": "rousettus",
        "source": "video",
        "background": "random",
        "model_name": "arcface",
    }


@pytest.fixture
def cfg() -> dict[str, Any]:
    """A canonical, fully-populated cfg for visual / naming tests."""

    return _build_cfg()


@pytest.fixture
def expected_components() -> dict[str, str]:
    return {
        "species": "rousettus",
        "source": "video",
        "background": "random",
        "model": "arcface",
        "loss": "arcface",
    }


@pytest.fixture
def cfg_namespace() -> SimpleNamespace:
    """Same fields, but exposed through attribute access (DictConfig-style)."""

    return SimpleNamespace(
        data=SimpleNamespace(species="mauritius", source="still", background="green"),
        model=SimpleNamespace(name="siamese"),
        loss=SimpleNamespace(name="bce"),
    )
