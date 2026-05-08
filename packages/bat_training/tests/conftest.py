"""Shared fixtures + dependency guards for bat_training tests.

Tests gate ``torch`` / ``accelerate`` imports through
``pytest.importorskip`` so collection stays graceful on environments
without those deps. We additionally insert the workspace package roots
into ``sys.path`` so tests run without `uv sync` if needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
for pkg in (
    "bat_core",
    "bat_data",
    "bat_evaluation",
    "bat_losses",
    "bat_models",
    "bat_tracking",
    "bat_training",
):
    pkg_root = _REPO_ROOT / "packages" / pkg
    if pkg_root.exists() and str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "true")


# ---------------------------------------------------------------------------
# Shared synthetic fixtures used by multiple test modules
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_pair_model_cls() -> Any:
    """Minimal pair-family model class.

    Returned as a *class* so tests can instantiate it cleanly when they
    need a fresh copy (e.g. permutation factories).
    """
    pytest.importorskip("torch")
    import torch

    class _TinyPairModel(torch.nn.Module):
        family = "pair"
        recommended_weight_decay = 1e-4

        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)
            self.classifier = torch.nn.Linear(4, 1)

        def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.fc(x))

        def forward_train(
            self,
            x_pair: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
            labels: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if isinstance(x_pair, tuple):
                a, b = x_pair
            else:
                a, b = x_pair[:, 0], x_pair[:, 1]
            ea = torch.sigmoid(self.fc(a))
            eb = torch.sigmoid(self.fc(b))
            return torch.sigmoid(self.classifier(torch.abs(ea - eb))).squeeze(-1)

        def forward(self, x_pair: Any, labels: Any = None) -> torch.Tensor:
            return self.forward_train(x_pair, labels)

        def export_for_inference(self) -> torch.nn.Module:
            return self.fc

    return _TinyPairModel


@pytest.fixture
def make_pair_loader() -> Any:
    """Factory returning a fresh deterministic 4-pair loader."""
    pytest.importorskip("torch")
    import torch

    def _build() -> list:
        torch.manual_seed(0)
        x_a = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        x_b = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        label = torch.tensor([1.0, 1.0, 0.0, 0.0])
        return [(x_a, x_b, label)]

    return _build
