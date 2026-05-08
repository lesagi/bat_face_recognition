from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    from bat_core.types import EvalReport, ImageRecord, RunArtifacts, SaliencyImage
    from torch import nn

ModelFamily = Literal["pair", "embedding"]
TrackerSection = Literal["train", "val", "test"]


@runtime_checkable
class FaceModel(Protocol):
    family: ModelFamily

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor: ...

    def forward_train(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...

    def export_for_inference(self) -> nn.Module: ...


@runtime_checkable
class Loss(Protocol):
    family: ModelFamily

    def __call__(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...


@runtime_checkable
class Trainer(Protocol):
    def fit(self, train_loader: Any, val_loader: Any) -> RunArtifacts: ...

    def test(self, test_loader: Any) -> EvalReport: ...


@runtime_checkable
class InterpretabilityAdapter(Protocol):
    def explain(self, model: FaceModel, samples: list[ImageRecord]) -> list[SaliencyImage]: ...


@runtime_checkable
class Tracker(Protocol):
    def log_metrics(
        self, section: TrackerSection, metrics: dict[str, float], step: int
    ) -> None: ...

    def log_artifact(self, path: Path, dest_dir: str | None = ...) -> None: ...

    def log_config(self, cfg: dict[str, Any]) -> None: ...

    def promote_to_champion(self, run_id: str, criterion: str) -> bool: ...
