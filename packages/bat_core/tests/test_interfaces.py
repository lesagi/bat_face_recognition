from pathlib import Path
from typing import Any

from bat_core import FaceModel, InterpretabilityAdapter, Loss, Tracker, Trainer
from bat_core.interfaces import ModelFamily


class _DummyModel:
    family: ModelFamily = "embedding"

    def forward_embedding(self, x: Any) -> Any:
        return x

    def forward_train(self, x: Any, labels: Any) -> Any:
        return x

    def export_for_inference(self) -> Any:
        return self


class _DummyLoss:
    family: ModelFamily = "embedding"

    def __call__(
        self,
        model_output: Any,
        labels: Any,
        sample_weights: Any = None,
    ) -> Any:
        return model_output.sum() if hasattr(model_output, "sum") else model_output


class _DummyTrainer:
    def fit(self, train_loader: Any, val_loader: Any) -> Any:
        return None

    def test(self, test_loader: Any) -> Any:
        return None


class _DummyAdapter:
    def explain(self, model: Any, samples: Any) -> Any:
        return []


class _DummyTracker:
    def log_metrics(self, section: Any, metrics: Any, step: int) -> None:
        pass

    def log_artifact(self, path: Path, dest_dir: Any = None) -> None:
        pass

    def log_config(self, cfg: Any) -> None:
        pass

    def promote_to_champion(self, run_id: str, criterion: str) -> bool:
        return False


def test_dummy_model_satisfies_protocol() -> None:
    assert isinstance(_DummyModel(), FaceModel)


def test_dummy_loss_satisfies_protocol() -> None:
    assert isinstance(_DummyLoss(), Loss)


def test_dummy_trainer_satisfies_protocol() -> None:
    assert isinstance(_DummyTrainer(), Trainer)


def test_dummy_adapter_satisfies_protocol() -> None:
    assert isinstance(_DummyAdapter(), InterpretabilityAdapter)


def test_dummy_tracker_satisfies_protocol() -> None:
    assert isinstance(_DummyTracker(), Tracker)


def test_non_conforming_object_fails_protocol() -> None:
    class NotAModel:
        pass

    assert not isinstance(NotAModel(), FaceModel)
