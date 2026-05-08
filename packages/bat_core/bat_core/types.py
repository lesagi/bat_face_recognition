from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from bat_core.exceptions import InvalidManifestError
from pydantic import BaseModel, ConfigDict, Field, field_validator

Split = Literal["train", "val", "test"]
Species = Literal["mauritius", "rousettus"]
Background = Literal["green", "random", "original"]
Source = Literal["video", "still"]


class ImageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    identity: str
    species: Species
    background: Background
    source: Source
    augmented: bool = False
    split: Split
    quality: float = Field(ge=0.0)

    @field_validator("identity")
    @classmethod
    def _identity_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("identity must be non-empty")
        return v


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[ImageRecord]
    manifest_hash: str

    @classmethod
    def from_records(cls, records: list[ImageRecord]) -> Manifest:
        payload = [
            {
                "path": str(r.path),
                "identity": r.identity,
                "species": r.species,
                "background": r.background,
                "source": r.source,
                "augmented": r.augmented,
                "split": r.split,
                "quality": round(r.quality, 6),
            }
            for r in records
        ]
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return cls(records=records, manifest_hash=digest)

    def filter_split(self, split: Split) -> list[ImageRecord]:
        return [r for r in self.records if r.split == split]

    def identities(self, split: Split | None = None) -> set[str]:
        rs = self.records if split is None else self.filter_split(split)
        return {r.identity for r in rs}

    def assert_identity_disjoint(self) -> None:
        train_ids = self.identities("train")
        val_ids = self.identities("val")
        test_ids = self.identities("test")
        for a, b, name in [
            (train_ids, val_ids, "train/val"),
            (train_ids, test_ids, "train/test"),
            (val_ids, test_ids, "val/test"),
        ]:
            overlap = a & b
            if overlap:
                raise InvalidManifestError(f"identity overlap in {name}: {sorted(overlap)[:5]}...")


@dataclass(frozen=True)
class Embedding:
    """Embedding vectors with their identity labels.

    `tensor` is shape (N, D); `identities` length N.
    """

    tensor: object  # torch.Tensor at runtime; typed loosely to keep bat_core import-light
    identities: tuple[str, ...]

    def __post_init__(self) -> None:
        n_ids = len(self.identities)
        # torch import is local to avoid hard dep at type-check time
        try:
            import torch
        except ImportError:  # pragma: no cover
            return
        if isinstance(self.tensor, torch.Tensor):
            if self.tensor.ndim != 2:
                raise ValueError(f"embedding tensor must be 2-D, got {self.tensor.ndim}")
            if self.tensor.shape[0] != n_ids:
                raise ValueError(
                    f"tensor rows ({self.tensor.shape[0]}) != identities length ({n_ids})"
                )


@dataclass(frozen=True)
class Predictions:
    """Pair-wise predictions from a verification model."""

    y_true: tuple[int, ...]
    y_score: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.y_true) != len(self.y_score):
            raise ValueError("y_true and y_score length mismatch")


@dataclass(frozen=True)
class VerificationMetrics:
    roc_auc: float
    youden_j: float
    optimal_threshold: float
    tar_at_far_1e3: float
    tar_at_far_1e4: float


@dataclass(frozen=True)
class IdentificationMetrics:
    top1: float
    top5: float
    map: float
    cmc: tuple[float, ...]


@dataclass(frozen=True)
class EvalReport:
    verification: VerificationMetrics
    identification: IdentificationMetrics | None
    predictions: Predictions | None = None


@dataclass(frozen=True)
class SaliencyImage:
    identity: str
    image_path: Path
    saliency: object  # numpy.ndarray; loose typing
    method: str


@dataclass
class RunArtifacts:
    run_id: str
    output_dir: Path
    best_metrics: dict[str, float] = field(default_factory=dict)
    final_metrics: dict[str, float] = field(default_factory=dict)
    checkpoints: dict[str, Path] = field(default_factory=dict)
    manifest_hash: str | None = None
