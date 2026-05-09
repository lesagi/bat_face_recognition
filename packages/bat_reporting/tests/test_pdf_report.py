"""End-to-end test: build_unified_pdf produces a non-empty PDF with the
experiment slug visible in the artifacts directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from bat_core.types import EvalReport, SaliencyImage

pytest.importorskip("reportlab")

from bat_reporting import (  # noqa: E402
    EmbeddingProjection,
    PermutationSummary,
    ReportData,
    TrainingHistory,
    build_unified_pdf,
)
from bat_stats.naming import experiment_name  # noqa: E402

EXPERIMENT_SIGNATURE_TOKENS = ("rousettus", "video", "random", "arcface")


def _build_report_data(
    cfg: dict[str, Any],
    eval_report: EvalReport,
    saliency_images: list[SaliencyImage],
    tmp_path: Path,
) -> ReportData:
    history = TrainingHistory(
        epochs=4,
        train={"loss": [1.0, 0.6, 0.45, 0.3], "f1": [0.4, 0.55, 0.7, 0.78]},
        val={"loss": [1.1, 0.65, 0.5, 0.4], "f1": [0.35, 0.5, 0.65, 0.72]},
    )
    # Build a minimal projection: just the original images stamped as t-SNE / UMAP.
    proj = EmbeddingProjection(
        images=[
            SaliencyImage(
                identity="<projection>",
                image_path=saliency_images[0].image_path,
                saliency=saliency_images[0].saliency,
                method="tsne",
            ),
            SaliencyImage(
                identity="<projection>",
                image_path=saliency_images[1].image_path,
                saliency=saliency_images[1].saliency,
                method="umap",
            ),
        ]
    )
    perm = PermutationSummary(
        results={
            "n_permutations": 20,
            "significance_level": 0.05,
            "metrics": {
                "f1": {
                    "observed": 0.85,
                    "null_mean": 0.50,
                    "null_std": 0.05,
                    "p_value": 0.01,
                    "significant": True,
                    "null_distribution": [
                        0.45,
                        0.46,
                        0.50,
                        0.51,
                        0.49,
                        0.52,
                        0.48,
                        0.55,
                        0.50,
                        0.47,
                    ],
                }
            },
        }
    )
    return ReportData(
        cfg=cfg,
        run_id="test-run-0001",
        manifest_hash="abc123def456",
        timestamp="2026-05-09T12:00:00",
        training_history=history,
        eval_report=eval_report,
        saliency_images=saliency_images,
        embedding_projection=proj,
        permutation=perm,
    )


def test_build_unified_pdf_full(
    cfg: dict[str, Any],
    eval_report: EvalReport,
    saliency_images: list[SaliencyImage],
    tmp_path: Path,
) -> None:
    data = _build_report_data(cfg, eval_report, saliency_images, tmp_path)
    out = tmp_path / "report.pdf"
    result = build_unified_pdf(data, out)
    assert result.exists()
    assert result.stat().st_size > 1024  # non-trivial PDF

    # The header bytes should signal a PDF.
    with open(result, "rb") as f:
        head = f.read(8)
    assert head.startswith(b"%PDF"), f"not a PDF: head={head!r}"

    # Naming regression: every plot PNG that landed alongside the PDF should
    # carry the experiment signature in its filename.
    sig_name = experiment_name(cfg)
    pngs = list(tmp_path.glob("*.png"))
    assert pngs, "expected at least one plot PNG alongside the PDF"
    for png in pngs:
        assert (
            sig_name in png.name
        ), f"plot {png.name!r} is missing experiment signature {sig_name!r}"
    for token in EXPERIMENT_SIGNATURE_TOKENS:
        assert any(token in p.name for p in pngs), f"no plot file mentions token {token!r}"


def test_build_unified_pdf_minimal(
    cfg: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Builder must not crash when optional sections are absent."""

    data = ReportData(cfg=cfg, run_id="minimal-run")
    out = tmp_path / "minimal.pdf"
    result = build_unified_pdf(data, out)
    assert result.exists() and result.stat().st_size > 512
    with open(result, "rb") as f:
        assert f.read(4) == b"%PDF"


def test_build_unified_pdf_pair_only_without_identification(
    cfg: dict[str, Any],
    eval_report: EvalReport,
    tmp_path: Path,
) -> None:
    """Pair-only runs (identification=None) should still produce a valid PDF."""

    pair_only = EvalReport(
        verification=eval_report.verification,
        identification=None,
        predictions=eval_report.predictions,
    )
    data = ReportData(
        cfg=cfg,
        run_id="pair-only-run",
        eval_report=pair_only,
    )
    out = tmp_path / "pair.pdf"
    result = build_unified_pdf(data, out)
    assert result.exists() and result.stat().st_size > 512
