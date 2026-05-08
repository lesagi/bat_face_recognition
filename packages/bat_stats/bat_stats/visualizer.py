"""Visualization for Permutation Test Results, with experiment-aware naming.

Ported from ``app/statistical_tests/permutation_visualizer.py``.

**Naming bug fix** (Phase 1, plan: see Worker G row): every plot title,
filename, and axis label embeds the experiment signature
``{species}_{source}_{background}_{model}_{loss}`` derived from a Hydra cfg.
This is enforced both by routing every output through
:mod:`bat_stats.naming` and by a regression test in
``packages/bat_stats/tests/test_visualizer_naming.py``.

The visualizer is matplotlib-only -- no torch / TF dependency.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from bat_stats.naming import (
    build_axis_label,
    build_filename,
    build_title_suffix,
    experiment_name,
)
from bat_stats.permutation_test import PermutationTestResults


class PermutationVisualizer:
    """Render permutation-test plots and exports.

    Every artifact emitted by this class is named via
    :func:`bat_stats.naming.build_filename` and titled via
    :func:`bat_stats.naming.build_title_suffix`, so output never escapes with a
    generic name like ``null_dist_f1.png``.
    """

    def __init__(
        self,
        results: PermutationTestResults,
        cfg: Any,
        output_dir: str | Path | None = None,
        dpi: int = 150,
        figsize: tuple[int, int] = (10, 6),
    ) -> None:
        self.results = results
        self.cfg = cfg
        self.output_dir = Path(output_dir) if output_dir else None
        self.dpi = dpi
        self.figsize = figsize

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # filename / title helpers (single point of naming truth)
    # ------------------------------------------------------------------
    def _filename(self, stem: str, suffix: str = ".png") -> str:
        return build_filename(stem, self.cfg, suffix=suffix)

    def _decorate_title(self, base: str) -> str:
        return f"{base}\n{build_title_suffix(self.cfg)}"

    def _decorate_axis(self, label: str) -> str:
        return build_axis_label(label, self.cfg)

    def _save(self, fig: Figure, fname: str) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir must be set to save figures")
        filepath = self.output_dir / fname
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        print(f"Saved: {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # individual plots
    # ------------------------------------------------------------------
    def plot_null_distribution(
        self,
        metric: str,
        show: bool = True,
        save: bool = True,
        filename: str | None = None,
    ) -> Figure:
        """Plot the null-distribution histogram for a single metric."""

        if metric not in self.results.metrics:
            raise ValueError(f"Metric '{metric}' not found in results")
        result = self.results.metrics[metric]
        if not result.null_distribution:
            raise ValueError(
                f"No null distribution data available for metric '{metric}'"
            )
        null_dist = np.asarray(result.null_distribution, dtype=np.float64)

        fig, ax = plt.subplots(figsize=self.figsize)
        _n, bins, _patches = ax.hist(
            null_dist,
            bins=30,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
            label="Null Distribution",
        )

        ax.axvline(
            result.observed,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Observed ({result.observed:.4f})",
        )
        ax.axvline(
            result.null_mean,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label=f"Null Mean ({result.null_mean:.4f})",
        )

        if result.observed > result.null_mean:
            extreme_bins = bins[bins >= result.observed]
            if len(extreme_bins) > 0:
                ax.axvspan(
                    result.observed,
                    max(float(null_dist.max()), result.observed) + 0.01,
                    alpha=0.3,
                    color="red",
                    label="P-value region",
                )

        ax.set_xlabel(self._decorate_axis(metric.replace("_", " ").title()), fontsize=12)
        ax.set_ylabel(self._decorate_axis("Density"), fontsize=12)
        ax.set_title(
            self._decorate_title(
                f"Permutation Test: {metric.replace('_', ' ').title()}\n"
                f"n={self.results.n_permutations}, p={result.p_value:.4f}"
                f"{' (significant)' if result.significant else ' (not significant)'}"
            ),
            fontsize=13,
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        textstr = "\n".join(
            [
                f"Observed: {result.observed:.4f}",
                f"Null Mean: {result.null_mean:.4f}",
                f"Null Std: {result.null_std:.4f}",
                f"P-value: {result.p_value:.4f}",
                f"Significant: {'Yes' if result.significant else 'No'}",
                f"Run: {experiment_name(self.cfg)}",
            ]
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()

        if save and self.output_dir is not None:
            fname = filename or self._filename(f"null_dist_{metric}")
            self._save(fig, fname)

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def plot_all_distributions(
        self,
        show: bool = True,
        save: bool = True,
        filename: str | None = None,
    ) -> Figure:
        """Plot null distributions for all metrics in a single figure."""

        metrics_with_data = [
            m for m, r in self.results.metrics.items() if r.null_distribution
        ]
        if not metrics_with_data:
            raise ValueError("No metrics have null distribution data")

        n_metrics = len(metrics_with_data)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, raw_axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(self.figsize[0] * n_cols * 0.6, self.figsize[1] * n_rows * 0.6),
        )

        if n_metrics == 1:
            axes = [raw_axes]
        else:
            axes = list(np.asarray(raw_axes).flatten())

        for i, metric in enumerate(metrics_with_data):
            ax = axes[i]
            result = self.results.metrics[metric]
            null_dist = np.asarray(result.null_distribution, dtype=np.float64)

            ax.hist(
                null_dist,
                bins=20,
                density=True,
                alpha=0.7,
                color="steelblue",
                edgecolor="white",
            )
            ax.axvline(result.observed, color="red", linestyle="--", linewidth=2)
            ax.axvline(result.null_mean, color="gray", linestyle=":", linewidth=1.5)

            sig_marker = "*" if result.significant else ""
            ax.set_title(
                f"{metric.title()}{sig_marker}\np={result.p_value:.3f}",
                fontsize=11,
            )
            ax.set_xlabel(self._decorate_axis(metric.title()), fontsize=9)
            ax.grid(True, alpha=0.3)

        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        legend_elements = [
            mpatches.Patch(color="steelblue", alpha=0.7, label="Null Distribution"),
            plt.Line2D([0], [0], color="red", linestyle="--", label="Observed"),
            plt.Line2D([0], [0], color="gray", linestyle=":", label="Null Mean"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.02),
        )

        plt.suptitle(
            self._decorate_title(
                f"Permutation Test Results (n={self.results.n_permutations})\n"
                f"* indicates p < {self.results.significance_level}"
            ),
            y=1.10,
            fontsize=13,
        )

        plt.tight_layout()
        if save and self.output_dir is not None:
            fname = filename or self._filename("null_distributions_all")
            self._save(fig, fname)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def plot_summary_table(
        self,
        show: bool = True,
        save: bool = True,
        filename: str | None = None,
    ) -> Figure:
        """Create a visual summary table of all metric results."""

        metrics = list(self.results.metrics.keys())
        columns = [
            "Metric",
            "Observed",
            "Null Mean",
            "Null Std",
            "P-value",
            "Significant",
        ]
        cell_data: list[list[str]] = []
        cell_colors: list[list[str]] = []

        for metric in metrics:
            result = self.results.metrics[metric]
            row = [
                metric.title(),
                f"{result.observed:.4f}",
                f"{result.null_mean:.4f}",
                f"{result.null_std:.4f}",
                f"{result.p_value:.4f}",
                "Yes" if result.significant else "No",
            ]
            cell_data.append(row)
            row_colors = (
                ["lightgreen"] * len(columns)
                if result.significant
                else ["white"] * len(columns)
            )
            cell_colors.append(row_colors)

        fig, ax = plt.subplots(figsize=(12, 2 + len(metrics) * 0.5))
        ax.axis("off")
        table = ax.table(
            cellText=cell_data,
            colLabels=columns,
            cellColours=cell_colors,
            colColours=["lightblue"] * len(columns),
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)

        plt.title(
            self._decorate_title(
                f"Permutation Test Summary\n"
                f"n={self.results.n_permutations}, "
                f"significance level={self.results.significance_level}"
            ),
            fontsize=13,
            pad=20,
        )

        if save and self.output_dir is not None:
            fname = filename or self._filename("permutation_summary")
            self._save(fig, fname)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def plot_degradation_curve(
        self,
        degradation_data: dict[str, list[tuple[float, float]]],
        show: bool = True,
        save: bool = True,
        filename: str | None = None,
    ) -> Figure:
        """Plot metric values vs fraction of permuted labels."""

        fig, ax = plt.subplots(figsize=self.figsize)
        colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
        for i, (metric, curve) in enumerate(sorted(degradation_data.items())):
            fractions = [pt[0] for pt in curve]
            values = [pt[1] for pt in curve]
            color = colors[i % len(colors)]
            ax.plot(
                fractions,
                values,
                "o-",
                color=color,
                linewidth=2,
                markersize=5,
                label=metric.title(),
            )

        ax.set_xlabel(
            self._decorate_axis("Fraction of Labels Permuted"), fontsize=12
        )
        ax.set_ylabel(self._decorate_axis("Metric Value"), fontsize=12)
        ax.set_title(
            self._decorate_title(
                "Degradation Curve: Performance vs Label Permutation"
            ),
            fontsize=13,
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        textstr = (
            "At 0% permutation: real performance\n"
            "At 100% permutation: chance level"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.98,
            0.02,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        plt.tight_layout()
        if save and self.output_dir is not None:
            fname = filename or self._filename("degradation_curve")
            self._save(fig, fname)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # exports
    # ------------------------------------------------------------------
    def export_csv(self, filename: str | None = None) -> Path:
        """Export results to CSV format (filename is experiment-aware)."""

        if self.output_dir is None:
            raise ValueError("output_dir must be set to export files")
        fname = filename or self._filename("permutation_results", suffix=".csv")
        filepath = self.output_dir / fname
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Metric",
                    "Observed",
                    "Null_Mean",
                    "Null_Std",
                    "P_Value",
                    "Significant",
                    "Experiment",
                ]
            )
            exp = experiment_name(self.cfg)
            for metric, result in self.results.metrics.items():
                writer.writerow(
                    [
                        metric,
                        result.observed,
                        result.null_mean,
                        result.null_std,
                        result.p_value,
                        result.significant,
                        exp,
                    ]
                )
        print(f"Exported: {filepath}")
        return filepath

    def export_json(self, filename: str | None = None) -> Path:
        """Export full results to a JSON file with an experiment-aware filename."""

        if self.output_dir is None:
            raise ValueError("output_dir must be set to export files")
        fname = filename or self._filename("permutation_results", suffix=".json")
        filepath = self.output_dir / fname
        self.results.save(filepath)
        print(f"Exported: {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # bulk
    # ------------------------------------------------------------------
    def generate_full_report(
        self,
        show: bool = False,
        export_csv: bool = True,
        export_json: bool = True,
        degradation_data: Optional[dict[str, list[tuple[float, float]]]] = None,
    ) -> list[Path]:
        """Emit every plot + export.  Returns the list of files written."""

        if self.output_dir is None:
            raise ValueError("output_dir must be set to generate report")

        print(f"\nGenerating permutation test report in: {self.output_dir}")
        print("=" * 60)

        artifacts: list[Path] = []

        fig = self.plot_summary_table(show=show, save=True)
        plt.close(fig)
        artifacts.append(self.output_dir / self._filename("permutation_summary"))

        try:
            fig = self.plot_all_distributions(show=show, save=True)
            plt.close(fig)
            artifacts.append(
                self.output_dir / self._filename("null_distributions_all")
            )
        except ValueError as e:
            print(f"Skipping combined distributions plot: {e}")

        for metric in self.results.metrics:
            try:
                fig = self.plot_null_distribution(metric, show=show, save=True)
                plt.close(fig)
                artifacts.append(
                    self.output_dir / self._filename(f"null_dist_{metric}")
                )
            except ValueError as e:
                print(f"Skipping {metric} distribution plot: {e}")

        if degradation_data is None:
            # Look both for an experiment-aware name and a legacy generic one.
            candidates = [
                self.output_dir
                / self._filename("degradation_curve", suffix=".json"),
                self.output_dir / "degradation_curve.json",
            ]
            for deg_path in candidates:
                if deg_path.exists():
                    try:
                        with open(deg_path) as f:
                            raw = json.load(f)
                        degradation_data = {
                            k: [tuple(pt) for pt in v] for k, v in raw.items()
                        }
                        break
                    except Exception:
                        pass

        if degradation_data:
            try:
                fig = self.plot_degradation_curve(
                    degradation_data, show=show, save=True
                )
                plt.close(fig)
                artifacts.append(
                    self.output_dir / self._filename("degradation_curve")
                )
            except Exception as e:
                print(f"Skipping degradation curve plot: {e}")

        if export_csv:
            artifacts.append(self.export_csv())
        if export_json:
            artifacts.append(self.export_json())

        print("=" * 60)
        print("Report generation complete!")
        self.results.print_summary()
        return artifacts


def assert_naming_is_experiment_aware(cfg: Any) -> str:
    """Helper to fail loudly if a cfg lacks the required components.

    Returns the resolved experiment_name; raises ValueError if any component
    resolved to ``"unknown"``.
    """

    from bat_stats.naming import EXPERIMENT_NAME_FIELDS, extract_components

    components = extract_components(cfg)
    missing = [f for f in EXPERIMENT_NAME_FIELDS if components[f] == "unknown"]
    if missing:
        raise ValueError(
            "cfg is missing experiment-naming fields: "
            f"{missing}; resolved={components}"
        )
    return experiment_name(cfg)


__all__ = ["PermutationVisualizer", "assert_naming_is_experiment_aware"]
