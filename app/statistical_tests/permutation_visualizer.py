"""
Visualization module for Permutation Test Results.

This module provides visualization capabilities for permutation test results,
including:
- Null distribution histograms with observed value marked
- P-value summary tables
- Confidence interval visualizations
- Export to various formats (PNG, PDF, CSV, JSON)
"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure


class PermutationVisualizer:
    def __init__(
        self,
        results: 'PermutationTestResults',  # type: ignore[name-defined]
        output_dir: Optional[str] = None,
        dpi: int = 150,
        figsize: Tuple[int, int] = (10, 6)
    ):
        self.results = results
        self.output_dir = output_dir
        self.dpi = dpi
        self.figsize = figsize
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def plot_null_distribution(
        self,
        metric: str,
        show: bool = True,
        save: bool = True,
        filename: Optional[str] = None
    ) -> Figure:
        """
        Plot the null distribution histogram for a specific metric.
        
        Args:
            metric: Name of the metric to plot
            show: Whether to display the plot
            save: Whether to save the plot to file
            filename: Custom filename (default: null_dist_{metric}.png)
            
        Returns:
            matplotlib Figure object
        """
        if metric not in self.results.metrics:
            raise ValueError(f"Metric '{metric}' not found in results")
        
        result = self.results.metrics[metric]
        
        if not result.null_distribution:
            raise ValueError(f"No null distribution data available for metric '{metric}'")
        
        null_dist = np.array(result.null_distribution)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot histogram
        n, bins, patches = ax.hist(
            null_dist, 
            bins=30, 
            density=True, 
            alpha=0.7, 
            color='steelblue',
            edgecolor='white',
            label='Null Distribution'
        )
        
        # Add vertical line for observed value
        ax.axvline(
            result.observed, 
            color='red', 
            linestyle='--', 
            linewidth=2,
            label=f'Observed ({result.observed:.4f})'
        )
        
        # Add vertical line for null mean
        ax.axvline(
            result.null_mean, 
            color='gray', 
            linestyle=':', 
            linewidth=1.5,
            label=f'Null Mean ({result.null_mean:.4f})'
        )
        
        # Shade the area more extreme than observed
        if result.observed > result.null_mean:
            # Right tail
            extreme_bins = bins[bins >= result.observed]
            if len(extreme_bins) > 0:
                ax.axvspan(
                    result.observed, 
                    max(null_dist.max(), result.observed) + 0.01,
                    alpha=0.3, 
                    color='red',
                    label=f'P-value region'
                )
        
        # Formatting
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(
            f'Permutation Test: {metric.replace("_", " ").title()}\n'
            f'n={self.results.n_permutations}, p={result.p_value:.4f}'
            f'{" (significant)" if result.significant else " (not significant)"}',
            fontsize=14
        )
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        textstr = '\n'.join([
            f'Observed: {result.observed:.4f}',
            f'Null Mean: {result.null_mean:.4f}',
            f'Null Std: {result.null_std:.4f}',
            f'P-value: {result.p_value:.4f}',
            f'Significant: {"Yes" if result.significant else "No"}'
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(
            0.02, 0.98, textstr, 
            transform=ax.transAxes, 
            fontsize=10,
            verticalalignment='top', 
            bbox=props
        )
        
        plt.tight_layout()
        
        if save and self.output_dir:
            fname = filename or f'null_dist_{metric}.png'
            filepath = os.path.join(self.output_dir, fname)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_all_distributions(
        self,
        show: bool = True,
        save: bool = True,
        filename: str = 'null_distributions_all.png'
    ) -> Figure:
        """
        Plot null distributions for all metrics in a single figure.
        
        Args:
            show: Whether to display the plot
            save: Whether to save the plot to file
            filename: Filename for the combined plot
            
        Returns:
            matplotlib Figure object
        """
        metrics_with_data = [
            m for m, r in self.results.metrics.items() 
            if r.null_distribution
        ]
        
        if not metrics_with_data:
            raise ValueError("No metrics have null distribution data")
        
        n_metrics = len(metrics_with_data)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(self.figsize[0] * n_cols * 0.6, self.figsize[1] * n_rows * 0.6)
        )
        
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics_with_data):
            ax = axes[i]
            result = self.results.metrics[metric]
            null_dist = np.array(result.null_distribution)
            
            # Histogram
            ax.hist(
                null_dist, 
                bins=20, 
                density=True, 
                alpha=0.7, 
                color='steelblue',
                edgecolor='white'
            )
            
            # Observed value line
            ax.axvline(
                result.observed, 
                color='red', 
                linestyle='--', 
                linewidth=2
            )
            
            # Null mean line
            ax.axvline(
                result.null_mean, 
                color='gray', 
                linestyle=':', 
                linewidth=1.5
            )
            
            # Title with significance
            sig_marker = '*' if result.significant else ''
            ax.set_title(
                f'{metric.title()}{sig_marker}\np={result.p_value:.3f}',
                fontsize=11
            )
            ax.set_xlabel(metric.title(), fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='steelblue', alpha=0.7, label='Null Distribution'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='Observed'),
            plt.Line2D([0], [0], color='gray', linestyle=':', label='Null Mean')
        ]
        fig.legend(
            handles=legend_elements, 
            loc='upper center', 
            ncol=3,
            bbox_to_anchor=(0.5, 1.02)
        )
        
        plt.suptitle(
            f'Permutation Test Results (n={self.results.n_permutations})\n'
            f'* indicates p < {self.results.significance_level}',
            y=1.08,
            fontsize=14
        )
        
        plt.tight_layout()
        
        if save and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_summary_table(
        self,
        show: bool = True,
        save: bool = True,
        filename: str = 'permutation_summary.png'
    ) -> Figure:
        """
        Create a visual summary table of all results.
        
        Args:
            show: Whether to display the plot
            save: Whether to save the plot to file
            filename: Filename for the summary table
            
        Returns:
            matplotlib Figure object
        """
        metrics = list(self.results.metrics.keys())
        
        # Prepare table data
        columns = ['Metric', 'Observed', 'Null Mean', 'Null Std', 'P-value', 'Significant']
        cell_data = []
        cell_colors = []
        
        for metric in metrics:
            result = self.results.metrics[metric]
            row = [
                metric.title(),
                f'{result.observed:.4f}',
                f'{result.null_mean:.4f}',
                f'{result.null_std:.4f}',
                f'{result.p_value:.4f}',
                'Yes' if result.significant else 'No'
            ]
            cell_data.append(row)
            
            # Color significant rows
            if result.significant:
                row_colors = ['lightgreen'] * len(columns)
            else:
                row_colors = ['white'] * len(columns)
            cell_colors.append(row_colors)
        
        fig, ax = plt.subplots(figsize=(12, 2 + len(metrics) * 0.5))
        ax.axis('off')
        
        table = ax.table(
            cellText=cell_data,
            colLabels=columns,
            cellColours=cell_colors,
            colColours=['lightblue'] * len(columns),
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        plt.title(
            f'Permutation Test Summary\n'
            f'n={self.results.n_permutations}, '
            f'significance level={self.results.significance_level}',
            fontsize=14,
            pad=20
        )
        
        if save and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def export_csv(self, filename: str = 'permutation_results.csv'):
        """Export results to CSV format."""
        if not self.output_dir:
            raise ValueError("output_dir must be set to export files")
        
        import csv
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Observed', 'Null_Mean', 'Null_Std', 'P_Value', 'Significant'])
            
            for metric, result in self.results.metrics.items():
                writer.writerow([
                    metric,
                    result.observed,
                    result.null_mean,
                    result.null_std,
                    result.p_value,
                    result.significant
                ])
        
        print(f"Exported: {filepath}")
    
    def generate_full_report(
        self,
        show: bool = False,
        export_csv: bool = True,
        export_json: bool = True
    ):
        """
        Generate all visualizations and exports.
        
        Args:
            show: Whether to display plots
            export_csv: Whether to export CSV
            export_json: Whether to export JSON
        """
        if not self.output_dir:
            raise ValueError("output_dir must be set to generate report")
        
        print(f"\nGenerating permutation test report in: {self.output_dir}")
        print("=" * 60)
        
        # Summary table
        self.plot_summary_table(show=show, save=True)
        
        # All distributions combined
        try:
            self.plot_all_distributions(show=show, save=True)
        except ValueError as e:
            print(f"Skipping combined distributions plot: {e}")
        
        # Individual distribution plots
        for metric in self.results.metrics:
            try:
                self.plot_null_distribution(metric, show=show, save=True)
            except ValueError as e:
                print(f"Skipping {metric} distribution plot: {e}")
        
        # Export data
        if export_csv:
            self.export_csv()
        
        if export_json:
            json_path = os.path.join(self.output_dir, 'permutation_results.json')
            self.results.save(json_path)
            print(f"Exported: {json_path}")
        
        print("=" * 60)
        print("Report generation complete!")
        
        # Print text summary
        self.results.print_summary()
