"""Metrics table exporter for classification results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.plots.export.base import BaseExporter, ExportConfig


class MetricsTableExporter(BaseExporter):
    """Generates publication-ready LaTeX tables of classification metrics.

    Creates tables showing TP, TN, FP, FN, F1, Recall, Precision, and Accuracy
    for each evaluation dataset, organized by category.
    """

    # Dataset categories for organizing rows
    DATASET_CATEGORIES = {
        "Roleplaying": [
            "roleplaying-plain",
            "roleplaying-actor",
            "roleplaying-offpolicy_train",
            "roleplaying-recital",
        ],
        "Instructed": [
            "instructed_alien",
            "instructed_peasant",
            "instructed_sarcasm",
            "instructed_counterfactual",
            "instructed_wrong_answers",
        ],
        "Control": [
            "instructed_pairs",
        ],
    }

    # Column configuration
    METRIC_COLUMNS = {
        "tp": "TP",
        "tn": "TN",
        "fp": "FP",
        "fn": "FN",
        "accuracy": "Acc",
        "precision": "Prec",
        "recall": "Recall",
        "f1": "F1",
    }

    def __init__(self, config: ExportConfig):
        super().__init__(config)

    def export_table(
        self,
        use_calibrated: bool = True,
        filename: str | None = None,
        caption: str | None = None,
        label: str | None = None,
        eval_datasets: list[str] | None = None,
        include_category_headers: bool = True,
    ) -> Path:
        """Export metrics table as LaTeX.

        Args:
            use_calibrated: If True, use calibrated threshold; otherwise use default.
            filename: Output filename (without extension).
            caption: LaTeX table caption.
            label: LaTeX table label for referencing.
            eval_datasets: Specific eval datasets to include. If None, uses categorized defaults.
            include_category_headers: If True, add category headers (Roleplaying, Instructed, etc.).

        Returns:
            Path to the exported .tex file.
        """
        if filename is None:
            threshold_suffix = "calibrated" if use_calibrated else "default"
            filename = f"metrics_table_{threshold_suffix}"

        if caption is None:
            threshold_label = "calibrated" if use_calibrated else "default"
            caption = f"Probe performance metrics at {threshold_label} threshold"

        if label is None:
            threshold_suffix = "calibrated" if use_calibrated else "default"
            label = f"tab:probe-metrics-{threshold_suffix}"

        # Get datasets (use categories if not specified)
        if eval_datasets is None:
            eval_datasets = self._get_categorized_datasets()

        # Build data
        rows = self._build_table_rows(eval_datasets, use_calibrated, include_category_headers)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Generate LaTeX
        latex = self._generate_latex(df, caption, label, include_category_headers)

        # Save
        output_path = self.config.output_dir / f"{filename}.tex"
        output_path.write_text(latex)

        print(f"✓ Exported table: {output_path}")

        # Also save CSV for reference
        csv_path = self.config.output_dir / f"{filename}.csv"
        if rows:
            # Filter out category header rows for CSV
            data_rows = [r for r in rows if r.get("Dataset", "").strip() and not r.get("_is_header", False)]
            if data_rows:
                pd.DataFrame(data_rows).to_csv(csv_path, index=False)
                print(f"✓ Exported CSV: {csv_path}")

        return output_path

    def export_comparison_table(
        self,
        filename: str | None = None,
        caption: str | None = None,
        label: str | None = None,
        eval_datasets: list[str] | None = None,
    ) -> Path:
        """Export table comparing calibrated vs default threshold metrics.

        Shows metrics side-by-side for both threshold settings.

        Args:
            filename: Output filename (without extension).
            caption: LaTeX table caption.
            label: LaTeX table label.
            eval_datasets: Specific eval datasets to include.

        Returns:
            Path to the exported .tex file.
        """
        if filename is None:
            filename = "metrics_comparison_table"

        if caption is None:
            caption = "Probe performance: calibrated vs default threshold"

        if label is None:
            label = "tab:probe-metrics-comparison"

        if eval_datasets is None:
            eval_datasets = self._get_categorized_datasets()

        rows = []
        for ds in eval_datasets:
            metrics_cal = self.get_metrics_at_calibrated_threshold(self.config.train_dataset, ds)
            metrics_def = self.get_metrics_at_default_threshold(self.config.train_dataset, ds)

            row = {"Dataset": self._format_dataset_name(ds)}

            # Calibrated metrics
            if metrics_cal:
                row["Recall (Cal)"] = f"{metrics_cal['recall']:.3f}"
                row["Prec (Cal)"] = f"{metrics_cal['precision']:.3f}"
                row["F1 (Cal)"] = f"{metrics_cal['f1']:.3f}"
            else:
                row["Recall (Cal)"] = "—"
                row["Prec (Cal)"] = "—"
                row["F1 (Cal)"] = "—"

            # Default metrics
            if metrics_def:
                row["Recall (Def)"] = f"{metrics_def['recall']:.3f}"
                row["Prec (Def)"] = f"{metrics_def['precision']:.3f}"
                row["F1 (Def)"] = f"{metrics_def['f1']:.3f}"
            else:
                row["Recall (Def)"] = "—"
                row["Prec (Def)"] = "—"
                row["F1 (Def)"] = "—"

            rows.append(row)

        df = pd.DataFrame(rows)
        latex = self._generate_comparison_latex(df, caption, label)

        output_path = self.config.output_dir / f"{filename}.tex"
        output_path.write_text(latex)

        print(f"✓ Exported comparison table: {output_path}")

        return output_path

    def _build_table_rows(
        self,
        eval_datasets: list[str],
        use_calibrated: bool,
        include_category_headers: bool,
    ) -> list[dict]:
        """Build table rows with optional category headers."""
        rows = []

        if include_category_headers:
            # Group by category
            datasets_by_category = self._group_by_category(eval_datasets)

            for category, datasets in datasets_by_category.items():
                if not datasets:
                    continue

                # Add category header row
                rows.append(
                    {
                        "Dataset": f"\\textbf{{{category}}}",
                        "_is_header": True,
                    }
                )

                # Add data rows
                for ds in datasets:
                    row = self._build_dataset_row(ds, use_calibrated)
                    if row:
                        rows.append(row)
        else:
            for ds in eval_datasets:
                row = self._build_dataset_row(ds, use_calibrated)
                if row:
                    rows.append(row)

        return rows

    def _build_dataset_row(self, eval_dataset: str, use_calibrated: bool) -> dict | None:
        """Build a single row of metrics for a dataset."""
        if use_calibrated:
            metrics = self.get_metrics_at_calibrated_threshold(self.config.train_dataset, eval_dataset)
        else:
            metrics = self.get_metrics_at_default_threshold(self.config.train_dataset, eval_dataset)

        if metrics is None:
            return None

        row = {
            "Dataset": self._format_dataset_name(eval_dataset),
            "TP": metrics["tp"],
            "TN": metrics["tn"],
            "FP": metrics["fp"],
            "FN": metrics["fn"],
            "Acc": f"{metrics['accuracy']:.3f}",
            "Prec": f"{metrics['precision']:.3f}",
            "Recall": f"{metrics['recall']:.3f}",
            "F1": f"{metrics['f1']:.3f}",
        }

        return row

    def _generate_latex(
        self,
        df: pd.DataFrame,
        caption: str,
        label: str,
        include_category_headers: bool,
    ) -> str:
        """Generate LaTeX table code with booktabs styling."""
        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{" + caption + "}",
            r"  \label{" + label + "}",
            r"  \small",
            r"  \begin{tabular}{l|rrrr|rrrr}",
            r"    \toprule",
            r"    Dataset & TP & TN & FP & FN & Acc & Prec & Recall & F1 \\",
            r"    \midrule",
        ]

        for _, row in df.iterrows():
            # Check if this is a header row (handle NaN from DataFrame)
            is_header = row.get("_is_header", False)
            if is_header is True:  # Explicit check, not truthy (NaN is truthy)
                # Category header spans all columns
                lines.append("    \\midrule")
                lines.append(f"    \\multicolumn{{9}}{{l}}{{{row['Dataset']}}} \\\\")
                lines.append("    \\midrule")
            else:
                # Regular data row
                values = [
                    str(row.get("Dataset", "")),
                    str(row.get("TP", "—")),
                    str(row.get("TN", "—")),
                    str(row.get("FP", "—")),
                    str(row.get("FN", "—")),
                    str(row.get("Acc", "—")),
                    str(row.get("Prec", "—")),
                    str(row.get("Recall", "—")),
                    str(row.get("F1", "—")),
                ]
                line = "    " + " & ".join(values) + r" \\"
                lines.append(line)

        lines.extend(
            [
                r"    \bottomrule",
                r"  \end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    def _generate_comparison_latex(
        self,
        df: pd.DataFrame,
        caption: str,
        label: str,
    ) -> str:
        """Generate LaTeX for comparison table."""
        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{" + caption + "}",
            r"  \label{" + label + "}",
            r"  \small",
            r"  \begin{tabular}{l|rrr|rrr}",
            r"    \toprule",
            r"    & \multicolumn{3}{c|}{Calibrated} & \multicolumn{3}{c}{Default} \\",
            r"    Dataset & Recall & Prec & F1 & Recall & Prec & F1 \\",
            r"    \midrule",
        ]

        for _, row in df.iterrows():
            values = [
                str(row.get("Dataset", "")),
                str(row.get("Recall (Cal)", "—")),
                str(row.get("Prec (Cal)", "—")),
                str(row.get("F1 (Cal)", "—")),
                str(row.get("Recall (Def)", "—")),
                str(row.get("Prec (Def)", "—")),
                str(row.get("F1 (Def)", "—")),
            ]
            line = "    " + " & ".join(values) + r" \\"
            lines.append(line)

        lines.extend(
            [
                r"    \bottomrule",
                r"  \end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    def _get_categorized_datasets(self) -> list[str]:
        """Get all datasets from categories that are available."""
        available = set(self.get_available_eval_datasets())
        result = []

        for category_datasets in self.DATASET_CATEGORIES.values():
            for ds in category_datasets:
                if ds in available:
                    result.append(ds)

        # Add any remaining datasets not in categories
        for ds in sorted(available):
            if ds not in result:
                result.append(ds)

        return result

    def _group_by_category(self, datasets: list[str]) -> dict[str, list[str]]:
        """Group datasets by their category."""
        result = {}
        categorized = set()

        for category, category_datasets in self.DATASET_CATEGORIES.items():
            matching = [ds for ds in datasets if ds in category_datasets]
            if matching:
                result[category] = matching
                categorized.update(matching)

        # Add uncategorized datasets
        uncategorized = [ds for ds in datasets if ds not in categorized]
        if uncategorized:
            result["Other"] = uncategorized

        return result

    def _format_dataset_name(self, dataset: str) -> str:
        """Format dataset name for table display."""
        # Handle hyphenated names
        parts = dataset.replace("_", " ").replace("-", " ").split()
        formatted = " ".join(p.capitalize() for p in parts)

        # Map to cleaner names
        name_map = {
            "Roleplaying Plain": "Roleplaying",
            "Roleplaying Actor": "Roleplaying (Actor)",
            "Roleplaying Offpolicy Train": "Roleplaying (Off-Policy)",
            "Roleplaying Recital": "Roleplaying (Recital)",
            "Instructed Alien": "Instructed (Alien)",
            "Instructed Peasant": "Instructed (Peasant)",
            "Instructed Sarcasm": "Instructed (Sarcasm)",
            "Instructed Counterfactual": "Instructed (Counterfactual)",
            "Instructed Wrong Answers": "Instructed (Wrong Answers)",
            "Instructed Pairs": "Instructed Pairs (Train)",
        }

        return name_map.get(formatted, formatted)
