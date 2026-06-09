# scripts/analysis/reporting/summary_console_reporter.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SummaryConsoleReporter:
    """
    Console reporter for aggregated task metrics.

    This class is responsible only for human-readable terminal output.
    CSV files should keep full precision and all columns.
    """

    decimals: int = 2

    metric_labels: dict[str, str] = field(
        default_factory=lambda: {
            "cutting_error_volume": "Cutting Error Volume [voxels]",
            "part_remaining_rate": "Part Remaining Rate [%]",
            "part_occupancy_rate": "Part Occupancy Rate [%]",
        }
    )

    def print(self, summary_df: pd.DataFrame) -> None:
        print(self.render(summary_df))

    def render(self, summary_df: pd.DataFrame) -> str:
        if summary_df.empty:
            return "(empty summary)"

        blocks = []
        for idx, row in summary_df.reset_index(drop=True).iterrows():
            blocks.append(self._render_row(idx=idx, row=row))

        return "\n\n".join(blocks)

    def _render_row(self, *, idx: int, row: pd.Series) -> str:
        condition = row.get("condition", f"row_{idx}")

        lines = [
            f"[{idx + 1}] {condition}",
            f"  settings: {self._render_settings(row)}",
        ]

        max_label_len = max(len(label) for label in self.metric_labels.values())

        for metric, label in self.metric_labels.items():
            mean_col = f"{metric}_mean"
            if mean_col not in row.index:
                continue

            value = self._format_mean_std(row=row, metric=metric)
            lines.append(f"  {label:<{max_label_len}} : {value}")

        return "\n".join(lines)

    def _render_settings(self, row: pd.Series) -> str:
        settings = []

        self._append_float_setting(settings, row, "eta", "eta")
        self._append_int_setting(settings, row, "delta", "delta")
        self._append_float_setting(settings, row, "guidance_scale", "w")
        self._append_int_setting(settings, row, "sample_image_num", "M")
        self._append_int_setting(settings, row, "sampling_timesteps", "S")
        self._append_int_setting(settings, row, "num_episodes", "episodes")
        self._append_int_setting(settings, row, "num_cases", "cases")

        return ", ".join(settings) if settings else "-"

    def _append_float_setting(
        self,
        settings: list[str],
        row: pd.Series,
        column: str,
        label: str,
    ) -> None:
        if column not in row.index:
            return

        value = row[column]
        if self._is_missing(value):
            return

        settings.append(f"{label}={self._format_float(value)}")

    def _append_int_setting(
        self,
        settings: list[str],
        row: pd.Series,
        column: str,
        label: str,
    ) -> None:
        if column not in row.index:
            return

        value = row[column]
        if self._is_missing(value):
            return

        # -1 is used as an unknown fallback in some old metadata paths.
        if int(value) < 0:
            return

        settings.append(f"{label}={int(value)}")

    def _format_mean_std(self, *, row: pd.Series, metric: str) -> str:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        mean = self._format_float(row[mean_col])

        if std_col not in row.index or self._is_missing(row[std_col]):
            return mean

        std = self._format_float(row[std_col])
        return f"{mean} ± {std}"


    def _format_float(self, value: Any) -> str:
        if self._is_missing(value):
            return "-"
        return f"{float(value):.{self.decimals}f}"

    @staticmethod
    def _is_missing(value: Any) -> bool:
        return value is None or pd.isna(value)
