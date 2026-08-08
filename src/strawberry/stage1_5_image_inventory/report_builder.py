"""Image-property report builder for Stage 1.5."""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .base import Stage15Tool, ToolRunResult, utc_now_iso
from .config import Stage15Config


class ImageInventoryReportBuilder(Stage15Tool):
    """Build a Markdown summary from the master inventory."""

    def __init__(self, config: Stage15Config) -> None:
        super().__init__()
        self.config = config

    def run(self, inventory_df: object) -> ToolRunResult:
        """Write the image-property summary report."""
        started_at = utc_now_iso()
        start = perf_counter()
        try:
            rows = self._load_rows()
            chart_paths = self._write_charts(rows)
            report_path = self._report_path()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(self._build_markdown(rows, chart_paths), encoding="utf-8")
        except Exception as exc:
            finished_at = utc_now_iso()
            return ToolRunResult(
                tool_name=self.tool_name,
                success=False,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=perf_counter() - start,
                message="Image inventory summary report failed.",
                errors=[str(exc)],
            )

        finished_at = utc_now_iso()
        return ToolRunResult(
            tool_name=self.tool_name,
            success=True,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=perf_counter() - start,
            message=f"Wrote image inventory summary report: {report_path}",
            data=report_path,
            debug={
                "summary_report_path": str(report_path),
                "row_count": len(rows),
                "chart_paths": [str(path) for path in chart_paths],
            },
        )

    def _load_rows(self) -> list[dict[str, str]]:
        inventory_path = self.config.resolve_path(self.config.output["inventory_csv_path"])
        with inventory_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _report_path(self) -> Path:
        return self.config.resolve_path(self.config.output["summary_report_path"])

    def _graphs_dir(self) -> Path:
        configured = self.config.output.get("graphs_dir", "output/graphs/stage_1_5")
        return self.config.resolve_path(configured)

    def _build_markdown(self, rows: list[dict[str, str]], chart_paths: list[Path]) -> str:
        readable = [row for row in rows if self._as_bool(row.get("is_readable"))]
        unreadable = [row for row in rows if not self._as_bool(row.get("is_readable"))]
        lines = [
            "# Stage 1.5 Image Inventory Summary",
            "",
            "This report summarizes technical image properties from the Stage 1.5 master inventory CSV.",
            "",
            "## Run Overview",
            "",
            f"- Inventory rows: {len(rows)}",
            f"- Readable images: {len(readable)}",
            f"- Unreadable images: {len(unreadable)}",
            f"- Naming rule matches: {sum(1 for row in rows if self._as_bool(row.get('naming_rule_matched')))}",
            f"- Naming rule failures: {sum(1 for row in rows if not self._as_bool(row.get('naming_rule_matched')))}",
            "",
            "## File Types",
            "",
            self._counter_table(rows, "extension", "Extension"),
            "",
            "## Dimensions",
            "",
            self._dimension_table(readable),
            "",
            "## Color Modes And Channels",
            "",
            self._counter_table(readable, "color_mode", "Color Mode"),
            "",
            self._counter_table(readable, "channels", "Channels"),
            "",
            "## Numeric Summaries",
            "",
            self._numeric_summary_table(
                readable,
                [
                    "file_size_bytes",
                    "mean_r",
                    "mean_g",
                    "mean_b",
                    "std_r",
                    "std_g",
                    "std_b",
                    "min_r",
                    "min_g",
                    "min_b",
                    "max_r",
                    "max_g",
                    "max_b",
                    "brightness_mean",
                    "brightness_std",
                ],
            ),
            "",
            "## Charts",
            "",
            *self._chart_markdown(chart_paths),
            "",
            "## Unreadable Files",
            "",
            self._unreadable_table(unreadable),
            "",
            "## Recommended Review Actions",
            "",
            "- Confirm whether the sample inventory columns are sufficient before full-dataset execution.",
            "- Review any unreadable files or naming-rule failures before treating the inventory as stable.",
            "- Use the full config only after the sample pipeline outputs are accepted.",
            "",
        ]
        return "\n".join(lines)

    def _write_charts(self, rows: list[dict[str, str]]) -> list[Path]:
        readable = [row for row in rows if self._as_bool(row.get("is_readable"))]
        graphs_dir = self._graphs_dir()
        graphs_dir.mkdir(parents=True, exist_ok=True)
        chart_paths: list[Path] = []

        chart_paths.append(
            self._bar_chart(
                Counter(row.get("extension", "NA") or "NA" for row in rows),
                "Image Count By Extension",
                "Extension",
                "Count",
                graphs_dir / "extension_counts.png",
            )
        )
        chart_paths.append(
            self._bar_chart(
                Counter(f"{row.get('width')}x{row.get('height')}" for row in readable),
                "Readable Image Dimensions",
                "Dimension",
                "Count",
                graphs_dir / "dimension_counts.png",
            )
        )
        chart_paths.append(
            self._histogram(
                self._numeric_values(readable, "file_size_bytes"),
                "File Size Distribution",
                "File Size (bytes)",
                graphs_dir / "file_size_distribution.png",
            )
        )
        chart_paths.append(
            self._histogram(
                self._numeric_values(readable, "brightness_mean"),
                "Brightness Mean Distribution",
                "Brightness Mean",
                graphs_dir / "brightness_mean_distribution.png",
            )
        )
        chart_paths.append(
            self._boxplot(
                [
                    self._numeric_values(readable, "mean_r"),
                    self._numeric_values(readable, "mean_g"),
                    self._numeric_values(readable, "mean_b"),
                ],
                ["mean_r", "mean_g", "mean_b"],
                "RGB Mean Distribution",
                graphs_dir / "rgb_mean_boxplot.png",
            )
        )
        return chart_paths

    def _bar_chart(
        self,
        counts: Counter[str],
        title: str,
        xlabel: str,
        ylabel: str,
        path: Path,
    ) -> Path:
        labels = list(counts.keys()) or ["NA"]
        values = [counts[label] for label in labels] or [0]
        image, draw = self._chart_canvas(title, xlabel, ylabel)
        left, top, right, bottom = 90, 70, 740, 330
        self._draw_axes(draw, left, top, right, bottom)
        max_value = max(values) if values else 1
        bar_width = max(20, int((right - left) / max(len(values), 1) * 0.6))
        step = (right - left) / max(len(values), 1)
        for index, (label, value) in enumerate(zip(labels, values)):
            center = left + step * index + step / 2
            bar_height = 0 if max_value == 0 else (value / max_value) * (bottom - top)
            x0 = int(center - bar_width / 2)
            y0 = int(bottom - bar_height)
            x1 = int(center + bar_width / 2)
            draw.rectangle([x0, y0, x1, bottom], fill="#3b82f6")
            draw.text((x0, bottom + 8), str(label)[:18], fill="#111827", font=self._font(12))
            draw.text((x0, y0 - 18), str(value), fill="#111827", font=self._font(12))
        image.save(path)
        return path

    def _histogram(self, values: list[float], title: str, xlabel: str, path: Path) -> Path:
        image, draw = self._chart_canvas(title, xlabel, "Count")
        left, top, right, bottom = 90, 70, 740, 330
        self._draw_axes(draw, left, top, right, bottom)
        if values:
            bin_count = min(30, max(5, len(values) // 20))
            min_value = min(values)
            max_value = max(values)
            width = (max_value - min_value) / bin_count if max_value != min_value else 1
            bins = [0 for _ in range(bin_count)]
            for value in values:
                index = min(bin_count - 1, int((value - min_value) / width))
                bins[index] += 1
            max_bin = max(bins) if bins else 1
            bar_width = (right - left) / bin_count
            for index, count in enumerate(bins):
                x0 = int(left + index * bar_width)
                x1 = int(left + (index + 1) * bar_width - 2)
                bar_height = 0 if max_bin == 0 else (count / max_bin) * (bottom - top)
                y0 = int(bottom - bar_height)
                draw.rectangle([x0, y0, x1, bottom], fill="#10b981", outline="#064e3b")
            draw.text((left, bottom + 8), f"{min_value:.2f}", fill="#111827", font=self._font(12))
            draw.text((right - 80, bottom + 8), f"{max_value:.2f}", fill="#111827", font=self._font(12))
        image.save(path)
        return path

    def _boxplot(
        self,
        series: list[list[float]],
        labels: list[str],
        title: str,
        path: Path,
    ) -> Path:
        image, draw = self._chart_canvas(title, "", "Channel Mean")
        left, top, right, bottom = 90, 70, 740, 330
        self._draw_axes(draw, left, top, right, bottom)
        clean_series = [sorted(values) if values else [0.0] for values in series]
        all_values = [value for values in clean_series for value in values]
        min_value = min(all_values)
        max_value = max(all_values)

        def y(value: float) -> int:
            if max_value == min_value:
                return int((top + bottom) / 2)
            return int(bottom - ((value - min_value) / (max_value - min_value)) * (bottom - top))

        step = (right - left) / max(len(clean_series), 1)
        for index, values in enumerate(clean_series):
            q1 = self._percentile(values, 25)
            q2 = self._percentile(values, 50)
            q3 = self._percentile(values, 75)
            low = min(values)
            high = max(values)
            center = int(left + step * index + step / 2)
            box_half = 35
            draw.line([center, y(low), center, y(high)], fill="#111827", width=2)
            draw.rectangle([center - box_half, y(q3), center + box_half, y(q1)], outline="#2563eb", width=2)
            draw.line([center - box_half, y(q2), center + box_half, y(q2)], fill="#ef4444", width=2)
            draw.text((center - box_half, bottom + 8), labels[index], fill="#111827", font=self._font(12))
        draw.text((left, top - 18), f"{max_value:.2f}", fill="#111827", font=self._font(12))
        draw.text((left, bottom + 8), f"{min_value:.2f}", fill="#111827", font=self._font(12))
        image.save(path)
        return path

    def _chart_canvas(self, title: str, xlabel: str, ylabel: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("RGB", (800, 450), "white")
        draw = ImageDraw.Draw(image)
        draw.text((30, 20), title, fill="#111827", font=self._font(20))
        if xlabel:
            draw.text((360, 405), xlabel, fill="#111827", font=self._font(14))
        if ylabel:
            draw.text((15, 180), ylabel, fill="#111827", font=self._font(14))
        return image, draw

    def _draw_axes(self, draw: ImageDraw.ImageDraw, left: int, top: int, right: int, bottom: int) -> None:
        draw.line([left, bottom, right, bottom], fill="#111827", width=2)
        draw.line([left, top, left, bottom], fill="#111827", width=2)
        for fraction in [0.25, 0.5, 0.75]:
            y = int(bottom - (bottom - top) * fraction)
            draw.line([left, y, right, y], fill="#e5e7eb", width=1)

    def _font(self, size: int) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except OSError:
            return ImageFont.load_default()

    def _percentile(self, values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        index = (len(values) - 1) * percentile / 100
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        weight = index - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    def _chart_markdown(self, chart_paths: list[Path]) -> list[str]:
        if not chart_paths:
            return ["No charts generated."]
        report_path = self._report_path()
        lines: list[str] = []
        for chart_path in chart_paths:
            link = Path(os.path.relpath(chart_path, report_path.parent)).as_posix()
            title = chart_path.stem.replace("_", " ").title()
            lines.extend([f"### {title}", "", f"![{title}]({link})", ""])
        return lines

    def _counter_table(self, rows: list[dict[str, str]], column: str, label: str) -> str:
        counts = Counter(row.get(column, "") or "NA" for row in rows)
        if not counts:
            return "No rows available."
        lines = [f"| {label} | Count |", "| --- | ---: |"]
        for key, count in sorted(counts.items()):
            lines.append(f"| {key} | {count} |")
        return "\n".join(lines)

    def _dimension_table(self, rows: list[dict[str, str]]) -> str:
        counts = Counter(f"{row.get('width')}x{row.get('height')}" for row in rows)
        if not counts:
            return "No readable rows available."
        lines = ["| Dimension | Count |", "| --- | ---: |"]
        for key, count in sorted(counts.items()):
            lines.append(f"| {key} | {count} |")
        return "\n".join(lines)

    def _numeric_summary_table(self, rows: list[dict[str, str]], columns: list[str]) -> str:
        lines = ["| Column | Count | Min | Max | Mean |", "| --- | ---: | ---: | ---: | ---: |"]
        for column in columns:
            values = [self._as_float(row.get(column)) for row in rows]
            values = [value for value in values if value is not None]
            if not values:
                lines.append(f"| {column} | 0 | NA | NA | NA |")
                continue
            lines.append(
                f"| {column} | {len(values)} | "
                f"{min(values):.4f} | {max(values):.4f} | {mean(values):.4f} |"
            )
        return "\n".join(lines)

    def _unreadable_table(self, rows: list[dict[str, str]]) -> str:
        if not rows:
            return "No unreadable files found."
        lines = ["| Relative Path | Error |", "| --- | --- |"]
        for row in rows[:25]:
            lines.append(f"| {row.get('relative_path', '')} | {row.get('read_error', '')} |")
        if len(rows) > 25:
            lines.append(f"| ... | {len(rows) - 25} additional unreadable files omitted |")
        return "\n".join(lines)

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() == "true"

    def _as_float(self, value: Any) -> float | None:
        try:
            if value in {None, ""}:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _numeric_values(self, rows: list[dict[str, str]], column: str) -> list[float]:
        values = [self._as_float(row.get(column)) for row in rows]
        return [value for value in values if value is not None]
