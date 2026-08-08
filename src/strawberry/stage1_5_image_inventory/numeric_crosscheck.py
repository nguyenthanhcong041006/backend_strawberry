"""Cross-check image inventory coverage against numeric Stage 1.5 data."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from .base import Stage15Tool, ToolRunResult, utc_now_iso
from .config import Stage15Config


class NumericDataCrossChecker(Stage15Tool):
    """Verify that configured numeric rows have corresponding inventory images."""

    def __init__(self, config: Stage15Config) -> None:
        super().__init__()
        self.config = config
        self.crosscheck = config.numeric_crosscheck

    def run(self) -> ToolRunResult:
        """Write image-side and numeric-side coverage CSVs plus a summary report."""
        started_at = utc_now_iso()
        start = perf_counter()
        warnings: list[str] = []
        try:
            if not self.crosscheck.get("enabled", False):
                finished_at = utc_now_iso()
                return ToolRunResult(
                    tool_name=self.tool_name,
                    success=True,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=perf_counter() - start,
                    message="Numeric data cross-check skipped because it is disabled.",
                    warnings=["numeric_crosscheck.enabled is false."],
                    debug={"enabled": False},
                )

            inventory_rows = self._read_csv(self.config.resolve_path(self.config.output["inventory_csv_path"]))
            image_index = self._build_image_index(inventory_rows, warnings)
            hardness_rows = self._read_optional_numeric_csv("hardness", warnings)
            environment_rows = self._read_optional_numeric_csv("environment", warnings)

            hardness_by_date = self._index_hardness_by_date(hardness_rows, warnings)
            environment_by_timestamp = self._index_environment_by_timestamp(environment_rows, warnings)

            image_coverage_rows = self._build_image_coverage_rows(
                image_index["rows"], hardness_by_date, environment_by_timestamp
            )
            numeric_coverage_rows = self._build_numeric_coverage_rows(
                image_index,
                hardness_rows,
                hardness_by_date,
                environment_rows,
                environment_by_timestamp,
                warnings,
            )

            output_paths = self._write_outputs(image_coverage_rows, numeric_coverage_rows)
            report_text = self._build_report(
                image_coverage_rows,
                numeric_coverage_rows,
                hardness_rows,
                environment_rows,
                warnings,
                output_paths,
            )
            output_paths["report_path"].write_text(report_text, encoding="utf-8")
        except Exception as exc:
            finished_at = utc_now_iso()
            return ToolRunResult(
                tool_name=self.tool_name,
                success=False,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=perf_counter() - start,
                message="Numeric data cross-check failed.",
                errors=[str(exc)],
            )

        summary = self._coverage_summary(image_coverage_rows, numeric_coverage_rows)
        finished_at = utc_now_iso()
        return ToolRunResult(
            tool_name=self.tool_name,
            success=True,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=perf_counter() - start,
            message=f"Wrote numeric coverage cross-check report: {output_paths['report_path']}",
            warnings=warnings,
            data={
                "image_coverage_rows": image_coverage_rows,
                "numeric_coverage_rows": numeric_coverage_rows,
            },
            debug={
                "enabled": True,
                "image_coverage_csv_path": str(output_paths["image_coverage_csv_path"]),
                "numeric_coverage_csv_path": str(output_paths["numeric_coverage_csv_path"]),
                "report_path": str(output_paths["report_path"]),
                **summary,
            },
        )

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _read_optional_numeric_csv(self, section_name: str, warnings: list[str]) -> list[dict[str, str]]:
        section = self.crosscheck.get(section_name, {})
        csv_path_value = section.get("csv_path") if isinstance(section, dict) else None
        if not csv_path_value:
            warnings.append(f"No {section_name} CSV configured; skipping {section_name} coverage.")
            return []
        return self._read_csv(self.config.resolve_path(csv_path_value))

    def _build_image_index(
        self,
        inventory_rows: list[dict[str, str]],
        warnings: list[str],
    ) -> dict[str, Any]:
        regex = re.compile(str(self.crosscheck["image_timestamp"]["regex"]))
        rows: list[dict[str, str]] = []
        by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_timestamp: dict[str, list[dict[str, str]]] = defaultdict(list)

        for row in inventory_rows:
            filename = row.get("filename", "")
            match = regex.match(filename)
            image_date = ""
            image_time = ""
            timestamp_key = ""
            timestamp_parse_error = ""
            if match:
                image_date = match.group("date")
                image_time = match.group("time").replace("-", ":")
                timestamp_key = f"{image_date}T{image_time}"
            else:
                timestamp_parse_error = "filename does not match image timestamp regex"

            indexed_row = {
                **row,
                "image_date": image_date,
                "image_time": image_time,
                "image_timestamp_key": timestamp_key,
                "timestamp_parse_error": timestamp_parse_error,
            }
            rows.append(indexed_row)
            if image_date:
                by_date[image_date].append(indexed_row)
            if timestamp_key:
                by_timestamp[timestamp_key].append(indexed_row)

        unmatched = sum(1 for row in rows if row["timestamp_parse_error"])
        if unmatched:
            warnings.append(f"{unmatched} inventory image(s) could not be parsed into timestamp keys.")

        return {
            "rows": rows,
            "by_date": by_date,
            "by_timestamp": by_timestamp,
        }

    def _index_hardness_by_date(
        self,
        hardness_rows: list[dict[str, str]],
        warnings: list[str],
    ) -> dict[str, list[dict[str, str]]]:
        section = self.crosscheck.get("hardness", {})
        date_column = section.get("date_column", "time") if isinstance(section, dict) else "time"
        indexed: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in hardness_rows:
            raw_date = row.get(str(date_column), "")
            normalized_date = self._normalize_date(raw_date)
            if normalized_date:
                indexed[normalized_date].append(row)
            else:
                warnings.append(f"Could not parse hardness date value: {raw_date}")
        return indexed

    def _index_environment_by_timestamp(
        self,
        environment_rows: list[dict[str, str]],
        warnings: list[str],
    ) -> dict[str, list[dict[str, str]]]:
        section = self.crosscheck.get("environment", {})
        timestamp_column = section.get("timestamp_column", "timestamp") if isinstance(section, dict) else "timestamp"
        indexed: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in environment_rows:
            raw_timestamp = row.get(str(timestamp_column), "")
            timestamp_key = self._normalize_timestamp_key(raw_timestamp)
            if timestamp_key:
                indexed[timestamp_key].append(row)
            else:
                warnings.append(f"Could not parse environment timestamp value: {raw_timestamp}")
        return indexed

    def _build_image_coverage_rows(
        self,
        image_rows: list[dict[str, str]],
        hardness_by_date: dict[str, list[dict[str, str]]],
        environment_by_timestamp: dict[str, list[dict[str, str]]],
    ) -> list[dict[str, object]]:
        coverage_rows: list[dict[str, object]] = []
        for row in image_rows:
            image_date = row.get("image_date", "")
            timestamp_key = row.get("image_timestamp_key", "")
            hardness_count = len(hardness_by_date.get(image_date, [])) if image_date else 0
            environment_count = len(environment_by_timestamp.get(timestamp_key, [])) if timestamp_key else 0
            coverage_rows.append(
                {
                    "image_id": row.get("image_id", ""),
                    "relative_path": row.get("relative_path", ""),
                    "filename": row.get("filename", ""),
                    "image_date": image_date,
                    "image_timestamp_key": timestamp_key,
                    "timestamp_parse_error": row.get("timestamp_parse_error", ""),
                    "has_hardness_date_record": hardness_count > 0,
                    "hardness_record_count": hardness_count,
                    "has_environment_timestamp_record": environment_count > 0,
                    "environment_record_count": environment_count,
                }
            )
        return coverage_rows

    def _build_numeric_coverage_rows(
        self,
        image_index: dict[str, Any],
        hardness_rows: list[dict[str, str]],
        hardness_by_date: dict[str, list[dict[str, str]]],
        environment_rows: list[dict[str, str]],
        environment_by_timestamp: dict[str, list[dict[str, str]]],
        warnings: list[str],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        section = self.crosscheck.get("hardness", {})
        date_column = section.get("date_column", "time") if isinstance(section, dict) else "time"
        fruit_prefix = section.get("fruit_value_prefix", "fruit_") if isinstance(section, dict) else "fruit_"

        for normalized_date, records in sorted(hardness_by_date.items()):
            images = image_index["by_date"].get(normalized_date, [])
            fruit_columns = [
                key
                for record in records
                for key in record.keys()
                if str(key).startswith(str(fruit_prefix))
            ]
            unique_fruit_columns = sorted(set(fruit_columns))
            values = [
                record.get(column, "")
                for record in records
                for column in unique_fruit_columns
            ]
            non_empty_values = [value for value in values if str(value).strip() != ""]
            nonzero_values = [value for value in non_empty_values if self._as_float(value) not in {None, 0.0}]
            rows.append(
                {
                    "numeric_source": "hardness",
                    "record_key": normalized_date,
                    "record_date": normalized_date,
                    "record_timestamp_key": "",
                    "has_matching_image": len(images) > 0,
                    "matching_image_count": len(images),
                    "numeric_record_count": len(records),
                    "value_columns": "|".join(unique_fruit_columns),
                    "value_count": len(non_empty_values),
                    "nonzero_value_count": len(nonzero_values),
                    "mapping_level": "date",
                    "mapping_note": (
                        f"Hardness rows use '{date_column}' plus fruit columns; "
                        "Stage 1.5 verifies image coverage by date, not fruit position."
                    ),
                }
            )

        missing_hardness_dates = [
            self._normalize_date(row.get(str(date_column), ""))
            for row in hardness_rows
            if self._normalize_date(row.get(str(date_column), "")) not in hardness_by_date
        ]
        if missing_hardness_dates:
            warnings.append(f"{len(missing_hardness_dates)} hardness rows were not indexed by date.")

        section = self.crosscheck.get("environment", {})
        timestamp_column = section.get("timestamp_column", "timestamp") if isinstance(section, dict) else "timestamp"
        value_columns = [
            key
            for row in environment_rows[:1]
            for key in row.keys()
            if key != timestamp_column
        ]
        for timestamp_key, records in sorted(environment_by_timestamp.items()):
            images = image_index["by_timestamp"].get(timestamp_key, [])
            rows.append(
                {
                    "numeric_source": "environment",
                    "record_key": timestamp_key,
                    "record_date": timestamp_key[:10],
                    "record_timestamp_key": timestamp_key,
                    "has_matching_image": len(images) > 0,
                    "matching_image_count": len(images),
                    "numeric_record_count": len(records),
                    "value_columns": "|".join(value_columns),
                    "value_count": len(value_columns) * len(records),
                    "nonzero_value_count": "",
                    "mapping_level": "timestamp",
                    "mapping_note": (
                        f"Environment rows use '{timestamp_column}'; "
                        "Stage 1.5 compares local timestamp text to webcam filename time."
                    ),
                }
            )
        return rows

    def _write_outputs(
        self,
        image_coverage_rows: list[dict[str, object]],
        numeric_coverage_rows: list[dict[str, object]],
    ) -> dict[str, Path]:
        output_config = self.crosscheck["output"]
        paths = {
            "image_coverage_csv_path": self.config.resolve_path(output_config["image_coverage_csv_path"]),
            "numeric_coverage_csv_path": self.config.resolve_path(output_config["numeric_coverage_csv_path"]),
            "report_path": self.config.resolve_path(output_config["report_path"]),
        }
        self._write_csv(paths["image_coverage_csv_path"], image_coverage_rows)
        self._write_csv(paths["numeric_coverage_csv_path"], numeric_coverage_rows)
        paths["report_path"].parent.mkdir(parents=True, exist_ok=True)
        return paths

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else ["message"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows or [{"message": "No rows available."}])

    def _build_report(
        self,
        image_rows: list[dict[str, object]],
        numeric_rows: list[dict[str, object]],
        hardness_rows: list[dict[str, str]],
        environment_rows: list[dict[str, str]],
        warnings: list[str],
        output_paths: dict[str, Path],
    ) -> str:
        summary = self._coverage_summary(image_rows, numeric_rows)
        numeric_counter = Counter(row["numeric_source"] for row in numeric_rows)
        lines = [
            "# Stage 1.5 Numeric Coverage Cross-Check",
            "",
            "This final Stage 1.5 layer verifies coverage between the master image inventory and available numeric CSV files.",
            "",
            "## Scope",
            "",
            "- Environment readings are checked at timestamp level against webcam filename timestamps.",
            "- Hardness readings are checked at date level because the current hardness CSV has daily fruit columns, not image-localized fruit IDs.",
            "- This report verifies coverage only; it does not create labels or model-ready targets.",
            "",
            "## Outputs",
            "",
            f"- Image coverage CSV: `{output_paths['image_coverage_csv_path']}`",
            f"- Numeric coverage CSV: `{output_paths['numeric_coverage_csv_path']}`",
            "",
            "## Coverage Summary",
            "",
            "| Metric | Count |",
            "| --- | ---: |",
            f"| Inventory images checked | {summary['image_rows']} |",
            f"| Images with parsed timestamp | {summary['images_with_timestamp']} |",
            f"| Images without parsed timestamp | {summary['images_without_timestamp']} |",
            f"| Images with hardness date record | {summary['images_with_hardness']} |",
            f"| Images missing hardness date record | {summary['images_missing_hardness']} |",
            f"| Images with environment timestamp record | {summary['images_with_environment']} |",
            f"| Images missing environment timestamp record | {summary['images_missing_environment']} |",
            f"| Numeric rows checked | {summary['numeric_rows']} |",
            f"| Numeric rows with matching image | {summary['numeric_rows_with_image']} |",
            f"| Numeric rows missing matching image | {summary['numeric_rows_missing_image']} |",
            "",
            "## Numeric Sources",
            "",
            "| Source | Raw Rows | Coverage Rows |",
            "| --- | ---: | ---: |",
            f"| hardness | {len(hardness_rows)} | {numeric_counter.get('hardness', 0)} |",
            f"| environment | {len(environment_rows)} | {numeric_counter.get('environment', 0)} |",
            "",
            "## Warnings",
            "",
            self._warning_lines(warnings),
            "",
        ]
        return "\n".join(lines)

    def _coverage_summary(
        self,
        image_rows: list[dict[str, object]],
        numeric_rows: list[dict[str, object]],
    ) -> dict[str, int]:
        return {
            "image_rows": len(image_rows),
            "images_with_timestamp": sum(1 for row in image_rows if row.get("image_timestamp_key")),
            "images_without_timestamp": sum(1 for row in image_rows if not row.get("image_timestamp_key")),
            "images_with_hardness": sum(1 for row in image_rows if row.get("has_hardness_date_record") is True),
            "images_missing_hardness": sum(1 for row in image_rows if row.get("has_hardness_date_record") is False),
            "images_with_environment": sum(
                1 for row in image_rows if row.get("has_environment_timestamp_record") is True
            ),
            "images_missing_environment": sum(
                1 for row in image_rows if row.get("has_environment_timestamp_record") is False
            ),
            "numeric_rows": len(numeric_rows),
            "numeric_rows_with_image": sum(1 for row in numeric_rows if row.get("has_matching_image") is True),
            "numeric_rows_missing_image": sum(1 for row in numeric_rows if row.get("has_matching_image") is False),
        }

    def _normalize_date(self, value: str) -> str:
        text = str(value).strip()
        if not text:
            return ""
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"]:
            try:
                return datetime.strptime(text, fmt).date().isoformat()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text).date().isoformat()
        except ValueError:
            return ""

    def _normalize_timestamp_key(self, value: str) -> str:
        text = str(value).strip()
        if not text or "T" not in text:
            return ""
        date_text, rest = text.split("T", 1)
        time_text = rest[:8]
        normalized_date = self._normalize_date(date_text)
        if not normalized_date or not re.match(r"^\d{2}:\d{2}:\d{2}$", time_text):
            return ""
        return f"{normalized_date}T{time_text}"

    def _as_float(self, value: Any) -> float | None:
        try:
            if value in {None, ""}:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _warning_lines(self, warnings: list[str]) -> str:
        if not warnings:
            return "No warnings."
        return "\n".join(f"- {warning}" for warning in warnings)
