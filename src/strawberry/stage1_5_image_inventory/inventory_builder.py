"""Master image inventory builder for Stage 1.5."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from .base import Stage15Tool, ToolRunResult, utc_now_iso
from .config import Stage15Config
from .file_discovery import ImageFileRecord
from .image_reader import ImageReader
from .image_statistics import ImageStatisticsExtractor
from .naming_rules import FilenameRuleParser


@dataclass
class ImageInventoryRow:
    """One master CSV row."""

    values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return row values as a dict."""
        return self.values


class ImageInventoryBuilder(Stage15Tool):
    """Build the master image inventory table."""

    def __init__(
        self,
        config: Stage15Config,
        reader: ImageReader | None = None,
        stats_extractor: ImageStatisticsExtractor | None = None,
        filename_parser: FilenameRuleParser | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.reader = reader or ImageReader()
        self.stats_extractor = stats_extractor or ImageStatisticsExtractor()
        self.filename_parser = filename_parser or FilenameRuleParser(config)

    def run(self, records: list[ImageFileRecord]) -> ToolRunResult:
        """Build and write the master inventory CSV."""
        started_at = utc_now_iso()
        start = perf_counter()
        warnings: list[str] = []
        errors: list[str] = []

        try:
            rows = self._build_rows(records)
            row_dicts = [row.to_dict() for row in rows]
            sort_column = str(self.config.processing.get("sort_output_by", "relative_path"))
            if row_dicts and sort_column in row_dicts[0]:
                row_dicts.sort(key=lambda row: str(row.get(sort_column, "")).lower())
            inventory_path = self._inventory_csv_path()
            inventory_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_csv(inventory_path, row_dicts)
        except Exception as exc:
            finished_at = utc_now_iso()
            return ToolRunResult(
                tool_name=self.tool_name,
                success=False,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=perf_counter() - start,
                message="Inventory CSV generation failed.",
                warnings=warnings,
                errors=[str(exc)],
                debug={"record_count": len(records)},
            )

        readable_count = sum(1 for row in row_dicts if row.get("is_readable") is True)
        unreadable_count = sum(1 for row in row_dicts if row.get("is_readable") is False)
        naming_matches = sum(1 for row in row_dicts if row.get("naming_rule_matched") is True)
        naming_failures = len(row_dicts) - naming_matches
        finished_at = utc_now_iso()
        return ToolRunResult(
            tool_name=self.tool_name,
            success=True,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=perf_counter() - start,
            message=f"Wrote master inventory CSV with {len(row_dicts)} rows.",
            data=row_dicts,
            warnings=warnings,
            errors=errors,
            debug={
                "inventory_csv_path": str(inventory_path),
                "row_count": len(row_dicts),
                "readable_count": readable_count,
                "unreadable_count": unreadable_count,
                "naming_rule_matches": naming_matches,
                "naming_rule_failures": naming_failures,
            },
        )

    def _build_rows(self, records: list[ImageFileRecord]) -> list[ImageInventoryRow]:
        parallel = bool(self.config.processing.get("parallel", True))
        max_workers = int(self.config.processing.get("max_workers", 1))
        if not parallel or max_workers <= 1:
            return [self._process_one(index, record) for index, record in enumerate(records, start=1)]

        rows: list[ImageInventoryRow] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_one, index, record): index
                for index, record in enumerate(records, start=1)
            }
            for future in as_completed(futures):
                rows.append(future.result())
        return rows

    def _process_one(self, image_id: int, record: ImageFileRecord) -> ImageInventoryRow:
        parse_result = self.filename_parser.parse(record.filename)
        base_values = self._base_values(image_id, record, parse_result)
        read_result = self.reader.read(record.path)
        base_values.update(
            {
                "is_readable": read_result.is_readable,
                "read_error": read_result.read_error,
                "width": read_result.width,
                "height": read_result.height,
                "channels": read_result.channels,
                "color_mode": read_result.color_mode,
            }
        )

        if not read_result.is_readable:
            base_values.update(self._empty_stat_values())
            return ImageInventoryRow(base_values)

        stats = self.stats_extractor.extract(record.path, read_result.image)
        base_values.update(
            {
                "mean_r": stats.mean_r,
                "mean_g": stats.mean_g,
                "mean_b": stats.mean_b,
                "std_r": stats.std_r,
                "std_g": stats.std_g,
                "std_b": stats.std_b,
                "min_r": stats.min_r,
                "min_g": stats.min_g,
                "min_b": stats.min_b,
                "max_r": stats.max_r,
                "max_g": stats.max_g,
                "max_b": stats.max_b,
                "brightness_mean": stats.brightness_mean,
                "brightness_std": stats.brightness_std,
                "image_hash": stats.image_hash,
            }
        )
        return ImageInventoryRow(base_values)

    def _base_values(
        self, image_id: int, record: ImageFileRecord, parse_result: object
    ) -> dict[str, Any]:
        parsed_fields = getattr(parse_result, "parsed_fields", {})
        values: dict[str, Any] = {
            "image_id": image_id,
            "relative_path": record.relative_path.as_posix(),
            "parent_folder": record.parent_folder,
            "filename": record.filename,
            "file_stem": record.file_stem,
            "extension": record.extension,
            "file_size_bytes": record.file_size_bytes,
            "created_time": self._format_datetime(record.created_time),
            "modified_time": self._format_datetime(record.modified_time),
            "naming_rule_matched": getattr(parse_result, "naming_rule_matched", False),
            "naming_rule_error": getattr(parse_result, "error", ""),
        }
        for field_name, field_value in parsed_fields.items():
            values[f"parsed_{field_name}"] = field_value
        return values

    def _empty_stat_values(self) -> dict[str, Any]:
        return {
            "mean_r": None,
            "mean_g": None,
            "mean_b": None,
            "std_r": None,
            "std_g": None,
            "std_b": None,
            "min_r": None,
            "min_g": None,
            "min_b": None,
            "max_r": None,
            "max_g": None,
            "max_b": None,
            "brightness_mean": None,
            "brightness_std": None,
            "image_hash": None,
        }

    def _inventory_csv_path(self) -> Path:
        return self.config.resolve_path(self.config.output["inventory_csv_path"])

    def _format_datetime(self, value: datetime) -> str:
        return value.isoformat()

    def _write_csv(self, inventory_path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = self._fieldnames(rows)
        with inventory_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _fieldnames(self, rows: list[dict[str, Any]]) -> list[str]:
        preferred = [
            "image_id",
            "relative_path",
            "parent_folder",
            "filename",
            "file_stem",
            "extension",
            "file_size_bytes",
            "created_time",
            "modified_time",
            "is_readable",
            "read_error",
            "width",
            "height",
            "channels",
            "color_mode",
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
            "image_hash",
            "naming_rule_matched",
            "naming_rule_error",
        ]
        all_keys: set[str] = set()
        for row in rows:
            all_keys.update(row.keys())
        parsed_keys = sorted(key for key in all_keys if key.startswith("parsed_"))
        remaining_keys = sorted(all_keys.difference(preferred).difference(parsed_keys))
        return [key for key in preferred if key in all_keys] + parsed_keys + remaining_keys
