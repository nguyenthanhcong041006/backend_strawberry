"""Image file discovery for Stage 1.5."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .base import Stage15Tool, ToolRunResult, utc_now_iso
from .config import Stage15Config


@dataclass
class ImageFileRecord:
    """File-level metadata for one discovered image candidate."""

    path: Path
    relative_path: Path
    parent_folder: str
    filename: str
    file_stem: str
    extension: str
    file_size_bytes: int
    created_time: datetime
    modified_time: datetime


class ImageFileDiscoverer(Stage15Tool):
    """Discover supported image files from configured raw roots."""

    def __init__(self, config: Stage15Config) -> None:
        super().__init__()
        self.config = config

    def run(self) -> ToolRunResult:
        """Return discovered image file records."""
        started_at = utc_now_iso()
        start_time = datetime.now(timezone.utc)
        warnings: list[str] = []
        errors: list[str] = []
        records: list[ImageFileRecord] = []
        counts_by_root: Counter[str] = Counter()
        counts_by_extension: Counter[str] = Counter()

        for raw_root in self.config.raw_roots:
            if not raw_root.exists() or not raw_root.is_dir():
                errors.append(f"Raw root is not available for discovery: {raw_root}")
                continue
            for path in raw_root.rglob("*"):
                if not path.is_file():
                    continue
                if not self._is_supported_extension(path):
                    continue
                try:
                    record = self._build_file_record(path)
                except OSError as exc:
                    warnings.append(f"Could not read file metadata for {path}: {exc}")
                    continue
                records.append(record)
                counts_by_root[str(raw_root)] += 1
                counts_by_extension[record.extension] += 1

        records.sort(key=lambda record: record.relative_path.as_posix().lower())
        finished_at = utc_now_iso()
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        success = not errors

        return ToolRunResult(
            tool_name=self.tool_name,
            success=success,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration,
            message=(
                f"Discovered {len(records)} supported image files."
                if success
                else "File discovery completed with errors."
            ),
            data=records if success else None,
            warnings=warnings,
            errors=errors,
            debug={
                "total_files": len(records),
                "counts_by_root": dict(counts_by_root),
                "counts_by_extension": dict(counts_by_extension),
                "raw_roots": [str(path) for path in self.config.raw_roots],
                "image_extensions": self.config.image_extensions,
            },
        )

    def _is_supported_extension(self, path: Path) -> bool:
        return path.suffix.lower() in set(self.config.image_extensions)

    def _build_file_record(self, path: Path) -> ImageFileRecord:
        stat = path.stat()
        return ImageFileRecord(
            path=path,
            relative_path=self._relative_to_project(path),
            parent_folder=path.parent.name,
            filename=path.name,
            file_stem=path.stem,
            extension=path.suffix.lower(),
            file_size_bytes=stat.st_size,
            created_time=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            modified_time=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

    def _relative_to_project(self, path: Path) -> Path:
        try:
            return path.relative_to(self.config.project_root)
        except ValueError:
            return path
