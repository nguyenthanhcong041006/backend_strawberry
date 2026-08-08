"""Standalone deterministic sample dataset builder for development testing."""

from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DevSampleConfig:
    """Configuration for creating a small sample copy of the raw dataset."""

    project_root: Path
    source_roots: list[Path]
    output_root: Path
    manifest_path: Path
    image_extensions: tuple[str, ...]
    sample_size: int
    random_seed: int
    include_source_root_name: bool
    copy_mode: str
    overwrite_existing: bool
    config_path: Path

    @classmethod
    def from_json(cls, config_path: str | Path) -> "DevSampleConfig":
        """Load sample builder settings from a JSON config file."""
        config_file = Path(config_path)
        with config_file.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)

        project_root = Path(raw.get("project_root", ".")).resolve()

        def resolve(path_value: str | Path) -> Path:
            path = Path(path_value)
            return path if path.is_absolute() else project_root / path

        extensions = tuple(str(ext).lower() for ext in raw.get("image_extensions", []))
        return cls(
            project_root=project_root,
            source_roots=[resolve(path_value) for path_value in raw.get("source_roots", [])],
            output_root=resolve(raw["output_root"]),
            manifest_path=resolve(raw["manifest_path"]),
            image_extensions=extensions,
            sample_size=int(raw.get("sample_size", 100)),
            random_seed=int(raw.get("random_seed", 20260629)),
            include_source_root_name=bool(raw.get("include_source_root_name", True)),
            copy_mode=str(raw.get("copy_mode", "copy")),
            overwrite_existing=bool(raw.get("overwrite_existing", False)),
            config_path=config_file,
        )


@dataclass(frozen=True)
class SampledImage:
    """Source and destination information for one sampled image."""

    sample_index: int
    source_path: Path
    source_root: Path
    destination_path: Path


@dataclass(frozen=True)
class DevSampleBuildResult:
    """Summary of one sample dataset build."""

    output_root: Path
    manifest_path: Path
    total_candidates: int
    requested_sample_size: int
    copied_count: int
    skipped_existing_count: int
    random_seed: int


class DevSampleDatasetBuilder:
    """Create a deterministic small copy of image files for development runs.

    This tool is intentionally outside the Stage 1.5 pipeline. It prepares a
    small, reproducible dataset for testing integrated tools without scanning or
    processing the full raw dataset.
    """

    def __init__(self, config: DevSampleConfig) -> None:
        self.config = config

    def build(self) -> DevSampleBuildResult:
        """Build the configured sample dataset and write its manifest."""
        self._validate()
        candidates = self._discover_candidates()
        sampled = self._sample(candidates)
        copied_count, skipped_count = self._materialize(sampled)
        self._write_manifest(sampled)
        return DevSampleBuildResult(
            output_root=self.config.output_root,
            manifest_path=self.config.manifest_path,
            total_candidates=len(candidates),
            requested_sample_size=self.config.sample_size,
            copied_count=copied_count,
            skipped_existing_count=skipped_count,
            random_seed=self.config.random_seed,
        )

    def _validate(self) -> None:
        if not self.config.source_roots:
            raise ValueError("source_roots must contain at least one path.")
        if self.config.sample_size <= 0:
            raise ValueError("sample_size must be greater than 0.")
        if self.config.copy_mode != "copy":
            raise ValueError("Only copy_mode='copy' is currently supported.")
        if not self.config.image_extensions:
            raise ValueError("image_extensions must contain at least one extension.")
        for root in self.config.source_roots:
            if not root.exists():
                raise FileNotFoundError(f"Source root does not exist: {root}")
            if not root.is_dir():
                raise NotADirectoryError(f"Source root is not a directory: {root}")

    def _discover_candidates(self) -> list[tuple[Path, Path]]:
        candidates: list[tuple[Path, Path]] = []
        accepted = set(self.config.image_extensions)
        for source_root in self.config.source_roots:
            for path in source_root.rglob("*"):
                if path.is_file() and path.suffix.lower() in accepted:
                    candidates.append((source_root, path))
        candidates.sort(key=lambda item: item[1].as_posix().lower())
        if len(candidates) < self.config.sample_size:
            raise ValueError(
                f"Requested {self.config.sample_size} images, but only found {len(candidates)} candidates."
            )
        return candidates

    def _sample(self, candidates: list[tuple[Path, Path]]) -> list[SampledImage]:
        rng = random.Random(self.config.random_seed)
        selected = rng.sample(candidates, self.config.sample_size)
        selected.sort(key=lambda item: item[1].as_posix().lower())

        sampled: list[SampledImage] = []
        for index, (source_root, source_path) in enumerate(selected, start=1):
            relative_path = source_path.relative_to(source_root)
            if self.config.include_source_root_name:
                relative_path = Path(source_root.name) / relative_path
            sampled.append(
                SampledImage(
                    sample_index=index,
                    source_path=source_path,
                    source_root=source_root,
                    destination_path=self.config.output_root / relative_path,
                )
            )
        return sampled

    def _materialize(self, sampled: list[SampledImage]) -> tuple[int, int]:
        copied_count = 0
        skipped_existing_count = 0
        for item in sampled:
            item.destination_path.parent.mkdir(parents=True, exist_ok=True)
            if item.destination_path.exists() and not self.config.overwrite_existing:
                skipped_existing_count += 1
                continue
            shutil.copy2(item.source_path, item.destination_path)
            copied_count += 1
        return copied_count, skipped_existing_count

    def _write_manifest(self, sampled: list[SampledImage]) -> None:
        self.config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_index",
                    "random_seed",
                    "source_root",
                    "source_relative_path",
                    "destination_relative_path",
                    "source_path",
                    "destination_path",
                ],
            )
            writer.writeheader()
            for item in sampled:
                writer.writerow(
                    {
                        "sample_index": item.sample_index,
                        "random_seed": self.config.random_seed,
                        "source_root": self._project_relative(item.source_root),
                        "source_relative_path": item.source_path.relative_to(item.source_root).as_posix(),
                        "destination_relative_path": item.destination_path.relative_to(
                            self.config.output_root
                        ).as_posix(),
                        "source_path": self._project_relative(item.source_path),
                        "destination_path": self._project_relative(item.destination_path),
                    }
                )

    def _project_relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.config.project_root).as_posix()
        except ValueError:
            return path.as_posix()

