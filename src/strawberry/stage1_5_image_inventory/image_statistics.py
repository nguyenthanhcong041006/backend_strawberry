"""Image statistics extraction for Stage 1.5."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ImageStats:
    """Technical statistics extracted from one readable image."""

    mean_r: float | None = None
    mean_g: float | None = None
    mean_b: float | None = None
    std_r: float | None = None
    std_g: float | None = None
    std_b: float | None = None
    min_r: int | None = None
    min_g: int | None = None
    min_b: int | None = None
    max_r: int | None = None
    max_g: int | None = None
    max_b: int | None = None
    brightness_mean: float | None = None
    brightness_std: float | None = None
    image_hash: str | None = None


class ImageStatisticsExtractor:
    """Compute technical image-property statistics."""

    def extract(self, path: Path, image: Any) -> ImageStats:
        """Extract RGB and brightness statistics from one readable image."""
        rgb_array = self._to_rgb_array(image)
        brightness = self._compute_brightness(rgb_array)
        return ImageStats(
            mean_r=float(rgb_array[:, :, 0].mean()),
            mean_g=float(rgb_array[:, :, 1].mean()),
            mean_b=float(rgb_array[:, :, 2].mean()),
            std_r=float(rgb_array[:, :, 0].std()),
            std_g=float(rgb_array[:, :, 1].std()),
            std_b=float(rgb_array[:, :, 2].std()),
            min_r=int(rgb_array[:, :, 0].min()),
            min_g=int(rgb_array[:, :, 1].min()),
            min_b=int(rgb_array[:, :, 2].min()),
            max_r=int(rgb_array[:, :, 0].max()),
            max_g=int(rgb_array[:, :, 1].max()),
            max_b=int(rgb_array[:, :, 2].max()),
            brightness_mean=float(brightness.mean()),
            brightness_std=float(brightness.std()),
            image_hash=self._compute_hash(path),
        )

    def _to_rgb_array(self, image: Any) -> np.ndarray:
        rgb_image = image.convert("RGB")
        return np.asarray(rgb_image, dtype=np.float32)

    def _compute_hash(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _compute_brightness(self, rgb_array: np.ndarray) -> np.ndarray:
        return (
            0.299 * rgb_array[:, :, 0]
            + 0.587 * rgb_array[:, :, 1]
            + 0.114 * rgb_array[:, :, 2]
        )
