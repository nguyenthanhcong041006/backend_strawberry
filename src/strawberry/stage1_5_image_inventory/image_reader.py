"""Safe image reading for Stage 1.5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class ImageReadResult:
    """Result of attempting to open one image file."""

    is_readable: bool
    read_error: str
    image: Any = None
    color_mode: str | None = None
    width: int | None = None
    height: int | None = None
    channels: int | None = None


class ImageReader:
    """Open images safely without mutating raw files."""

    def read(self, path: Path) -> ImageReadResult:
        """Read one image path and return basic image metadata."""
        try:
            with Image.open(path) as image:
                image.load()
                loaded = image.copy()
        except Exception as exc:
            return ImageReadResult(is_readable=False, read_error=str(exc))

        width, height = loaded.size
        return ImageReadResult(
            is_readable=True,
            read_error="",
            image=loaded,
            color_mode=loaded.mode,
            width=width,
            height=height,
            channels=self._detect_channels(loaded),
        )

    def _detect_channels(self, image: Any) -> int:
        if image.mode == "1":
            return 1
        bands = image.getbands()
        return len(bands) if bands else 1
