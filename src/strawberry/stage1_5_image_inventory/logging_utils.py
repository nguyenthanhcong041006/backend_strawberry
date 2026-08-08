"""Logging utilities for Stage 1.5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Stage15Logger:
    """Human-readable run logger for Stage 1.5."""

    log_path: Path

    @classmethod
    def create(cls, logs_root: str | Path) -> "Stage15Logger":
        """Create a timestamped logger under logs_root/YYYY_MM_DD/HH_MM_SS.txt."""
        now = datetime.now()
        root = Path(logs_root)
        log_dir = root / now.strftime("%Y_%m_%d")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = _unique_log_path(log_dir, now.strftime("%H_%M_%S"))
        logger = cls(log_path=log_path)
        logger.info(f"Log file created: {log_path}")
        return logger

    def info(self, message: str) -> None:
        """Record an info message."""
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        """Record a warning message."""
        self._write("WARNING", message)

    def error(self, message: str) -> None:
        """Record an error message."""
        self._write("ERROR", message)

    def close(self) -> None:
        """Finalize the log file."""
        self.info("Log file closed.")

    def _write(self, level: str, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] [{level}] {message}\n")


def _unique_log_path(log_dir: Path, time_stem: str) -> Path:
    """Return HH_MM_SS.txt, adding a suffix only when a collision exists."""
    candidate = log_dir / f"{time_stem}.txt"
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = log_dir / f"{time_stem}_{counter:03d}.txt"
        if not candidate.exists():
            return candidate
        counter += 1
