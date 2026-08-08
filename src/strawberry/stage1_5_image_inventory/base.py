"""Shared execution contract for Stage 1.5 tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable


def utc_now_iso() -> str:
    """Return a timezone-aware ISO timestamp for reproducible run metadata."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolRunResult:
    """Structured result returned by each Stage 1.5 executable component."""

    tool_name: str
    success: bool
    started_at: str
    finished_at: str
    duration_seconds: float
    message: str
    data: Any = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_debug_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary without large payload objects."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "message": self.message,
            "warnings": self.warnings,
            "errors": self.errors,
            "debug": self.debug,
        }


class Stage15Tool(ABC):
    """Base class for Stage 1.5 components that produce ToolRunResult objects."""

    tool_name: str

    def __init__(self, tool_name: str | None = None) -> None:
        self.tool_name = tool_name or self.__class__.__name__

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> ToolRunResult:
        """Execute the tool and return a structured result."""

    def _timed_result(
        self,
        action: Callable[[], Any],
        success_message: str,
        failure_message: str,
    ) -> ToolRunResult:
        """Run a small action and wrap success or failure in ToolRunResult."""
        started_at = utc_now_iso()
        start = perf_counter()
        try:
            data = action()
        except Exception as exc:  # pragma: no cover - skeleton safety wrapper
            finished_at = utc_now_iso()
            return ToolRunResult(
                tool_name=self.tool_name,
                success=False,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=perf_counter() - start,
                message=failure_message,
                errors=[str(exc)],
            )

        finished_at = utc_now_iso()
        return ToolRunResult(
            tool_name=self.tool_name,
            success=True,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=perf_counter() - start,
            message=success_message,
            data=data,
        )

