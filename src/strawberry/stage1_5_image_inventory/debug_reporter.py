"""Structured debug reporting for Stage 1.5."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .base import Stage15Tool, ToolRunResult, utc_now_iso
from .config import Stage15Config


class PipelineDebugReporter(Stage15Tool):
    """Write a structured JSON debug report for the full pipeline run."""

    def __init__(self, config: Stage15Config) -> None:
        super().__init__()
        self.config = config

    def run(self, tool_results: list[ToolRunResult]) -> ToolRunResult:
        """Write a debug report from collected ToolRunResult objects."""
        started_at = utc_now_iso()
        start = perf_counter()
        try:
            debug_path = self._debug_report_path()
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._payload(tool_results)
            debug_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            finished_at = utc_now_iso()
            return ToolRunResult(
                tool_name=self.tool_name,
                success=False,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=perf_counter() - start,
                message="Pipeline debug report failed.",
                errors=[str(exc)],
            )

        finished_at = utc_now_iso()
        return ToolRunResult(
            tool_name=self.tool_name,
            success=True,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=perf_counter() - start,
            message=f"Wrote pipeline debug report: {debug_path}",
            data=debug_path,
            debug={"debug_report_path": str(debug_path)},
        )

    def _debug_report_path(self) -> Path:
        return self.config.resolve_path(self.config.output["debug_report_path"])

    def _payload(self, tool_results: list[ToolRunResult]) -> dict[str, Any]:
        return {
            "config_path": str(self.config.config_path) if self.config.config_path else None,
            "project_root": str(self.config.project_root),
            "raw_roots": [str(path) for path in self.config.raw_roots],
            "image_extensions": self.config.image_extensions,
            "output": {
                key: str(self.config.resolve_path(value))
                for key, value in self.config.output.items()
            },
            "processing": self.config.processing,
            "active_naming_rule": self.config.active_naming_rule,
            "tool_results": [result.to_debug_dict() for result in tool_results],
            "success": all(result.success for result in tool_results),
            "warning_count": sum(len(result.warnings) for result in tool_results),
            "error_count": sum(len(result.errors) for result in tool_results),
        }
