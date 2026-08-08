"""Pipeline orchestration for Stage 1.5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import ToolRunResult
from .config import Stage15Config
from .debug_reporter import PipelineDebugReporter
from .file_discovery import ImageFileDiscoverer
from .inventory_builder import ImageInventoryBuilder
from .logging_utils import Stage15Logger
from .numeric_crosscheck import NumericDataCrossChecker
from .report_builder import ImageInventoryReportBuilder


@dataclass
class Stage15RunResult:
    """Final result returned by the Stage 1.5 pipeline."""

    inventory_csv_path: Path | None
    report_path: Path | None
    debug_report_path: Path | None
    numeric_crosscheck_report_path: Path | None
    log_path: Path | None
    total_files: int
    readable_files: int
    unreadable_files: int
    tool_results: list[ToolRunResult]


class Stage15Pipeline:
    """Orchestrate the validated Stage 1.5 workflow."""

    def __init__(self, config: Stage15Config, logger: Stage15Logger | None = None) -> None:
        self.config = config
        self.logger = logger
        self.discoverer = ImageFileDiscoverer(config)
        self.inventory_builder = ImageInventoryBuilder(config)
        self.report_builder = ImageInventoryReportBuilder(config)
        self.numeric_crosschecker = NumericDataCrossChecker(config)
        self.debug_reporter = PipelineDebugReporter(config)

    def run(self) -> Stage15RunResult:
        """Run the currently implemented Stage 1.5 workflow."""
        tool_results: list[ToolRunResult] = []
        discovery_result = self.discoverer.run()
        tool_results.append(discovery_result)

        if self.logger:
            if discovery_result.success:
                self.logger.info(discovery_result.message)
            else:
                self.logger.error(discovery_result.message)
            for warning in discovery_result.warnings:
                self.logger.warning(warning)
            for error in discovery_result.errors:
                self.logger.error(error)
            self.logger.info(f"Discovery debug: {discovery_result.debug}")

        records = discovery_result.data if discovery_result.success else []
        inventory_result = self.inventory_builder.run(records)
        tool_results.append(inventory_result)

        if self.logger:
            if inventory_result.success:
                self.logger.info(inventory_result.message)
            else:
                self.logger.error(inventory_result.message)
            for warning in inventory_result.warnings:
                self.logger.warning(warning)
            for error in inventory_result.errors:
                self.logger.error(error)
            self.logger.info(f"Inventory debug: {inventory_result.debug}")

        readable_files = int(inventory_result.debug.get("readable_count", 0))
        unreadable_files = int(inventory_result.debug.get("unreadable_count", 0))
        inventory_csv_path = inventory_result.debug.get("inventory_csv_path")

        report_result = self.report_builder.run(inventory_result.data)
        tool_results.append(report_result)
        if self.logger:
            if report_result.success:
                self.logger.info(report_result.message)
            else:
                self.logger.error(report_result.message)
            for warning in report_result.warnings:
                self.logger.warning(warning)
            for error in report_result.errors:
                self.logger.error(error)

        numeric_crosscheck_report_path = None
        if self.config.numeric_crosscheck.get("enabled", False):
            numeric_result = self.numeric_crosschecker.run()
            tool_results.append(numeric_result)
            if self.logger:
                if numeric_result.success:
                    self.logger.info(numeric_result.message)
                else:
                    self.logger.error(numeric_result.message)
                for warning in numeric_result.warnings:
                    self.logger.warning(warning)
                for error in numeric_result.errors:
                    self.logger.error(error)
                self.logger.info(f"Numeric cross-check debug: {numeric_result.debug}")
            report_path_value = numeric_result.debug.get("report_path")
            if report_path_value:
                numeric_crosscheck_report_path = Path(str(report_path_value))

        debug_result = self.debug_reporter.run(tool_results)
        tool_results.append(debug_result)
        if self.logger:
            if debug_result.success:
                self.logger.info(debug_result.message)
            else:
                self.logger.error(debug_result.message)
            for warning in debug_result.warnings:
                self.logger.warning(warning)
            for error in debug_result.errors:
                self.logger.error(error)

        summary_report_path = report_result.debug.get("summary_report_path")
        debug_report_path = debug_result.debug.get("debug_report_path")

        return Stage15RunResult(
            inventory_csv_path=Path(str(inventory_csv_path)) if inventory_csv_path else None,
            report_path=Path(str(summary_report_path)) if summary_report_path else None,
            debug_report_path=Path(str(debug_report_path)) if debug_report_path else None,
            numeric_crosscheck_report_path=numeric_crosscheck_report_path,
            log_path=self.logger.log_path if self.logger else None,
            total_files=len(records),
            readable_files=readable_files,
            unreadable_files=unreadable_files,
            tool_results=tool_results,
        )
