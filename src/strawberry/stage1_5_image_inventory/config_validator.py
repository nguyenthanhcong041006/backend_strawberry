"""Config validation for Stage 1.5."""

from __future__ import annotations

import re
from time import perf_counter

from .base import Stage15Tool, ToolRunResult, utc_now_iso
from .config import Stage15Config


class Stage15ConfigValidator(Stage15Tool):
    """Validate Stage 1.5 config before the full pipeline runs."""

    def run(self, config: Stage15Config) -> ToolRunResult:
        """Validate config paths and settings."""
        started_at = utc_now_iso()
        start = perf_counter()
        warnings: list[str] = []
        errors: list[str] = []
        debug: dict[str, object] = {}

        if config.config_path and not config.config_path.exists():
            errors.append(f"Config path does not exist: {config.config_path}")

        if not config.project_root.exists():
            errors.append(f"project_root does not exist: {config.project_root}")
        elif not config.project_root.is_dir():
            errors.append(f"project_root is not a directory: {config.project_root}")

        self._validate_raw_roots(config, errors, debug)
        self._validate_extensions(config, errors, warnings, debug)
        self._validate_output_paths(config, errors, debug)
        self._validate_logging(config, errors, debug)
        self._validate_processing(config, errors, warnings, debug)
        self._validate_naming_rules(config, errors, warnings, debug)
        self._validate_numeric_crosscheck(config, errors, warnings, debug)

        success = not errors
        finished_at = utc_now_iso()
        return ToolRunResult(
            tool_name=self.tool_name,
            success=success,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=perf_counter() - start,
            message="Config validation passed." if success else "Config validation failed.",
            data=config if success else None,
            warnings=warnings,
            errors=errors,
            debug=debug,
        )

    def _validate_raw_roots(
        self, config: Stage15Config, errors: list[str], debug: dict[str, object]
    ) -> None:
        debug["raw_roots"] = [str(path) for path in config.raw_roots]
        if not config.raw_roots:
            errors.append("raw_roots must contain at least one path.")
            return
        existing_roots: list[str] = []
        for root in config.raw_roots:
            if not root.exists():
                errors.append(f"Raw root does not exist: {root}")
            elif not root.is_dir():
                errors.append(f"Raw root is not a directory: {root}")
            else:
                existing_roots.append(str(root))
        debug["existing_raw_roots"] = existing_roots

    def _validate_extensions(
        self,
        config: Stage15Config,
        errors: list[str],
        warnings: list[str],
        debug: dict[str, object],
    ) -> None:
        debug["image_extensions"] = config.image_extensions
        if not config.image_extensions:
            errors.append("image_extensions must contain at least one extension.")
            return
        normalized: list[str] = []
        for extension in config.image_extensions:
            if not extension.startswith("."):
                errors.append(f"Image extension must start with '.': {extension}")
            normalized.append(extension.lower())
        if len(set(normalized)) != len(normalized):
            warnings.append("image_extensions contains duplicate values after normalization.")

    def _validate_output_paths(
        self, config: Stage15Config, errors: list[str], debug: dict[str, object]
    ) -> None:
        required = ["inventory_csv_path", "summary_report_path", "debug_report_path"]
        output_debug: dict[str, str] = {}
        for key in required:
            value = config.output.get(key)
            if not value:
                errors.append(f"output.{key} is required.")
                continue
            path = config.resolve_path(value)
            output_debug[key] = str(path)
            if path.exists() and path.is_dir():
                errors.append(f"output.{key} points to a directory, expected file path: {path}")
        graphs_dir = config.output.get("graphs_dir")
        if graphs_dir:
            output_debug["graphs_dir"] = str(config.resolve_path(graphs_dir))
        debug["output_paths"] = output_debug

    def _validate_logging(
        self, config: Stage15Config, errors: list[str], debug: dict[str, object]
    ) -> None:
        logs_root = config.logging.get("logs_root")
        if not logs_root:
            errors.append("logging.logs_root is required.")
            return
        log_level = str(config.logging.get("level", "INFO")).upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            errors.append(f"Unsupported logging.level: {log_level}")
        debug["logs_root"] = str(config.resolve_path(logs_root))
        debug["logging_level"] = log_level

    def _validate_processing(
        self,
        config: Stage15Config,
        errors: list[str],
        warnings: list[str],
        debug: dict[str, object],
    ) -> None:
        max_workers = config.processing.get("max_workers", 1)
        try:
            max_workers_int = int(max_workers)
        except (TypeError, ValueError):
            errors.append(f"processing.max_workers must be an integer: {max_workers}")
            max_workers_int = 0
        if max_workers_int < 1:
            errors.append("processing.max_workers must be at least 1.")
        if max_workers_int > 32:
            warnings.append("processing.max_workers is high; confirm the machine can handle it.")

        sort_output_by = config.processing.get("sort_output_by", "relative_path")
        if not sort_output_by:
            errors.append("processing.sort_output_by cannot be empty.")

        for bool_key in ["parallel", "fail_fast"]:
            if bool_key in config.processing and not isinstance(config.processing[bool_key], bool):
                errors.append(f"processing.{bool_key} must be a boolean.")

        debug["processing"] = dict(config.processing)

    def _validate_naming_rules(
        self,
        config: Stage15Config,
        errors: list[str],
        warnings: list[str],
        debug: dict[str, object],
    ) -> None:
        active_rule_name = config.active_naming_rule
        rules = config.naming_rules.get("rules", {})
        debug["active_naming_rule"] = active_rule_name
        if not active_rule_name:
            errors.append("naming_rules.active_rule is required.")
            return
        if not isinstance(rules, dict) or not rules:
            errors.append("naming_rules.rules must contain at least one rule.")
            return
        active_rule = rules.get(active_rule_name)
        if not isinstance(active_rule, dict):
            errors.append(f"Active naming rule does not exist: {active_rule_name}")
            return

        regex = active_rule.get("regex")
        if not regex:
            errors.append(f"naming rule '{active_rule_name}' must define regex.")
            return
        try:
            compiled = re.compile(str(regex))
        except re.error as exc:
            errors.append(f"Invalid regex for naming rule '{active_rule_name}': {exc}")
            return

        required_fields = active_rule.get("required_fields", [])
        if not isinstance(required_fields, list):
            errors.append(f"required_fields for naming rule '{active_rule_name}' must be a list.")
            required_fields = []
        missing_fields = [field for field in required_fields if field not in compiled.groupindex]
        if missing_fields:
            errors.append(
                f"Naming rule '{active_rule_name}' regex is missing required groups: {missing_fields}"
            )

        optional_fields = active_rule.get("optional_fields", [])
        if not isinstance(optional_fields, list):
            errors.append(f"optional_fields for naming rule '{active_rule_name}' must be a list.")

        if not active_rule.get("description"):
            warnings.append(f"Naming rule '{active_rule_name}' has no description.")

        debug["naming_rule_groups"] = list(compiled.groupindex.keys())

    def _validate_numeric_crosscheck(
        self,
        config: Stage15Config,
        errors: list[str],
        warnings: list[str],
        debug: dict[str, object],
    ) -> None:
        crosscheck = config.numeric_crosscheck
        if not crosscheck:
            debug["numeric_crosscheck"] = {"enabled": False}
            return

        enabled = crosscheck.get("enabled", False)
        if not isinstance(enabled, bool):
            errors.append("numeric_crosscheck.enabled must be a boolean.")
            enabled = False
        debug["numeric_crosscheck"] = {"enabled": enabled}
        if not enabled:
            return

        output = crosscheck.get("output", {})
        if not isinstance(output, dict):
            errors.append("numeric_crosscheck.output must be a mapping.")
            output = {}
        required_outputs = [
            "image_coverage_csv_path",
            "numeric_coverage_csv_path",
            "report_path",
        ]
        output_debug: dict[str, str] = {}
        for key in required_outputs:
            value = output.get(key)
            if not value:
                errors.append(f"numeric_crosscheck.output.{key} is required when enabled.")
                continue
            path = config.resolve_path(value)
            output_debug[key] = str(path)
            if path.exists() and path.is_dir():
                errors.append(f"numeric_crosscheck.output.{key} points to a directory: {path}")

        for section_name, path_key, column_key in [
            ("hardness", "csv_path", "date_column"),
            ("environment", "csv_path", "timestamp_column"),
        ]:
            section = crosscheck.get(section_name, {})
            if not isinstance(section, dict):
                errors.append(f"numeric_crosscheck.{section_name} must be a mapping.")
                continue
            csv_path_value = section.get(path_key)
            if not csv_path_value:
                warnings.append(f"numeric_crosscheck.{section_name}.{path_key} is not configured.")
                continue
            csv_path = config.resolve_path(csv_path_value)
            output_debug[f"{section_name}_{path_key}"] = str(csv_path)
            if not csv_path.exists():
                errors.append(f"Configured {section_name} CSV does not exist: {csv_path}")
            elif not csv_path.is_file():
                errors.append(f"Configured {section_name} CSV is not a file: {csv_path}")
            if not section.get(column_key):
                errors.append(f"numeric_crosscheck.{section_name}.{column_key} is required.")

        image_timestamp = crosscheck.get("image_timestamp", {})
        if not isinstance(image_timestamp, dict):
            errors.append("numeric_crosscheck.image_timestamp must be a mapping.")
        else:
            regex = image_timestamp.get("regex")
            if not regex:
                errors.append("numeric_crosscheck.image_timestamp.regex is required when enabled.")
            else:
                try:
                    compiled = re.compile(str(regex))
                except re.error as exc:
                    errors.append(f"Invalid numeric_crosscheck.image_timestamp.regex: {exc}")
                else:
                    missing = [field for field in ["date", "time"] if field not in compiled.groupindex]
                    if missing:
                        errors.append(
                            "numeric_crosscheck.image_timestamp.regex must include named "
                            f"groups {missing}."
                        )
        debug["numeric_crosscheck_paths"] = output_debug
