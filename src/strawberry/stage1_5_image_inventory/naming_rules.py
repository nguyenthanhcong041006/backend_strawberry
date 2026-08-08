"""Filename rule parsing for Stage 1.5."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Stage15Config


@dataclass
class FilenameParseResult:
    """Result of validating and parsing one filename."""

    naming_rule_matched: bool
    parsed_fields: dict[str, str] = field(default_factory=dict)
    error: str = ""


class FilenameRuleParser:
    """Apply config-driven naming rules to filenames."""

    def __init__(self, config: Stage15Config) -> None:
        self.config = config

    def parse(self, filename: str) -> FilenameParseResult:
        """Parse a filename using the active naming rule."""
        rule_name = self.config.active_naming_rule
        active_rule = self._get_active_rule()
        if active_rule is None:
            return FilenameParseResult(
                naming_rule_matched=False,
                error=f"Active naming rule is not available: {rule_name}",
            )

        regex = active_rule.get("regex")
        if not regex:
            return FilenameParseResult(
                naming_rule_matched=False,
                error=f"Naming rule '{rule_name}' does not define regex.",
            )

        file_stem = Path(filename).stem
        try:
            match = re.match(str(regex), file_stem)
        except re.error as exc:
            return FilenameParseResult(
                naming_rule_matched=False,
                error=f"Invalid regex for naming rule '{rule_name}': {exc}",
            )

        if match is None:
            return FilenameParseResult(
                naming_rule_matched=False,
                error=f"Filename did not match rule '{rule_name}'.",
            )

        parsed_fields = {key: value for key, value in match.groupdict().items() if value is not None}
        missing_required = [
            field
            for field in active_rule.get("required_fields", [])
            if not parsed_fields.get(str(field))
        ]
        if missing_required:
            return FilenameParseResult(
                naming_rule_matched=False,
                parsed_fields=parsed_fields,
                error=f"Missing required parsed fields: {missing_required}",
            )

        return FilenameParseResult(naming_rule_matched=True, parsed_fields=parsed_fields)

    def _get_active_rule(self) -> dict[str, object] | None:
        rules = self.config.naming_rules.get("rules", {})
        if not isinstance(rules, dict):
            return None
        active_rule = rules.get(self.config.active_naming_rule)
        return active_rule if isinstance(active_rule, dict) else None
