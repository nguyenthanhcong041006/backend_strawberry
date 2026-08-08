"""Configuration objects for Stage 1.5."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Stage15Config:
    """Project-relative configuration for the Stage 1.5 pipeline."""

    project_root: Path
    raw_roots: list[Path]
    image_extensions: list[str]
    output: dict[str, Any]
    logging: dict[str, Any]
    processing: dict[str, Any]
    naming_rules: dict[str, Any]
    numeric_crosscheck: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Stage15Config":
        """Load a YAML config file.

        PyYAML is used when available. The fallback parser supports the simple
        project config subset used by Stage 1.5: nested mappings and scalar
        lists.
        """
        config_file = Path(config_path)
        raw = _load_yaml(config_file)
        project_root = Path(raw.get("project_root", ".")).resolve()

        def resolve(path_value: str | Path) -> Path:
            path = Path(path_value)
            return path if path.is_absolute() else project_root / path

        return cls(
            project_root=project_root,
            raw_roots=[resolve(path_value) for path_value in raw.get("raw_roots", [])],
            image_extensions=[str(ext).lower() for ext in raw.get("image_extensions", [])],
            output=dict(raw.get("output", {})),
            logging=dict(raw.get("logging", {})),
            processing=dict(raw.get("processing", {})),
            naming_rules=dict(raw.get("naming_rules", {})),
            numeric_crosscheck=dict(raw.get("numeric_crosscheck", {})),
            config_path=config_file,
            raw=raw,
        )

    def resolve_path(self, path_value: str | Path) -> Path:
        """Resolve a path relative to the configured project root."""
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def active_naming_rule(self) -> str:
        """Return the active naming rule name from the config."""
        return str(self.naming_rules.get("active_rule", ""))


def _load_yaml(config_path: Path) -> dict[str, Any]:
    """Load YAML with PyYAML if available, otherwise use the local subset parser."""
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _parse_simple_yaml(config_path.read_text(encoding="utf-8"))

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return loaded


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by this project config."""
    lines = _prepare_yaml_lines(text)
    if not lines:
        return {}
    parsed, next_index = _parse_mapping(lines, 0, lines[0][0])
    if next_index != len(lines):
        raise ValueError("Could not parse the full YAML config.")
    return parsed


def _prepare_yaml_lines(text: str) -> list[tuple[int, str]]:
    prepared: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        prepared.append((indent, stripped))
    return prepared


def _parse_mapping(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation near: {content}")
        if content.startswith("- "):
            break
        if ":" not in content:
            raise ValueError(f"Expected key-value mapping near: {content}")

        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1

        if raw_value:
            result[key] = _parse_scalar(raw_value)
            continue

        if index >= len(lines) or lines[index][0] <= current_indent:
            result[key] = {}
            continue

        child_indent, child_content = lines[index]
        if child_content.startswith("- "):
            result[key], index = _parse_list(lines, index, child_indent)
        else:
            result[key], index = _parse_mapping(lines, index, child_indent)
    return result, index


def _parse_list(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected list indentation near: {content}")
        if not content.startswith("- "):
            break
        result.append(_parse_scalar(content[2:].strip()))
        index += 1
    return result, index


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value == "{}":
        return {}
    if value == "[]":
        return []
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
