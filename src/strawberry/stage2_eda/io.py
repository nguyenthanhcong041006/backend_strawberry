from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


def find_project_root(start: Optional[Path] = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if (path / 'README.md').exists() and (path / 'data').exists():
            return path
    raise FileNotFoundError('Could not find project root containing README.md and data/')


def configure_paths(root: Optional[Path] = None) -> tuple[Path, Path, Path, Path]:
    project_root = find_project_root(root)
    manifest_dir = project_root / 'data' / '02_processed' / 'strawberry' / 'manifests_mock'
    report_dir = project_root / 'output' / 'reports' / 'eda'
    graph_dir = project_root / 'output' / 'graphs' / 'eda'
    report_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    return project_root, manifest_dir, report_dir, graph_dir


def display_path(path: Path, root: Optional[Path] = None) -> str:
    base = (root or find_project_root(path)).resolve()
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f'Warning: missing required CSV file: {path}')
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f'Error loading CSV {path}: {exc}')
        return pd.DataFrame()


def load_summary_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_report(lines: list[str], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / 'dataset_inventory.md'
    with output_path.open('w', encoding='utf-8') as f:
        f.write('\n'.join(lines).strip() + '\n')
    print(f'Saved report: {display_path(output_path)}')
