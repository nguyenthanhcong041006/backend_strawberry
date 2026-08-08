"""Notebook helpers for the strawberry-only RUL lab.

This module is intentionally a thin compatibility layer over
``src/strawberry/training`` so notebooks and training code build sequences,
features, and LOOCV searches from the same source of truth.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from strawberry.training.config import load_training_config
from strawberry.training.data import (
    DEFAULT_METADATA_PATH,
    DEFAULT_SPLIT_ROOT,
    build_sequence_feature_table,
    build_sequence_table,
    load_metadata,
)
from strawberry.training.lab_search import run_lab_search


SPLIT_ROOT = DEFAULT_SPLIT_ROOT
METADATA_PATH = DEFAULT_METADATA_PATH
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLITS = ("train", "val", "test")


def load_all_labels(
    metadata_path: Path = METADATA_PATH,
    split_root: Path = SPLIT_ROOT,
) -> pd.DataFrame:
    """Load consolidated strawberry metadata with inferred train/val/test split."""

    return load_metadata(metadata_path, project_root=PROJECT_ROOT, split_root=split_root)


def load_split_labels(
    split: str,
    metadata_path: Path = METADATA_PATH,
    split_root: Path = SPLIT_ROOT,
) -> pd.DataFrame:
    """Load labels for one split from the consolidated metadata."""

    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    labels = load_all_labels(metadata_path=metadata_path, split_root=split_root)
    return labels[labels["split"] == split].reset_index(drop=True)


def sequence_counts(
    seq_lens: Iterable[int],
    metadata_path: Path = METADATA_PATH,
    split_root: Path = SPLIT_ROOT,
    max_gap_hours: float = 1.0,
) -> pd.DataFrame:
    """Count leakage-safe sequence windows per fruit and split."""

    labels = load_all_labels(metadata_path=metadata_path, split_root=split_root)
    rows: list[dict[str, object]] = []
    for seq_len in seq_lens:
        sequence_df = build_sequence_table(
            labels,
            int(seq_len),
            max_gap_hours=max_gap_hours,
            pre_eol_only=True,
        )
        if sequence_df.empty:
            continue
        counts = (
            sequence_df.groupby(["split", "fruit_id"])
            .size()
            .reset_index(name="sequences")
        )
        counts["seq_len"] = int(seq_len)
        rows.extend(counts.to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_features_for_seq_len(
    seq_len: int,
    metadata_path: Path = METADATA_PATH,
    max_gap_hours: float = 1.0,
) -> pd.DataFrame:
    """Build the same low-cost image/env feature table used by lab search."""

    labels = load_all_labels(metadata_path=metadata_path)
    return build_sequence_feature_table(
        labels,
        int(seq_len),
        max_gap_hours=max_gap_hours,
        pre_eol_only=True,
    )


def run_ml_baseline_sweep(
    config_path: Path = PROJECT_ROOT / "configs" / "strawberry_training.json",
) -> pd.DataFrame:
    """Run the current LOOCV sequence/fusion lab sweep."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return run_lab_search(config_path)


def current_large_model_metrics(
    run_root: Path = PROJECT_ROOT / "output" / "runs" / "strawberry" / "strawberry_loocv_seq3_late_env_branch",
) -> pd.DataFrame:
    """Return model summary for the new LOOCV run if it exists."""

    summary_path = run_root / "model_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    return pd.read_csv(summary_path)


def selected_training_config(
    config_path: Path = PROJECT_ROOT / "configs" / "strawberry_training.json",
) -> dict:
    """Read the selected large-model config as a notebook-friendly dict."""

    return load_training_config(config_path).to_dict()


if __name__ == "__main__":
    result = run_ml_baseline_sweep()
    print(result.head(12).to_string(index=False))
