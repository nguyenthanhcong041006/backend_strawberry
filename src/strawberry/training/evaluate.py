from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def summarize_run(run_root: str | Path) -> dict[str, object]:
    root = Path(run_root)
    fold_results_path = root / "fold_results.csv"
    if not fold_results_path.exists():
        raise FileNotFoundError(f"Missing fold_results.csv: {fold_results_path}")

    fold_results = pd.read_csv(fold_results_path)
    summary = {
        "run_root": str(root),
        "folds": int(len(fold_results)),
        "model_summary": {},
    }
    if not fold_results.empty:
        model_summary = fold_results.groupby("model_key").agg(
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_r2_mean=("test_r2", "mean"),
            test_smape_mean=("test_smape", "mean"),
        )
        summary["model_summary"] = model_summary.reset_index().to_dict(orient="records")
        model_summary.to_csv(root / "evaluation_summary.csv")
    with (root / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return summary

