from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

from .config import TrainingConfig, load_training_config
from .data import build_sequence_feature_table, load_metadata


def _late_fusion_weight(y_true: np.ndarray, pred_img: np.ndarray, pred_env: np.ndarray) -> float:
    best_alpha = 0.8
    best_mae = float("inf")
    for alpha in np.linspace(0.0, 1.0, 11):
        fused = alpha * pred_img + (1.0 - alpha) * pred_env
        mae = float(mean_absolute_error(y_true, fused))
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)
    return best_alpha


def _fit_regressor(X_train: np.ndarray, y_train: np.ndarray) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=250,
        min_samples_leaf=12,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def run_lab_search(config_path: str | Path | None = None) -> pd.DataFrame:
    config = load_training_config(config_path)
    project_root = Path(__file__).resolve().parents[3]
    split_root = project_root / "data" / "03_split" / "strawberry"
    frame_df = load_metadata(split_root / "metadata.csv", project_root=project_root, split_root=split_root)

    results = []
    for seq_len in [3, 5, 8, 10, 12]:
        seq_df = build_sequence_feature_table(frame_df, seq_len, max_gap_hours=config.max_gap_hours, pre_eol_only=True)
        if seq_df.empty:
            continue
        groups = seq_df["fruit_id"].astype(str).to_numpy()
        logo = LeaveOneGroupOut()
        for fusion_mode in ["image_only", "early_concat", "late_env_branch"]:
            fold_mae = []
            for outer_index, (train_idx, test_idx) in enumerate(logo.split(seq_df, groups=groups)):
                train_df = seq_df.iloc[train_idx].reset_index(drop=True)
                test_df = seq_df.iloc[test_idx].reset_index(drop=True)
                train_groups = sorted(train_df["fruit_id"].astype(str).unique())
                val_group = train_groups[outer_index % len(train_groups)]
                fit_df = train_df[train_df["fruit_id"].astype(str) != val_group].reset_index(drop=True)
                val_df = train_df[train_df["fruit_id"].astype(str) == val_group].reset_index(drop=True)

                if fusion_mode == "image_only":
                    img_cols = [c for c in seq_df.columns if c.startswith("img_")]
                    model = _fit_regressor(fit_df[img_cols].to_numpy(dtype=np.float32), fit_df["target"].to_numpy(dtype=np.float32))
                    pred = model.predict(test_df[img_cols].to_numpy(dtype=np.float32))
                elif fusion_mode == "early_concat":
                    cols = [c for c in seq_df.columns if c.startswith("img_") or c.startswith("env_")]
                    model = _fit_regressor(fit_df[cols].to_numpy(dtype=np.float32), fit_df["target"].to_numpy(dtype=np.float32))
                    pred = model.predict(test_df[cols].to_numpy(dtype=np.float32))
                else:
                    img_cols = [c for c in seq_df.columns if c.startswith("img_")]
                    env_cols = [c for c in seq_df.columns if c.startswith("env_")]
                    img_model = _fit_regressor(fit_df[img_cols].to_numpy(dtype=np.float32), fit_df["target"].to_numpy(dtype=np.float32))
                    env_model = _fit_regressor(fit_df[env_cols].to_numpy(dtype=np.float32), fit_df["target"].to_numpy(dtype=np.float32))
                    alpha = _late_fusion_weight(
                        val_df["target"].to_numpy(dtype=np.float32),
                        img_model.predict(val_df[img_cols].to_numpy(dtype=np.float32)),
                        env_model.predict(val_df[env_cols].to_numpy(dtype=np.float32)),
                    )
                    pred = alpha * img_model.predict(test_df[img_cols].to_numpy(dtype=np.float32)) + (1.0 - alpha) * env_model.predict(test_df[env_cols].to_numpy(dtype=np.float32))
                fold_mae.append(float(mean_absolute_error(test_df["target"].to_numpy(dtype=np.float32), pred)))

            results.append(
                {
                    "seq_len": seq_len,
                    "fusion_mode": fusion_mode,
                    "fold_mae_mean": float(np.mean(fold_mae)) if fold_mae else float("nan"),
                    "fold_mae_std": float(np.std(fold_mae)) if fold_mae else float("nan"),
                }
            )

    result_df = pd.DataFrame(results).sort_values(["fold_mae_mean", "seq_len", "fusion_mode"]).reset_index(drop=True)
    results_dir = project_root / "notebooks" / "strawberry" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(results_dir / "lab_seq_fusion_search.csv", index=False)
    with (results_dir / "lab_seq_fusion_search.json").open("w", encoding="utf-8") as handle:
        json.dump(result_df.to_dict(orient="records"), handle, indent=2)
    return result_df


if __name__ == "__main__":
    run_lab_search()
