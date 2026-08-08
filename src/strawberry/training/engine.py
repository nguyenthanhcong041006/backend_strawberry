from __future__ import annotations

import contextlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneGroupOut
from torch import nn
from torch.utils.data import DataLoader

from .config import MODEL_SPECS, TrainingConfig
from .data import (
    StrawberrySequenceDataset,
    build_sequence_table,
    compute_env_stats,
    fit_target_scaler,
    inverse_scale_targets,
    load_metadata,
    scale_targets,
)
from .models import build_model


@dataclass(frozen=True)
class FoldSplit:
    model_key: str
    fold_name: str
    test_group: str
    val_group: str
    train_groups: tuple[str, ...]
    val_sequence_df: pd.DataFrame
    train_sequence_df: pd.DataFrame
    test_sequence_df: pd.DataFrame
    train_frame_df: pd.DataFrame


@dataclass(frozen=True)
class FoldResult:
    model_key: str
    fold_name: str
    test_group: str
    val_group: str
    best_epoch: int
    best_val_mae: float
    test_mae: float
    test_rmse: float
    test_r2: float
    test_smape: float
    checkpoint_path: str
    best_history_path: str
    prediction_path: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()


def _grad_scaler(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.amp.GradScaler("cuda", enabled=True)
    return None


def _safe_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-6)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-8 else 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "smape": _safe_smape(y_true, y_pred),
        "median_ae": float(np.median(np.abs(residuals))),
    }


def _choose_val_group(train_groups: list[str], outer_index: int) -> str:
    ordered = sorted(train_groups)
    if not ordered:
        raise ValueError("Cannot choose a validation group from an empty set")
    return ordered[outer_index % len(ordered)]


def build_loocv_splits(sequence_df: pd.DataFrame, model_key: str) -> list[FoldSplit]:
    fruit_ids = sorted(sequence_df["fruit_id"].astype(str).unique())
    logo = LeaveOneGroupOut()
    groups = sequence_df["fruit_id"].astype(str).to_numpy()
    folds: list[FoldSplit] = []
    for outer_index, (_, test_idx) in enumerate(logo.split(sequence_df, groups=groups)):
        test_sequence_df = sequence_df.iloc[test_idx].reset_index(drop=True)
        test_group = str(test_sequence_df["fruit_id"].iloc[0])
        remaining = sequence_df[sequence_df["fruit_id"].astype(str) != test_group].reset_index(drop=True)
        remaining_groups = sorted(remaining["fruit_id"].astype(str).unique())
        val_group = _choose_val_group(remaining_groups, outer_index)
        val_sequence_df = remaining[remaining["fruit_id"].astype(str) == val_group].reset_index(drop=True)
        train_sequence_df = remaining[remaining["fruit_id"].astype(str) != val_group].reset_index(drop=True)
        train_frame_df = pd.DataFrame()
        folds.append(
            FoldSplit(
                model_key=model_key,
                fold_name=f"holdout_{test_group}",
                test_group=test_group,
                val_group=val_group,
                train_groups=tuple(train_sequence_df["fruit_id"].astype(str).unique().tolist()),
                val_sequence_df=val_sequence_df,
                train_sequence_df=train_sequence_df,
                test_sequence_df=test_sequence_df,
                train_frame_df=train_frame_df,
            )
        )
    return folds


def _make_loader(
    sequence_df: pd.DataFrame,
    *,
    image_size: int,
    env_stats,
    train: bool,
    batch_size: int,
    num_workers: int,
    image_mean: Iterable[float],
    image_std: Iterable[float],
) -> DataLoader:
    dataset = StrawberrySequenceDataset(
        sequence_df,
        image_size=image_size,
        env_stats=env_stats,
        train=train,
        image_mean=image_mean,
        image_std=image_std,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _sequence_loss(
    model: nn.Module,
    batch: dict[str, object],
    *,
    device: torch.device,
    target_center: float,
    target_scale: float,
    criterion: nn.Module,
    amp_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch["images"].to(device)
    env = batch["env"].to(device)
    target = batch["target"].to(device)
    target_scaled = (target - target_center) / target_scale
    with _autocast(device, amp_enabled):
        pred_scaled = model(images, env)
        loss = criterion(pred_scaled, target_scaled)
    return loss, pred_scaled.detach(), target.detach()


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    target_center: float,
    target_scale: float,
    criterion: nn.Module,
    amp_enabled: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    losses: list[float] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            loss, pred_scaled, target = _sequence_loss(
                model,
                batch,
                device=device,
                target_center=target_center,
                target_scale=target_scale,
                criterion=criterion,
                amp_enabled=amp_enabled,
            )
            losses.append(float(loss.item()))
            pred = inverse_scale_targets(pred_scaled.cpu().numpy(), {"center": target_center, "scale": target_scale})
            actual = target.cpu().numpy()
            y_true.append(actual)
            y_pred.append(pred)
            for idx in range(len(pred)):
                rows.append(
                    {
                        "sample_id": batch["sample_id"][idx],
                        "fruit_id": batch["fruit_id"][idx],
                        "split": batch["split"][idx],
                        "start_timestamp": batch["start_timestamp"][idx],
                        "end_timestamp": batch["end_timestamp"][idx],
                        "actual_rul_hours": float(actual[idx]),
                        "predicted_rul_hours": float(np.clip(pred[idx], 0.0, None)),
                        "error_hours": float(actual[idx] - pred[idx]),
                    }
                )
    if y_true:
        y_true_arr = np.concatenate(y_true, axis=0)
        y_pred_arr = np.concatenate(y_pred, axis=0)
        metrics = compute_metrics(y_true_arr, y_pred_arr)
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    else:
        metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "smape": 0.0, "median_ae": 0.0, "loss": 0.0}
    return metrics, pd.DataFrame(rows)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    target_center: float,
    target_scale: float,
    criterion: nn.Module,
    amp_enabled: bool,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, pred_scaled, target = _sequence_loss(
            model,
            batch,
            device=device,
            target_center=target_center,
            target_scale=target_scale,
            criterion=criterion,
            amp_enabled=amp_enabled,
        )
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        losses.append(float(loss.item()))
        pred = inverse_scale_targets(pred_scaled.cpu().numpy(), {"center": target_center, "scale": target_scale})
        actual = target.cpu().numpy()
        y_true.append(actual)
        y_pred.append(pred)
    y_true_arr = np.concatenate(y_true, axis=0) if y_true else np.empty(0, dtype=np.float32)
    y_pred_arr = np.concatenate(y_pred, axis=0) if y_pred else np.empty(0, dtype=np.float32)
    metrics = compute_metrics(y_true_arr, y_pred_arr) if len(y_true_arr) else {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "smape": 0.0, "median_ae": 0.0}
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def _build_run_paths(project_root: Path, config: TrainingConfig, model_key: str, fold_name: str) -> tuple[Path, Path, Path]:
    run_root = config.artifact_root_path(project_root) / config.run_name / model_key / fold_name
    checkpoint_root = config.checkpoint_root_path(project_root) / config.run_name / model_key / fold_name
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    return run_root, checkpoint_root, run_root / "predictions.csv"


def train_single_fold(
    model_key: str,
    fold_split: FoldSplit,
    *,
    config: TrainingConfig,
    project_root: Path,
    sequence_df: pd.DataFrame,
) -> FoldResult:
    device = _device()
    run_root, checkpoint_root, prediction_path = _build_run_paths(project_root, config, model_key, fold_split.fold_name)
    split_root = project_root / "data" / "03_split" / "strawberry"
    metadata_path = split_root / "metadata.csv"
    train_frame_df = load_metadata(metadata_path, project_root=project_root, split_root=split_root)
    train_frame_df = train_frame_df[train_frame_df["fruit_id"].astype(str).isin(fold_split.train_groups)].reset_index(drop=True)
    env_stats = compute_env_stats(train_frame_df)
    target_scaler = fit_target_scaler(fold_split.train_sequence_df["target_rul_hours"].astype(float).tolist())

    train_loader = _make_loader(
        fold_split.train_sequence_df,
        image_size=config.image_size,
        env_stats=env_stats,
        train=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_mean=config.image_mean,
        image_std=config.image_std,
    )
    val_loader = _make_loader(
        fold_split.val_sequence_df,
        image_size=config.image_size,
        env_stats=env_stats,
        train=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_mean=config.image_mean,
        image_std=config.image_std,
    )
    test_loader = _make_loader(
        fold_split.test_sequence_df,
        image_size=config.image_size,
        env_stats=env_stats,
        train=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_mean=config.image_mean,
        image_std=config.image_std,
    )

    model = build_model(
        model_key,
        hidden_size=config.hidden_size,
        env_hidden_size=config.env_hidden_size,
        dropout=config.dropout,
        backbone_dropout=config.backbone_dropout,
        temporal_pooling=config.temporal_pooling,
        fusion_mode=config.fusion_mode,
        pretrained_backbone=config.pretrained_backbone,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.SmoothL1Loss(beta=1.0)
    scaler = _grad_scaler(device, config.amp)
    target_center = float(target_scaler["center"])
    target_scale = float(target_scaler["scale"])

    best_val_mae = float("inf")
    best_epoch = 0
    best_path = checkpoint_root / "best_model.pt"
    last_path = checkpoint_root / "last_model.pt"
    history_rows: list[dict[str, object]] = []
    patience_left = config.patience

    for epoch in range(1, config.epochs + 1):
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            target_center=target_center,
            target_scale=target_scale,
            criterion=criterion,
            amp_enabled=config.amp,
            grad_clip=config.grad_clip,
        )
        val_metrics, _ = _evaluate_loader(
            model,
            val_loader,
            device=device,
            target_center=target_center,
            target_scale=target_scale,
            criterion=criterion,
            amp_enabled=config.amp,
        )
        scheduler.step(val_metrics["mae"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "val_smape": val_metrics["smape"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        checkpoint = {
            "model_key": model_key,
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "target_scaler": target_scaler,
            "env_stats": {"mean": env_stats.mean.tolist(), "std": env_stats.std.tolist()},
            "epoch": epoch,
            "fold_name": fold_split.fold_name,
            "test_group": fold_split.test_group,
            "val_group": fold_split.val_group,
            "val_mae": val_metrics["mae"],
        }
        torch.save(checkpoint, last_path)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            torch.save(checkpoint, best_path)
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics, prediction_df = _evaluate_loader(
        model,
        test_loader,
        device=device,
        target_center=target_center,
        target_scale=target_scale,
        criterion=criterion,
        amp_enabled=config.amp,
    )
    prediction_df.to_csv(prediction_path, index=False)

    history_df = pd.DataFrame(history_rows)
    history_path = run_root / "history.csv"
    history_df.to_csv(history_path, index=False)

    metrics_path = run_root / "metrics.json"
    metrics_payload = {
        "model_key": model_key,
        "fold_name": fold_split.fold_name,
        "test_group": fold_split.test_group,
        "val_group": fold_split.val_group,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_metrics": test_metrics,
        "target_scaler": target_scaler,
        "env_stats": {"mean": env_stats.mean.tolist(), "std": env_stats.std.tolist()},
        "checkpoint_path": str(best_path),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, default=str)

    return FoldResult(
        model_key=model_key,
        fold_name=fold_split.fold_name,
        test_group=fold_split.test_group,
        val_group=fold_split.val_group,
        best_epoch=best_epoch,
        best_val_mae=float(best_val_mae),
        test_mae=float(test_metrics["mae"]),
        test_rmse=float(test_metrics["rmse"]),
        test_r2=float(test_metrics["r2"]),
        test_smape=float(test_metrics["smape"]),
        checkpoint_path=str(best_path),
        best_history_path=str(history_path),
        prediction_path=str(prediction_path),
    )


def train_loocv_run(
    config: TrainingConfig,
    *,
    project_root: Path | None = None,
    model_keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    root = project_root or Path(__file__).resolve().parents[3]
    set_seed(config.seed)
    split_root = root / "data" / "03_split" / "strawberry"
    metadata_path = split_root / "metadata.csv"
    frame_df = load_metadata(metadata_path, project_root=root, split_root=split_root)
    sequence_df = build_sequence_table(
        frame_df,
        config.seq_len,
        max_gap_hours=config.max_gap_hours,
        pre_eol_only=True,
    )
    if sequence_df.empty:
        raise RuntimeError("No sequence windows were generated from the strawberry metadata.")

    selected_model_keys = [key.upper() for key in (model_keys or config.model_keys)]
    results: list[dict[str, object]] = []
    run_root = config.artifact_root_path(root) / config.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    sequence_df.to_csv(run_root / "sequence_index.csv", index=False)

    for model_key in selected_model_keys:
        for fold_split in build_loocv_splits(sequence_df, model_key):
            fold_result = train_single_fold(
                model_key,
                fold_split,
                config=config,
                project_root=root,
                sequence_df=sequence_df,
            )
            results.append(
                {
                    "model_key": fold_result.model_key,
                    "fold_name": fold_result.fold_name,
                    "test_group": fold_result.test_group,
                    "val_group": fold_result.val_group,
                    "best_epoch": fold_result.best_epoch,
                    "best_val_mae": fold_result.best_val_mae,
                    "test_mae": fold_result.test_mae,
                    "test_rmse": fold_result.test_rmse,
                    "test_r2": fold_result.test_r2,
                    "test_smape": fold_result.test_smape,
                    "checkpoint_path": fold_result.checkpoint_path,
                    "history_path": fold_result.best_history_path,
                    "prediction_path": fold_result.prediction_path,
                }
            )

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(run_root / "fold_results.csv", index=False)
    if not summary_df.empty:
        aggregated = summary_df.groupby("model_key").agg(
            folds=("fold_name", "count"),
            val_mae_mean=("best_val_mae", "mean"),
            val_mae_std=("best_val_mae", "std"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_r2_mean=("test_r2", "mean"),
            test_smape_mean=("test_smape", "mean"),
        ).reset_index()
        aggregated.to_csv(run_root / "model_summary.csv", index=False)
        with (run_root / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_name": config.run_name,
                    "seq_len": config.seq_len,
                    "fusion_mode": config.fusion_mode,
                    "temporal_pooling": config.temporal_pooling,
                    "rows": summary_df.to_dict(orient="records"),
                },
                handle,
                indent=2,
                default=str,
            )
    return summary_df
