from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import LeaveOneGroupOut
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "data" / "03_split" / "strawberry"
DEFAULT_METADATA_PATH = DEFAULT_SPLIT_ROOT / "metadata.csv"
LEGACY_FINAL_METADATA_PATH = PROJECT_ROOT / "data" / "02_processed" / "strawberry" / "final" / "metadata.csv"


@dataclass(frozen=True)
class EnvStats:
    mean: np.ndarray
    std: np.ndarray


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = project_root / path
    return candidate


def _resolve_image_path(
    path_text: object,
    *,
    project_root: Path,
    split_root: Path,
    split: str | None = None,
    fruit_id: str | None = None,
) -> str:
    if pd.isna(path_text):
        return ""
    raw = Path(str(path_text))
    if raw.is_absolute():
        return str(raw)

    candidates: list[Path] = []
    if split:
        split_dir = split_root / split
        candidates.extend(
            [
                split_dir / raw,
                split_dir / "images" / raw,
            ]
        )
        if fruit_id:
            candidates.append(split_dir / "images" / fruit_id / raw.name)
    candidates.extend(
        [
            split_root / raw,
            project_root / raw,
            LEGACY_FINAL_METADATA_PATH.parent / raw,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0] if candidates else project_root / raw)


def infer_split_map(split_root: Path = DEFAULT_SPLIT_ROOT) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not split_root.exists():
        return mapping
    for split_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        labels_path = split_dir / "labels.csv"
        if not labels_path.exists():
            continue
        labels_df = pd.read_csv(labels_path, usecols=["fruit_id"])
        for fruit_id in labels_df["fruit_id"].astype(str).dropna().unique():
            mapping[str(fruit_id)] = split_dir.name
    if mapping:
        return mapping
    fallback = {"F01": "train", "F02": "train", "F03": "train", "F04": "train", "F05": "val", "F06": "test"}
    return fallback


def _split_labels_available(split_root: Path) -> bool:
    return any((split_root / split / "labels.csv").exists() for split in ("train", "val", "test"))


def _trust_split_environment(frame_df: pd.DataFrame) -> pd.DataFrame:
    frame_df = frame_df.copy()
    for column in ("temperature_c", "humidity_pct"):
        if column not in frame_df.columns:
            frame_df[column] = pd.NA
    has_env = frame_df[["temperature_c", "humidity_pct"]].notna().all(axis=1)
    if "environment_source" not in frame_df.columns:
        frame_df["environment_source"] = pd.NA
    if "sensor_status" not in frame_df.columns:
        frame_df["sensor_status"] = pd.NA
    frame_df.loc[has_env & frame_df["environment_source"].isna(), "environment_source"] = "real_sensor"
    frame_df.loc[has_env & frame_df["sensor_status"].isna(), "sensor_status"] = "matched"
    return frame_df


def _load_split_labels(
    *,
    project_root: Path,
    split_root: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        labels_path = split_root / split / "labels.csv"
        if not labels_path.exists():
            continue
        split_df = pd.read_csv(labels_path)
        split_df["split"] = split
        split_df = _trust_split_environment(split_df)
        if "image_path" in split_df.columns:
            split_df["abs_image_path"] = [
                _resolve_image_path(
                    value,
                    project_root=project_root,
                    split_root=split_root,
                    split=split,
                    fruit_id=str(fruit_id) if pd.notna(fruit_id) else None,
                )
                for value, fruit_id in zip(split_df["image_path"], split_df.get("fruit_id", pd.Series([None] * len(split_df))))
            ]
        frames.append(split_df)
    if not frames:
        raise FileNotFoundError(f"No split labels found under: {split_root}")
    return pd.concat(frames, ignore_index=True)


def load_metadata(
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    *,
    project_root: Path = PROJECT_ROOT,
    split_root: Path = DEFAULT_SPLIT_ROOT,
) -> pd.DataFrame:
    path = Path(metadata_path)
    split_root = Path(split_root)
    project_root = Path(project_root)
    if path == DEFAULT_METADATA_PATH and _split_labels_available(split_root):
        frame_df = _load_split_labels(project_root=project_root, split_root=split_root)
    else:
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        frame_df = pd.read_csv(path)
        if path.resolve().is_relative_to(split_root.resolve()):
            frame_df = _trust_split_environment(frame_df)
    if not path.exists():
        path = DEFAULT_METADATA_PATH
    if "timestamp" in frame_df.columns:
        frame_df["timestamp"] = pd.to_datetime(frame_df["timestamp"], errors="coerce")
    if "eol_timestamp" in frame_df.columns:
        frame_df["eol_timestamp"] = pd.to_datetime(frame_df["eol_timestamp"], errors="coerce")

    if "image_path" in frame_df.columns:
        frame_df["abs_image_path"] = [
            _resolve_image_path(
                value,
                project_root=project_root,
                split_root=split_root,
                split=str(split) if pd.notna(split) else None,
                fruit_id=str(fruit_id) if pd.notna(fruit_id) else None,
            )
            for value, split, fruit_id in zip(
                frame_df["image_path"],
                frame_df.get("split", pd.Series([None] * len(frame_df))),
                frame_df.get("fruit_id", pd.Series([None] * len(frame_df))),
            )
        ]

    if "fruit_id" in frame_df.columns and "split" not in frame_df.columns:
        split_map = infer_split_map(split_root)
        frame_df["split"] = frame_df["fruit_id"].astype(str).map(split_map)

    if "temperature_c" not in frame_df.columns:
        frame_df["temperature_c"] = pd.NA
    if "humidity_pct" not in frame_df.columns:
        frame_df["humidity_pct"] = pd.NA
    if "environment_source" not in frame_df.columns:
        frame_df["environment_source"] = "missing_sensor_provenance"
    if "sensor_status" not in frame_df.columns:
        frame_df["sensor_status"] = "missing_sensor_provenance"
    real_sensor_mask = frame_df["environment_source"].astype(str).str.lower().eq("real_sensor")
    if "sensor_status" in frame_df.columns:
        real_sensor_mask &= frame_df["sensor_status"].astype(str).str.lower().eq("matched")
    frame_df.loc[~real_sensor_mask, ["temperature_c", "humidity_pct"]] = pd.NA
    if "label_status" not in frame_df.columns:
        frame_df["label_status"] = "approved"

    if "time_gap_hours" not in frame_df.columns and "fruit_id" in frame_df.columns:
        frame_df = frame_df.sort_values(["fruit_id", "timestamp"]).reset_index(drop=True)
        frame_df["time_gap_hours"] = (
            frame_df.groupby("fruit_id")["timestamp"].diff().dt.total_seconds().div(3600).fillna(0.0)
        )
    if "elapsed_hours" not in frame_df.columns and "fruit_id" in frame_df.columns:
        first_ts = frame_df.groupby("fruit_id")["timestamp"].transform("min")
        frame_df["elapsed_hours"] = (
            frame_df["timestamp"] - first_ts
        ).dt.total_seconds().div(3600).fillna(0.0)

    return frame_df.sort_values(["fruit_id", "timestamp"]).reset_index(drop=True)


def make_image_transform(image_size: int = 224, *, train: bool = False) -> T.Compose:
    base: list = [T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)]
    if train:
        base = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAutocontrast(p=0.25),
            T.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.04, hue=0.02),
            *base,
        ]
    base.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return T.Compose(base)


def compute_env_stats(frame_df: pd.DataFrame) -> EnvStats:
    values = frame_df[["temperature_c", "humidity_pct"]].apply(pd.to_numeric, errors="coerce")
    mean = values.mean(skipna=True).to_numpy(dtype=np.float32)
    std = values.std(skipna=True).to_numpy(dtype=np.float32)
    std = np.where(np.isnan(std) | (std < 1e-6), 1.0, std)
    mean = np.where(np.isnan(mean), 0.0, mean)
    return EnvStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def fit_target_scaler(targets: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(targets), dtype=np.float32)
    if array.size == 0:
        return {"center": 0.0, "scale": 1.0}
    center = float(np.median(array))
    q1 = float(np.quantile(array, 0.25))
    q3 = float(np.quantile(array, 0.75))
    iqr = max(q3 - q1, 1e-6)
    scale = iqr / 1.349
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = float(array.std() if array.std() > 1e-6 else 1.0)
    return {"center": center, "scale": float(scale)}


def scale_targets(targets: np.ndarray, target_scaler: dict[str, float]) -> np.ndarray:
    center = target_scaler.get("center", 0.0)
    scale = target_scaler.get("scale", 1.0) or 1.0
    return (targets - center) / scale


def inverse_scale_targets(targets: np.ndarray, target_scaler: dict[str, float]) -> np.ndarray:
    center = target_scaler.get("center", 0.0)
    scale = target_scaler.get("scale", 1.0) or 1.0
    return targets * scale + center


def _session_ids(timestamps: pd.Series, max_gap_hours: float) -> pd.Series:
    gaps = timestamps.diff().dt.total_seconds().div(3600).fillna(0.0)
    return gaps.gt(max_gap_hours).cumsum().astype(int)


def build_sequence_table(
    frame_df: pd.DataFrame,
    seq_len: int,
    *,
    max_gap_hours: float = 1.0,
    pre_eol_only: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame_df.empty:
        return pd.DataFrame(rows)

    required_columns = {"fruit_id", "timestamp", "image_path", "abs_image_path", "rul_hours"}
    missing = required_columns - set(frame_df.columns)
    if missing:
        raise ValueError(f"frame_df is missing required columns: {sorted(missing)}")

    for fruit_id, fruit_df in frame_df.groupby("fruit_id", sort=True):
        fruit_df = fruit_df.sort_values("timestamp").reset_index(drop=True)
        if pre_eol_only and "eol_timestamp" in fruit_df.columns:
            fruit_df = fruit_df[fruit_df["timestamp"] <= fruit_df["eol_timestamp"]].reset_index(drop=True)
        if fruit_df.empty:
            continue
        fruit_df = fruit_df.assign(_session_id=_session_ids(fruit_df["timestamp"], max_gap_hours))

        for session_id, session_df in fruit_df.groupby("_session_id", sort=True):
            session_df = session_df.reset_index(drop=True)
            if len(session_df) < seq_len:
                continue
            for start in range(0, len(session_df) - seq_len + 1):
                window = session_df.iloc[start : start + seq_len]
                start_ts = window.iloc[0]["timestamp"]
                end_ts = window.iloc[-1]["timestamp"]
                gaps = window["time_gap_hours"].to_numpy(dtype=np.float32) if "time_gap_hours" in window.columns else np.zeros(seq_len, dtype=np.float32)
                temp_series = pd.to_numeric(window["temperature_c"], errors="coerce")
                hum_series = pd.to_numeric(window["humidity_pct"], errors="coerce")
                temp_missing = temp_series.isna().to_numpy(dtype=np.float32)
                hum_missing = hum_series.isna().to_numpy(dtype=np.float32)

                rows.append(
                    {
                        "sample_id": f"{fruit_id}_s{int(session_id):02d}_{start:04d}",
                        "fruit_id": str(fruit_id),
                        "split": window.iloc[-1].get("split", None),
                        "seq_len": int(seq_len),
                        "session_id": int(session_id),
                        "start_index": int(start),
                        "start_timestamp": start_ts,
                        "end_timestamp": end_ts,
                        "target_rul_hours": float(window.iloc[-1]["rul_hours"]),
                        "image_paths": tuple(window["abs_image_path"].astype(str).tolist()),
                        "temperature_seq": tuple(temp_series.fillna(0.0).astype(np.float32).tolist()),
                        "humidity_seq": tuple(hum_series.fillna(0.0).astype(np.float32).tolist()),
                        "temp_missing_seq": tuple(temp_missing.tolist()),
                        "humidity_missing_seq": tuple(hum_missing.tolist()),
                        "gap_hours_max": float(np.max(gaps)) if len(gaps) else 0.0,
                        "gap_hours_mean": float(np.mean(gaps)) if len(gaps) else 0.0,
                        "elapsed_hours_end": float(window.iloc[-1].get("elapsed_hours", 0.0)),
                    }
                )

    return pd.DataFrame(rows)


def sequence_group_splits(sequence_df: pd.DataFrame) -> LeaveOneGroupOut:
    if "fruit_id" not in sequence_df.columns:
        raise ValueError("sequence_df must contain fruit_id")
    return LeaveOneGroupOut()


@lru_cache(maxsize=8192)
def extract_frame_features(image_path: str) -> np.ndarray:
    img = Image.open(Path(image_path)).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    gray = arr.mean(axis=2)
    saturation_proxy = arr.std(axis=2)
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    grad_mean = float((grad_x.mean() + grad_y.mean()) / 2.0)
    grad_std = float(np.sqrt((grad_x.std() ** 2 + grad_y.std() ** 2) / 2.0))
    return np.array(
        [
            red.mean(),
            green.mean(),
            blue.mean(),
            red.std(),
            green.std(),
            blue.std(),
            gray.mean(),
            gray.std(),
            saturation_proxy.mean(),
            arr.max(axis=2).mean(),
            grad_mean,
            grad_std,
        ],
        dtype=np.float32,
    )


def _slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=np.float32)
    x = x - x.mean()
    denom = float((x ** 2).sum())
    if denom <= 0:
        return 0.0
    centered = values.astype(np.float32) - values.mean()
    return float((x * centered).sum() / denom)


def _sequence_summary(values: np.ndarray, names: list[str], prefix: str) -> dict[str, float]:
    summary: dict[str, float] = {}
    for idx, name in enumerate(names):
        series = values[:, idx].astype(np.float32)
        summary[f"{prefix}{name}_first"] = float(series[0])
        summary[f"{prefix}{name}_last"] = float(series[-1])
        summary[f"{prefix}{name}_mean"] = float(series.mean())
        summary[f"{prefix}{name}_std"] = float(series.std())
        summary[f"{prefix}{name}_min"] = float(series.min())
        summary[f"{prefix}{name}_max"] = float(series.max())
        summary[f"{prefix}{name}_delta"] = float(series[-1] - series[0])
        summary[f"{prefix}{name}_slope"] = _slope(series)
    return summary


def build_sequence_feature_table(
    frame_df: pd.DataFrame,
    seq_len: int,
    *,
    max_gap_hours: float = 1.0,
    pre_eol_only: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    sequence_df = build_sequence_table(
        frame_df,
        seq_len,
        max_gap_hours=max_gap_hours,
        pre_eol_only=pre_eol_only,
    )
    for _, row in sequence_df.iterrows():
        image_paths = list(row["image_paths"])
        image_features = np.stack([extract_frame_features(path) for path in image_paths], axis=0)
        env_values = np.column_stack(
            [
                np.asarray(row["temperature_seq"], dtype=np.float32),
                np.asarray(row["humidity_seq"], dtype=np.float32),
            ]
        )
        feature_row: dict[str, float | str | int] = {
            "sample_id": str(row["sample_id"]),
            "fruit_id": str(row["fruit_id"]),
            "split": str(row.get("split", "")),
            "seq_len": int(row["seq_len"]),
            "target": float(row["target_rul_hours"]),
        }
        feature_row.update(_sequence_summary(image_features, [
            "r_mean",
            "g_mean",
            "b_mean",
            "r_std",
            "g_std",
            "b_std",
            "gray_mean",
            "gray_std",
            "sat_mean",
            "val_mean",
            "grad_mean",
            "grad_std",
        ], "img_"))
        feature_row.update(_sequence_summary(env_values, ["temperature_c", "humidity_pct"], "env_"))
        rows.append(feature_row)
    return pd.DataFrame(rows)


class StrawberrySequenceDataset(Dataset):
    def __init__(
        self,
        sequence_df: pd.DataFrame,
        *,
        image_size: int = 224,
        env_stats: EnvStats | None = None,
        train: bool = False,
        image_mean: Iterable[float] = (0.485, 0.456, 0.406),
        image_std: Iterable[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.sequence_df = sequence_df.reset_index(drop=True)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.env_stats = env_stats or EnvStats(
            mean=np.zeros(2, dtype=np.float32),
            std=np.ones(2, dtype=np.float32),
        )
        self.image_transform = self._build_image_transform(image_mean, image_std)

    def _build_image_transform(self, image_mean: Iterable[float], image_std: Iterable[float]) -> T.Compose:
        ops: list = []
        if self.train:
            ops.extend(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomAutocontrast(p=0.25),
                    T.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.04, hue=0.02),
                ]
            )
        ops.extend(
            [
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=list(image_mean), std=list(image_std)),
            ]
        )
        return T.Compose(ops)

    def __len__(self) -> int:
        return len(self.sequence_df)

    def _load_image(self, path_text: str) -> torch.Tensor:
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"Missing sequence image: {path}")
        image = Image.open(path).convert("RGB")
        return self.image_transform(image)

    def _normalize_env(self, temp: np.ndarray, hum: np.ndarray, temp_missing: np.ndarray, hum_missing: np.ndarray) -> np.ndarray:
        values = np.stack([temp, hum], axis=1).astype(np.float32)
        valid = np.stack([1.0 - temp_missing, 1.0 - hum_missing], axis=1).astype(np.float32)
        normalized = (values - self.env_stats.mean) / self.env_stats.std
        normalized = np.where(valid > 0, normalized, 0.0)
        return np.concatenate([normalized, temp_missing[:, None], hum_missing[:, None]], axis=1).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.sequence_df.iloc[index]
        image_seq = torch.stack([self._load_image(path) for path in row["image_paths"]], dim=0)
        temp = np.asarray(row["temperature_seq"], dtype=np.float32)
        hum = np.asarray(row["humidity_seq"], dtype=np.float32)
        temp_missing = np.asarray(row["temp_missing_seq"], dtype=np.float32)
        hum_missing = np.asarray(row["humidity_missing_seq"], dtype=np.float32)
        env_seq = self._normalize_env(temp, hum, temp_missing, hum_missing)
        return {
            "images": image_seq,
            "env": torch.from_numpy(env_seq),
            "target": torch.tensor(float(row["target_rul_hours"]), dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
            "fruit_id": str(row["fruit_id"]),
            "split": str(row.get("split", "")),
            "start_timestamp": pd.Timestamp(row["start_timestamp"]).isoformat(),
            "end_timestamp": pd.Timestamp(row["end_timestamp"]).isoformat(),
        }
