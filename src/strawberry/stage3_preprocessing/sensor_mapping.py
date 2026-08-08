from __future__ import annotations

from pathlib import Path

import pandas as pd


def _resolve_path(path_text: str | Path, project_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = project_root / path
    return candidate


def load_sensor_frame(sensor_csv: Path, timestamp_column: str = "timestamp") -> pd.DataFrame:
    if not sensor_csv.exists():
        raise FileNotFoundError(f"Real sensor CSV not found: {sensor_csv}")

    sensor_df = pd.read_csv(sensor_csv)
    rename_map = {}
    if "humidity_rh" in sensor_df.columns and "humidity_pct" not in sensor_df.columns:
        rename_map["humidity_rh"] = "humidity_pct"
    if "temperature" in sensor_df.columns and "temperature_c" not in sensor_df.columns:
        rename_map["temperature"] = "temperature_c"
    sensor_df = sensor_df.rename(columns=rename_map)

    required_columns = {timestamp_column, "temperature_c", "humidity_pct"}
    missing = required_columns - set(sensor_df.columns)
    if missing:
        raise ValueError(
            f"Sensor CSV must contain columns {sorted(required_columns)}; missing {sorted(missing)}"
        )

    sensor_df = sensor_df.copy()
    sensor_df[timestamp_column] = pd.to_datetime(sensor_df[timestamp_column], errors="coerce")
    sensor_df["temperature_c"] = pd.to_numeric(sensor_df["temperature_c"], errors="coerce")
    sensor_df["humidity_pct"] = pd.to_numeric(sensor_df["humidity_pct"], errors="coerce")
    sensor_df = sensor_df.dropna(subset=[timestamp_column, "temperature_c", "humidity_pct"])
    sensor_df = sensor_df.sort_values(timestamp_column).drop_duplicates(
        subset=[timestamp_column], keep="last"
    )
    return sensor_df.reset_index(drop=True)


def map_sensor_to_labels(
    labels_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    tolerance: str | pd.Timedelta = "45min",
) -> pd.DataFrame:
    if timestamp_column not in labels_df.columns:
        raise ValueError(f"labels_df must contain a '{timestamp_column}' column")

    labels = labels_df.copy()
    labels[timestamp_column] = pd.to_datetime(labels[timestamp_column], errors="coerce")
    labels = labels.dropna(subset=[timestamp_column]).sort_values(timestamp_column).reset_index(drop=True)
    stale_environment_columns = [
        "temperature_c",
        "humidity_pct",
        "sensor_timestamp",
        "mapping_method",
        "mapping_delta_seconds",
        "environment_source",
        "sensor_status",
    ]
    labels = labels.drop(columns=[col for col in stale_environment_columns if col in labels.columns])

    sensor = sensor_df.copy().rename(columns={timestamp_column: "sensor_timestamp"})
    sensor = sensor.sort_values("sensor_timestamp").reset_index(drop=True)

    mapped = pd.merge_asof(
        labels,
        sensor[["sensor_timestamp", "temperature_c", "humidity_pct"]],
        left_on=timestamp_column,
        right_on="sensor_timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )

    mapped["mapping_method"] = "nearest_sensor_timestamp"
    mapped["environment_source"] = "real_sensor"
    mapped["mapping_delta_seconds"] = (
        mapped[timestamp_column] - mapped["sensor_timestamp"]
    ).abs().dt.total_seconds()
    mapped["sensor_match"] = mapped["sensor_timestamp"].notna()
    mapped.loc[~mapped["sensor_match"], ["temperature_c", "humidity_pct"]] = pd.NA
    mapped["sensor_status"] = mapped["sensor_match"].map({True: "matched", False: "missing"})
    mapped = mapped.drop(columns=["sensor_match"])
    return mapped


def remap_final_labels(
    project_root: Path,
    sensor_csv: str | Path,
    *,
    tolerance: str | pd.Timedelta = "45min",
) -> list[Path]:
    root = Path(project_root)
    final_dir = root / "data" / "02_processed" / "strawberry" / "final"
    sensor_path = _resolve_path(sensor_csv, root)
    sensor_df = load_sensor_frame(sensor_path)

    output_paths: list[Path] = []
    report_rows: list[pd.DataFrame] = []

    for labels_path in sorted(final_dir.glob("F*/labels.csv")):
        labels_df = pd.read_csv(labels_path)
        mapped_df = map_sensor_to_labels(labels_df, sensor_df, tolerance=tolerance)
        mapped_df.to_csv(labels_path, index=False)
        output_paths.append(labels_path)

        report_columns = [
            "fruit_id",
            "timestamp",
            "sensor_timestamp",
            "temperature_c",
            "humidity_pct",
            "mapping_method",
            "mapping_delta_seconds",
            "environment_source",
            "sensor_status",
        ]
        available = [column for column in report_columns if column in mapped_df.columns]
        report_rows.append(mapped_df[available])

    if report_rows:
        report_df = pd.concat(report_rows, ignore_index=True)
        report_df.to_csv(final_dir / "sensor_mapping_report.csv", index=False)

    return output_paths


def main(sensor_csv: str | Path, project_root: str | Path | None = None) -> None:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[3]
    remap_final_labels(root, sensor_csv)


if __name__ == "__main__":
    raise SystemExit(
        "Run this module from main_preprocessing.py or import remap_final_labels() with a real sensor CSV."
    )
