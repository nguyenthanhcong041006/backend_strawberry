from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEGMENTED_DIR = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "segmented"
DEFAULT_FIRMNESS_CSV = PROJECT_ROOT / "data" / "01_raw" / "avocado" / "hardness" / "hardness.csv"
DEFAULT_ENV_CSV = PROJECT_ROOT / "data" / "01_raw" / "avocado" / "th10s_readings.csv"
DEFAULT_FEATURES_CSV = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "eda" / "features" / "avocado_features.csv"
DEFAULT_EOL_CSV = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "labels" / "eol_annotations.csv"
DEFAULT_EXCLUSIONS_CSV = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "labels" / "frame_exclusions.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "rul_inspection"
TIMESTAMP_PATTERN = re.compile(r"webcam_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_fruit_(\d+)\.png$")


@dataclass(frozen=True)
class FruitFrame:
    fruit_id: str
    timestamp: pd.Timestamp
    image_path: Path


def parse_frame_timestamp(path: Path) -> pd.Timestamp:
    match = TIMESTAMP_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse timestamp from {path.name}")
    date_text, time_text, _fruit_num = match.groups()
    return pd.Timestamp(datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H-%M-%S"))


def load_frames(segmented_dir: Path) -> dict[str, list[FruitFrame]]:
    frames_by_fruit: dict[str, list[FruitFrame]] = {}
    for fruit_dir in sorted(segmented_dir.glob("F*")):
        if not fruit_dir.is_dir():
            continue
        frames: list[FruitFrame] = []
        for image_path in sorted(fruit_dir.glob("*.png")):
            try:
                frames.append(
                    FruitFrame(
                        fruit_id=fruit_dir.name,
                        timestamp=parse_frame_timestamp(image_path),
                        image_path=image_path,
                    )
                )
            except ValueError:
                continue
        if frames:
            frames_by_fruit[fruit_dir.name] = sorted(frames, key=lambda row: row.timestamp)
    return frames_by_fruit


def load_firmness(firmness_csv: Path) -> pd.DataFrame:
    if not firmness_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(firmness_csv)
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    return df


def load_environment(env_csv: Path) -> pd.DataFrame:
    if not env_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(env_csv)
    df["datetime"] = pd.to_datetime(df["timestamp"].str[:19])
    return df.sort_values("datetime").reset_index(drop=True)


def load_features(features_csv: Path) -> pd.DataFrame:
    if not features_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(features_csv)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["fruit_id", "timestamp"]).reset_index(drop=True)


def load_eol_annotations(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "eol_timestamp" in df.columns:
        df["eol_timestamp"] = pd.to_datetime(df["eol_timestamp"])
    return df


def load_exclusions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for column in ["start_timestamp", "end_timestamp", "updated_at"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def firmness_column(fruit_id: str) -> str:
    return f"fruit_{int(fruit_id[1:])}"


def firmness_context(firmness_df: pd.DataFrame, fruit_id: str, timestamp: pd.Timestamp) -> dict[str, object]:
    if firmness_df.empty:
        return {
            "firmness_latest": "",
            "firmness_date": "",
            "first_zero_date": "",
            "first_le5_date": "",
        }
    col = firmness_column(fruit_id)
    if col not in firmness_df.columns:
        return {
            "firmness_latest": "",
            "firmness_date": "",
            "first_zero_date": "",
            "first_le5_date": "",
        }

    current_date = timestamp.normalize()
    previous = firmness_df[firmness_df["date"] <= current_date]
    latest_value = ""
    latest_date = ""
    if not previous.empty:
        latest_row = previous.iloc[-1]
        latest_value = latest_row[col]
        latest_date = latest_row["date"].date().isoformat()

    first_zero = firmness_df.loc[firmness_df[col] <= 0, "date"]
    first_le5 = firmness_df.loc[firmness_df[col] <= 5, "date"]
    return {
        "firmness_latest": latest_value,
        "firmness_date": latest_date,
        "first_zero_date": first_zero.iloc[0].date().isoformat() if not first_zero.empty else "",
        "first_le5_date": first_le5.iloc[0].date().isoformat() if not first_le5.empty else "",
    }


def environment_context(env_df: pd.DataFrame, timestamp: pd.Timestamp) -> dict[str, object]:
    if env_df.empty:
        return {
            "temperature_c": "",
            "humidity_rh": "",
            "env_timestamp": "",
        }
    deltas = (env_df["datetime"] - timestamp).abs()
    row = env_df.loc[deltas.idxmin()]
    return {
        "temperature_c": row["temperature_c"],
        "humidity_rh": row["humidity_rh"],
        "env_timestamp": row["datetime"],
    }


def feature_context(feature_df: pd.DataFrame, fruit_id: str, timestamp: pd.Timestamp) -> dict[str, object]:
    if feature_df.empty:
        return {}
    if "fruit_id" not in feature_df.columns or "timestamp" not in feature_df.columns:
        return {}
    fruit_df = feature_df[feature_df["fruit_id"] == fruit_id]
    if fruit_df.empty:
        return {}
    deltas = (fruit_df["timestamp"] - timestamp).abs()
    row = fruit_df.loc[deltas.idxmin()]
    return row.to_dict()


def build_exclusion_lookup(exclusions_df: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str, str]]]:
    lookup: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str, str]]] = {}
    if exclusions_df.empty:
        return lookup
    for _, row in exclusions_df.iterrows():
        fruit_id = str(row.get("fruit_id", ""))
        if not fruit_id:
            continue
        start = row.get("start_timestamp")
        end = row.get("end_timestamp")
        if pd.isna(start) or pd.isna(end):
            continue
        lookup.setdefault(fruit_id, []).append(
            (
                pd.Timestamp(start),
                pd.Timestamp(end),
                str(row.get("reason", "")),
                str(row.get("note", "")),
            )
        )
    return lookup


def timestamp_in_ranges(timestamp: pd.Timestamp, ranges: list[tuple[pd.Timestamp, pd.Timestamp, str, str]]) -> tuple[bool, str, str]:
    for start, end, reason, note in ranges:
        if start <= timestamp <= end:
            return True, reason, note
    return False, "", ""


def nearest_index(values: pd.Series, target: pd.Timestamp) -> int:
    deltas = (pd.to_datetime(values.reset_index(drop=True)) - target).abs()
    return int(deltas.argmin())


def build_frame_table(
    frames_by_fruit: dict[str, list[FruitFrame]],
    firmness_df: pd.DataFrame,
    env_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    eol_df: pd.DataFrame,
    exclusions_df: pd.DataFrame,
) -> pd.DataFrame:
    eol_lookup = eol_df.set_index("fruit_id").to_dict(orient="index") if not eol_df.empty and "fruit_id" in eol_df.columns else {}
    exclusion_lookup = build_exclusion_lookup(exclusions_df)
    rows: list[dict[str, object]] = []

    for fruit_id, frames in frames_by_fruit.items():
        eol_row = eol_lookup.get(fruit_id, {})
        eol_timestamp = eol_row.get("eol_timestamp")
        for frame_idx, frame in enumerate(frames):
            excluded, exclude_reason, exclude_note = timestamp_in_ranges(frame.timestamp, exclusion_lookup.get(fruit_id, []))
            if pd.notna(eol_timestamp):
                rul_raw = (pd.Timestamp(eol_timestamp) - frame.timestamp).total_seconds() / 3600
                rul_clipped = max(rul_raw, 0.0)
                post_eol_flag = frame.timestamp > pd.Timestamp(eol_timestamp)
            else:
                rul_raw = pd.NA
                rul_clipped = pd.NA
                post_eol_flag = False
            frame_status = "excluded" if excluded else "post_eol" if post_eol_flag else "kept" if pd.notna(eol_timestamp) else "missing_eol"
            env = environment_context(env_df, frame.timestamp)
            firmness = firmness_context(firmness_df, fruit_id, frame.timestamp)
            features = feature_context(feature_df, fruit_id, frame.timestamp)

            row = {
                "fruit_id": fruit_id,
                "frame_idx": frame_idx,
                "timestamp": frame.timestamp,
                "date": frame.timestamp.normalize(),
                "image_path": str(frame.image_path),
                "day": frame.timestamp.date().isoformat(),
                "elapsed_hours": round((frame.timestamp - frames[0].timestamp).total_seconds() / 3600, 3),
                "eol_timestamp": eol_timestamp,
                "eol_basis": eol_row.get("eol_basis", ""),
                "eol_confidence": eol_row.get("eol_confidence", ""),
                "visual_note": eol_row.get("visual_note", ""),
                "rul_hours_raw": None if pd.isna(rul_raw) else round(float(rul_raw), 3),
                "rul_hours": None if pd.isna(rul_clipped) else round(float(rul_clipped), 3),
                "pre_eol_flag": bool(pd.notna(eol_timestamp) and frame.timestamp <= pd.Timestamp(eol_timestamp)),
                "post_eol_flag": bool(post_eol_flag),
                "excluded_flag": bool(excluded),
                "exclude_reason": exclude_reason,
                "exclude_note": exclude_note,
                "frame_status": frame_status,
                "usable_for_baseline": bool((not excluded) and pd.notna(eol_timestamp) and frame.timestamp <= pd.Timestamp(eol_timestamp)),
                "temperature_c": env["temperature_c"],
                "humidity_rh": env["humidity_rh"],
                "env_timestamp": env["env_timestamp"],
                "firmness_latest": firmness["firmness_latest"],
                "firmness_date": firmness["firmness_date"],
                "first_firmness_zero_date": firmness["first_zero_date"],
                "first_firmness_le5_date": firmness["first_le5_date"],
            }
            for column in [
                "mean_r",
                "mean_g",
                "mean_b",
                "green_ratio",
                "excess_green",
                "mean_lab_l",
                "mean_lab_a",
                "mean_lab_b",
                "dark_coverage",
                "largest_dark_spot_fraction",
                "mask_area",
            ]:
                row[column] = features.get(column, pd.NA)
            rows.append(row)

    frame_df = pd.DataFrame(rows)
    if not frame_df.empty:
        frame_df = frame_df.sort_values(["fruit_id", "timestamp"]).reset_index(drop=True)
    return frame_df


def build_fruit_summary(frame_df: pd.DataFrame, eol_df: pd.DataFrame, exclusions_df: pd.DataFrame) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    for fruit_id, group in frame_df.groupby("fruit_id", sort=True):
        eol_row = eol_df[eol_df["fruit_id"] == fruit_id].iloc[0].to_dict() if not eol_df.empty and not eol_df[eol_df["fruit_id"] == fruit_id].empty else {}
        cut_group = exclusions_df[exclusions_df["fruit_id"] == fruit_id] if not exclusions_df.empty else pd.DataFrame()
        eol_timestamp = eol_row.get("eol_timestamp")
        eol_frame_idx = None
        eol_frame_timestamp = None
        if pd.notna(eol_timestamp):
            eol_idx = nearest_index(group["timestamp"], pd.Timestamp(eol_timestamp))
            eol_frame_idx = int(group.iloc[eol_idx]["frame_idx"])
            eol_frame_timestamp = group.iloc[eol_idx]["timestamp"]

        summary_rows.append(
            {
                "fruit_id": fruit_id,
                "frame_count": len(group),
                "kept_frames": int(group["usable_for_baseline"].sum()),
                "excluded_frames": int(group["excluded_flag"].sum()),
                "pre_eol_frames": int(group["pre_eol_flag"].sum()),
                "post_eol_frames": int(group["post_eol_flag"].sum()),
                "first_timestamp": group["timestamp"].min(),
                "last_timestamp": group["timestamp"].max(),
                "eol_timestamp": eol_timestamp,
                "eol_frame_idx": eol_frame_idx,
                "eol_frame_timestamp": eol_frame_timestamp,
                "eol_basis": eol_row.get("eol_basis", ""),
                "eol_confidence": eol_row.get("eol_confidence", ""),
                "cut_ranges": int(len(cut_group)),
                "cut_frame_count": int(cut_group["frame_count"].astype(int).sum()) if not cut_group.empty and "frame_count" in cut_group.columns else 0,
                "temperature_mean_c": group["temperature_c"].mean() if pd.api.types.is_numeric_dtype(group["temperature_c"]) else pd.NA,
                "humidity_mean_rh": group["humidity_rh"].mean() if pd.api.types.is_numeric_dtype(group["humidity_rh"]) else pd.NA,
                "mean_dark_coverage": group["dark_coverage"].astype(float).mean() if "dark_coverage" in group.columns else pd.NA,
                "mean_green_ratio": group["green_ratio"].astype(float).mean() if "green_ratio" in group.columns else pd.NA,
                "firmness_start": group["firmness_latest"].iloc[0],
                "firmness_end": group["firmness_latest"].iloc[-1],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("fruit_id").reset_index(drop=True)
    return summary_df


def write_report(output_dir: Path, frame_df: pd.DataFrame, summary_df: pd.DataFrame) -> Path:
    report_path = output_dir / "avocado_rul_inspection_report.md"
    summary_text = summary_df.to_string(index=False) if not summary_df.empty else "_No data_"
    lines = [
        "# Avocado RUL Inspection Report",
        "",
        f"- Total frames: {len(frame_df)}",
        f"- Fruits: {summary_df['fruit_id'].nunique() if not summary_df.empty else 0}",
        f"- Baseline-usable frames: {int(frame_df['usable_for_baseline'].sum()) if not frame_df.empty else 0}",
        f"- Excluded frames: {int(frame_df['excluded_flag'].sum()) if not frame_df.empty else 0}",
        "",
        "## Per-fruit summary",
        "",
        "```text",
        summary_text,
        "```",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run(
    segmented_dir: Path = DEFAULT_SEGMENTED_DIR,
    firmness_csv: Path = DEFAULT_FIRMNESS_CSV,
    env_csv: Path = DEFAULT_ENV_CSV,
    features_csv: Path = DEFAULT_FEATURES_CSV,
    eol_csv: Path = DEFAULT_EOL_CSV,
    exclusions_csv: Path = DEFAULT_EXCLUSIONS_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_csv = output_dir / "avocado_rul_frame_table.csv"
    ready_csv = output_dir / "avocado_rul_ready_frames.csv"
    summary_csv = output_dir / "avocado_rul_fruit_summary.csv"
    report_md = output_dir / "avocado_rul_inspection_report.md"

    if not force and frame_csv.exists() and ready_csv.exists() and summary_csv.exists() and report_md.exists():
        print(f"Inspection outputs already exist in {output_dir}. Skipping.")
        return {
            "frame_csv": frame_csv,
            "ready_csv": ready_csv,
            "summary_csv": summary_csv,
            "report_md": report_md,
        }

    frames_by_fruit = load_frames(segmented_dir)
    if not frames_by_fruit:
        raise RuntimeError(f"No segmented avocado frames found in {segmented_dir}")

    firmness_df = load_firmness(firmness_csv)
    env_df = load_environment(env_csv)
    feature_df = load_features(features_csv)
    eol_df = load_eol_annotations(eol_csv)
    exclusions_df = load_exclusions(exclusions_csv)

    frame_df = build_frame_table(frames_by_fruit, firmness_df, env_df, feature_df, eol_df, exclusions_df)
    summary_df = build_fruit_summary(frame_df, eol_df, exclusions_df)
    ready_df = frame_df[frame_df["usable_for_baseline"]].copy()

    frame_df.to_csv(frame_csv, index=False)
    ready_df.to_csv(ready_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    report_path = write_report(output_dir, frame_df, summary_df)

    print(f"Saved frame table to {frame_csv}")
    print(f"Saved ready-frame table to {ready_csv}")
    print(f"Saved fruit summary to {summary_csv}")
    print(f"Saved report to {report_path}")

    return {
        "frame_csv": frame_csv,
        "ready_csv": ready_csv,
        "summary_csv": summary_csv,
        "report_md": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect avocado RUL labels and build inspection tables.")
    parser.add_argument("--force", action="store_true", help="Regenerate outputs even if they already exist.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Inspection output directory.")
    parser.add_argument("--segmented-dir", type=Path, default=DEFAULT_SEGMENTED_DIR)
    parser.add_argument("--firmness-csv", type=Path, default=DEFAULT_FIRMNESS_CSV)
    parser.add_argument("--env-csv", type=Path, default=DEFAULT_ENV_CSV)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--eol-csv", type=Path, default=DEFAULT_EOL_CSV)
    parser.add_argument("--exclusions-csv", type=Path, default=DEFAULT_EXCLUSIONS_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        segmented_dir=args.segmented_dir,
        firmness_csv=args.firmness_csv,
        env_csv=args.env_csv,
        features_csv=args.features_csv,
        eol_csv=args.eol_csv,
        exclusions_csv=args.exclusions_csv,
        output_dir=args.output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
