from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import re

from time_compression import compress_recording_timestamp


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    configs = json.load(f)

active_dataset = configs["active_dataset"]
dataset_cfg = configs["datasets"][active_dataset]

ROOT_DIR = PROJECT_ROOT / "data" / "02_processed" / active_dataset
MANIFEST_DIR = ROOT_DIR / "manifests"


MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

FRUIT_TYPE = active_dataset
EXPERIMENT_ID = f"{active_dataset}_experiment"



def parse_datetime(date_str: str, filename: str):

    if active_dataset == "strawberry":

        match = re.search(
            r"frame-\d+_(\d{2})-(\d{2})-(\d{2})",
            filename
        )

        if not match:
            raise ValueError(
                f"Cannot parse time from {filename}"
            )

        hh, mm, ss = match.groups()

        return compress_recording_timestamp(
            datetime.strptime(
                f"{date_str} {hh}:{mm}:{ss}",
                "%d-%m-%Y %H:%M:%S"
            )
        )

    else:

        match = re.search(
            r"webcam_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})",
            filename
        )

        if not match:
            raise ValueError(
                f"Cannot parse time from {filename}"
            )

        date_part = match.group(1)
        time_part = match.group(2)

        return datetime.strptime(
            f"{date_part} {time_part}",
            "%Y-%m-%d %H-%M-%S"
        )


def extract_frame_base(filename: str):
    """
    frame-1_12-26-28_F01.png
        ->
    frame-1_12-26-28
    """

    return re.sub(
        r"_F\d+\.png$",
        "",
        filename
    )


def fruit_to_mask_name(frame_base, fruit_id):
    """
    F01 -> strawberry_1

    frame-1_12-26-28
        ->
    frame-1_12-26-28_strawberry_1.png
    """

    fruit_num = int(fruit_id.replace("F", ""))

    return (
        f"{frame_base}_{active_dataset}_{fruit_num}.png"
    )


def parse_bool(value, default=False):
    if pd.isna(value):
        return default

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n", ""}:
        return False

    return default


def append_reason(reasons, value):
    if pd.isna(value):
        return

    text = str(value).strip()
    if text and text.lower() != "nan" and text not in reasons:
        reasons.append(text)


def main():
    # LOAD QC REPORTS
    qc_lookup = {}

    report_paths = sorted(ROOT_DIR.glob("frame_differencing_report_*.csv"))
    report_paths.extend(
        sorted(
            ROOT_DIR.glob(
                "frame_differencing_results_*/frame_differencing_report_*.csv"
            )
        )
    )

    for report_path in report_paths:

        report_df = pd.read_csv(report_path)

        for _, row in report_df.iterrows():

            frame_name = Path(
                row["frame_path"]
            ).stem

            qc = qc_lookup.setdefault(
                frame_name,
                {
                    "motion_detected": False,
                    "mask_valid": True,
                    "reason": [],
                    "mask_reason": []
                }
            )

            qc["motion_detected"] = (
                qc["motion_detected"]
                or parse_bool(row.get("motion_detected"), default=False)
            )
            qc["mask_valid"] = (
                qc["mask_valid"]
                and parse_bool(row.get("mask_valid"), default=True)
            )
            append_reason(qc["reason"], row.get("reason"))
            append_reason(qc["mask_reason"], row.get("mask_reason"))

    for qc in qc_lookup.values():
        qc["reason"] = "|".join(qc["reason"])
        qc["mask_reason"] = "|".join(qc["mask_reason"])


    # BUILD MANIFESTS
    accepted_rows = []
    excluded_rows = []

    fruit_sequence = {}

    if active_dataset == "strawberry":

        assigned_folders = sorted(ROOT_DIR.glob("assigned_*"))

        assigned_folders = [
        p for p in assigned_folders
        if re.match(
            r"assigned_\d{2}-\d{2}-\d{4}",
            p.name
            )
        ]

    else:
        assigned_folders = [ROOT_DIR / f"assigned_{active_dataset}"]


    for assigned_dir in assigned_folders:

        if active_dataset == "strawberry":
            date_str = assigned_dir.name.replace("assigned_", "")

        else:
            date_str = active_dataset

        if active_dataset == "strawberry":
            mask_dir = ROOT_DIR / f"segmented_{date_str}"
            if not mask_dir.exists():
                mask_dir = ROOT_DIR / f"mask_{date_str}"

        else:
            mask_dir = ROOT_DIR / f"mask_{active_dataset}"

        fruit_dirs = sorted(
            [
                p for p in assigned_dir.iterdir()
                if p.is_dir()
            ]
        )

        for fruit_dir in fruit_dirs:

            fruit_id = fruit_dir.name

            if fruit_id not in fruit_sequence:
                fruit_sequence[fruit_id] = 0

            if active_dataset == "strawberry":

                image_files = sorted(
                    fruit_dir.glob("*.png"),
                    key=lambda x: int(
                        re.search(
                            r"frame-(\d+)",
                            x.name
                        ).group(1)
                    )
                )

            else:

                image_files = sorted(
                    fruit_dir.glob("*.png"),
                    key=lambda x: datetime.strptime(
                        re.search(
                            r"webcam_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})",
                            x.name
                        ).group(0),
                        "webcam_%Y-%m-%d_%H-%M-%S"
                    )
                )
        

            for image_path in image_files:

                timestamp = parse_datetime(
                    date_str,
                    image_path.name
                )

                frame_base = extract_frame_base(
                    image_path.name
                )

                qc = qc_lookup.get(
                    frame_base,
                    {
                        "motion_detected": False,
                        "mask_valid": True,
                        "reason": "",
                        "mask_reason": ""
                    }
                )

                mask_name = fruit_to_mask_name(
                    frame_base,
                    fruit_id
                )

                mask_path = mask_dir / mask_name
            
                if not mask_path.exists():
                    print(f"WARNING: Missing mask -> {mask_path}")

                sequence_index = fruit_sequence[
                    fruit_id
                ]

                fruit_sequence[
                    fruit_id
                ] += 1

                accepted = (
                    (not qc["motion_detected"])
                    and
                    qc["mask_valid"]
                )

                row = {
                    "experiment_id":
                        EXPERIMENT_ID,

                    "fruit_type":
                        FRUIT_TYPE,

                    "fruit_id":
                        fruit_id,

                    "sequence_index":
                        sequence_index,

                    "timestamp":
                        timestamp,

                    "image_path":
                        str(image_path),

                    "mask_path":
                        str(mask_path),

                    "temperature_c":
                        None,

                    "humidity_pct":
                        None,

                    "motion_detected":
                        qc["motion_detected"],

                    "mask_valid":
                        qc["mask_valid"]
                }

                if accepted:

                    accepted_rows.append(row)

                else:

                    excluded_rows.append({
                        **row,

                        "reason":
                            qc["reason"],

                        "mask_reason":
                            qc["mask_reason"]
                    })


    # SAVE frame_manifest.csv
    frame_manifest = pd.DataFrame(
        accepted_rows
    )

    if frame_manifest.empty:
        raise RuntimeError(
            "No valid frames found."
        )

    frame_manifest = (
        frame_manifest
        .sort_values(
            ["fruit_id", "timestamp"]
        )
        .reset_index(drop=True)
    )

    frame_manifest["sequence_index"] = (
        frame_manifest
        .groupby("fruit_id")
        .cumcount()
    )

    frame_manifest.to_csv(
        MANIFEST_DIR /
        "frame_manifest.csv",
        index=False
    )



    # SAVE numeric_mapping.csv
    numeric_mapping = frame_manifest[
        [
            "experiment_id",
            "fruit_type",
            "fruit_id",
            "timestamp",
            "temperature_c",
            "humidity_pct"
        ]
    ]

    numeric_mapping.to_csv(
        MANIFEST_DIR /
        "numeric_mapping.csv",
        index=False
    )


    # SAVE excluded_frames.csv


    excluded_df = pd.DataFrame(
        excluded_rows
    )

    excluded_df.to_csv(
        MANIFEST_DIR /
        "excluded_frames.csv",
        index=False
    )


    # SAVE preprocessing_summary.json


    summary = {

        "experiment_id":
            EXPERIMENT_ID,

        "fruit_type":
            FRUIT_TYPE,

        "fruit_count":
            len(fruit_sequence),

        "days_processed":
            len(assigned_folders),

        "accepted_frames":
            len(frame_manifest),

        "excluded_frames":
            len(excluded_df),

        "total_frames":
            len(frame_manifest)
            + len(excluded_df)
    }

    with open(
        MANIFEST_DIR /
        "preprocessing_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            default=str
        )

    print(
        "\nManifest generation completed."
    )

    print(
        f"Accepted: {len(frame_manifest)}"
    )

    print(
        f"Excluded: {len(excluded_df)}"
    ) 
    
if __name__ == "__main__":
    main()
