from pathlib import Path
import pandas as pd
import json


def generate_labels(
    manifest_csv: Path,
    eol_csv: Path,
    environment_csv: Path,
    firmness_csv: Path,
    output_csv: Path,
    
):
    """
    Generate labels.csv for Strawberry RUL prediction.
    """

    print("Loading frame manifest...")
    manifest_df = pd.read_csv(manifest_csv)

    print("Loading EOL anchors...")
    eol_df = pd.read_csv(eol_csv)
    
    print("Loading environment data...")
    environment_df = pd.read_csv(environment_csv)
    environment_df = environment_df.rename(
    columns={
        "humidity_rh":"humidity_pct"
        }
    )
    
    print("Loading hardness data...")
    firmness_df = pd.read_csv(firmness_csv)

    
    # Convert datetime
    manifest_df["timestamp"] = pd.to_datetime(
        manifest_df["timestamp"]
    )

    eol_df["eol_timestamp"] = pd.to_datetime(
        eol_df["eol_timestamp"]
    )
    
    # Merge EOL information
    labels_df = manifest_df.merge(
    eol_df[
        [
            "fruit_id",
            "eol_timestamp",
            "eol_basis"
        ]
    ],
    on="fruit_id",
    how="left"
    )
    
    # environment data
    environment_df["timestamp"] = pd.to_datetime(
    environment_df["timestamp"]
    )

    labels_df = labels_df.merge(
    environment_df[ 
            [
                "timestamp", 
                "temperature_c", 
                "humidity_pct"
            ] 
        ],
    on="timestamp",
    how="left"
    )
    
    # hardness data
    labels_df["date"] = labels_df["timestamp"].dt.date

    firmness_df["date"] = pd.to_datetime(
    firmness_df["date"]
    ).dt.date
    
    labels_df = labels_df.merge(
    firmness_df,
    on=["date","fruit_id"],
    how="left"
    )

    
    # Check missing EOL
    missing_eol = labels_df[
        labels_df["eol_timestamp"].isna()
    ]["fruit_id"].unique()

    if len(missing_eol) > 0:
        raise ValueError(
            f"Missing EOL information for fruits: {missing_eol}"
        )

    
    # Calculate RUL (hours)
    # formular: rul = eol - current time
    labels_df["rul_hours"] = (
        labels_df["eol_timestamp"]
        - labels_df["timestamp"]
    ).dt.total_seconds() / 3600

    # Clip negative values
    labels_df["rul_hours"] = (
        labels_df["rul_hours"]
        .clip(lower=0)
        .round(2)
    )

    
    # Columns required by specification
    labels_df["roi_id"] = labels_df["fruit_id"]

    labels_df["raw_path"] = ""

    labels_df["firmness_avg"] = labels_df["firmness"]  # declare your hardness (firmness) here

    labels_df["firmness_available"] = ( labels_df["firmness_avg"].notna() )   # declare your hardness (firmness) here

    labels_df["valid_frame"] = labels_df["mask_valid"]

    labels_df["exclude_reason"] = ""

    labels_df["label_status"] = "approved"

    
    # Output columns
    output_columns = [
        "experiment_id",
        "fruit_type",
        "fruit_id",
        "roi_id",
        "image_path",
        "raw_path",
        "timestamp",
        "eol_timestamp",
        "rul_hours",
        "temperature_c",
        "humidity_pct",
        "firmness_avg",
        "firmness_available",
        "valid_frame",
        "exclude_reason",
        "eol_basis",
        "label_status"
    ]

    labels_df = labels_df[output_columns]

    
    # Sort by fruit and time
    labels_df = labels_df.sort_values(
        by=["fruit_id", "timestamp"]
    )

    
    # Save
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    labels_df.to_csv(
        output_csv,
        index=False
    )

    print("\n" + "=" * 60)
    print("Labels generated successfully!")
    print(f"Rows: {len(labels_df)}")
    print(f"Output: {output_csv}")
    print("=" * 60)


def main():

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    
    CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        configs = json.load(f)

    active_dataset = configs["active_dataset"]

    MANIFEST_DIR = (
        PROJECT_ROOT
        / "data"
        / "02_processed"
        / "manifests"
        / active_dataset
    )

    frame_manifest = (
        MANIFEST_DIR
        / "frame_manifest.csv"
    )

    eol_anchors = (
        MANIFEST_DIR
        / "eol_anchors.csv"
    )

    labels_csv = (
        MANIFEST_DIR
        / "labels.csv"
    )
    
    environment_csv = (
        MANIFEST_DIR
        / "numeric_mapping.csv"  # filled by sensor_ocr.py
    )
    
    firmness_csv = (
        MANIFEST_DIR
        / "hardness.csv"  # CHANGE YOUR CSV NAME !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    )

    generate_labels(
        manifest_csv=frame_manifest,
        eol_csv=eol_anchors,
        environment_csv=environment_csv,
        firmness_csv=firmness_csv,
        output_csv=labels_csv
    )


if __name__ == "__main__":
    main()
