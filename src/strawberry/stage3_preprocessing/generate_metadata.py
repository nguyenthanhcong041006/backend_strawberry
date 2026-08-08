import pandas as pd
from pathlib import Path

def generate_metadata():
    project_root = Path(__file__).resolve().parents[3]
    final_dir = project_root / "data" / "02_processed" / "strawberry" / "final"
    split_dir = project_root / "data" / "03_split" / "strawberry"
    
    label_files = sorted(list(final_dir.glob("F0*/labels.csv")))
    if not label_files:
        print("No labels.csv files found to generate metadata.")
        return
        
    print(f"Aggregating {len(label_files)} label files into metadata.csv...")
    dfs = []
    for lf in label_files:
        df = pd.read_csv(lf)
        dfs.append(df)
        
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
    combined_df = combined_df.sort_values(by=["fruit_id", "timestamp"]).reset_index(drop=True)
    
    # Selected columns for metadata
    target_columns = [
        "experiment_id",
        "fruit_type",
        "fruit_id",
        "roi_id",
        "image_path",
        "raw_path",
        "timestamp",
        "temperature_c",
        "humidity_pct",
        "sensor_timestamp",
        "mapping_method",
        "mapping_delta_seconds",
        "environment_source",
        "sensor_status",
        "rul_hours",
        "eol_timestamp",
        "label_status"
    ]
    
    # Filter columns that are present
    out_cols = [col for col in target_columns if col in combined_df.columns]
    metadata_df = combined_df[out_cols]
    
    # Save to final_dir / metadata.csv
    out_csv = final_dir / "metadata.csv"
    metadata_df.to_csv(out_csv, index=False)
    print(f"Saved metadata to {out_csv} ({len(metadata_df)} rows)")
    
    # Also save to split_dir / metadata.csv if split_dir exists
    if split_dir.exists():
        split_out_csv = split_dir / "metadata.csv"
        metadata_df.to_csv(split_out_csv, index=False)
        print(f"Saved metadata copy to {split_out_csv}")

def main():
    generate_metadata()

if __name__ == "__main__":
    main()
