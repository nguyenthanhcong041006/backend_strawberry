import pandas as pd
from pathlib import Path
import os

from time_compression import compress_recording_timestamp

def make_relative(p_str, project_root):
    if pd.isna(p_str) or not p_str:
        return ""
    p = Path(p_str)
    try:
        rel = p.relative_to(project_root)
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(p_str).replace("\\", "/")

def generate_final_labels():
    # Paths
    project_root = Path(__file__).resolve().parents[3]
    final_dir = project_root / "data" / "02_processed" / "strawberry" / "final"
    manifest_csv = final_dir / "final_manifest.csv"
    
    print(f"Loading manifest from {manifest_csv}...")
    df = pd.read_csv(manifest_csv)
    
    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Define EOL map
    # 26/03/2026 8:00:06 -> F01, F03, F04, F06
    # 28/03/2026 8:00:06 -> F02, F05
    eol_map = {
        "F01": compress_recording_timestamp(pd.to_datetime("2026-03-26 08:00:06").to_pydatetime()),
        "F03": compress_recording_timestamp(pd.to_datetime("2026-03-26 08:00:06").to_pydatetime()),
        "F04": compress_recording_timestamp(pd.to_datetime("2026-03-26 08:00:06").to_pydatetime()),
        "F06": compress_recording_timestamp(pd.to_datetime("2026-03-26 08:00:06").to_pydatetime()),
        "F02": compress_recording_timestamp(pd.to_datetime("2026-03-28 08:00:06").to_pydatetime()),
        "F05": compress_recording_timestamp(pd.to_datetime("2026-03-28 08:00:06").to_pydatetime())
    }
    
    # Map EOL timestamp to df
    df["eol_timestamp"] = df["fruit_id"].map(eol_map)
    
    # Calculate RUL in hours
    df["rul_hours"] = (df["eol_timestamp"] - df["timestamp"]).dt.total_seconds() / 3600
    df["rul_hours"] = df["rul_hours"].clip(lower=0).round(2)
    
    # Add specification columns based on label_rul.py
    df["experiment_id"] = "RUL_FINAL"
    df["fruit_type"] = "strawberry"
    df["roi_id"] = df["fruit_id"]
    df["image_path"] = df["final_path"].apply(lambda p: make_relative(p, final_dir))
    df["raw_path"] = df["source_path"].apply(lambda p: make_relative(p, project_root))
    df["temperature_c"] = pd.NA
    df["humidity_pct"] = pd.NA
    df["firmness_avg"] = pd.NA
    df["firmness_available"] = False
    df["valid_frame"] = True
    df["exclude_reason"] = ""
    df["eol_basis"] = "manual"
    df["label_status"] = "approved"
    
    # Output columns list based on label_rul.py
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
    
    # Process each fruit and save to its specific folder
    for fruit_id, group in df.groupby("fruit_id"):
        fruit_dir = final_dir / fruit_id
        fruit_dir.mkdir(parents=True, exist_ok=True)
        
        output_csv = fruit_dir / "labels.csv"
        
        fruit_df = group.copy().sort_values(by="timestamp").reset_index(drop=True)
        
        fruit_df_out = fruit_df[output_columns].copy()
        
        fruit_df_out.to_csv(output_csv, index=False)
        print(f"Saved {len(fruit_df_out)} labels for {fruit_id} to {output_csv}")
        
    print("All labels generated successfully!")

if __name__ == "__main__":
    generate_final_labels()
