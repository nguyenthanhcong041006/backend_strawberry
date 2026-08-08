import glob
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.avocado.stage2_eda.modules.feature_extractor import extract_features
from src.avocado.stage2_eda.modules.temporal_features import calculate_temporal_features

def parse_timestamp_from_filename(filename):
    """
    Parses timestamp from webcam_YYYY-MM-DD_HH-MM-SS_fruit_X.png
    """
    # e.g. webcam_2026-06-14_20-30-44_fruit_1.png
    parts = filename.split('_')
    date_str = parts[1] # 2026-06-14
    time_str = parts[2] # 20-30-44
    
    time_str = time_str.replace('-', ':')
    dt_str = f"{date_str} {time_str}"
    
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").isoformat()

def main():
    print("--- Avocado Stage 2 EDA: Feature Extraction ---")
    
    segmented_dir = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "segmented"
    output_csv = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "eda" / "features" / "avocado_features.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        "dark_threshold_l": 35, # Lightness threshold
        "min_dark_component_size": 20, 
        "glcm_levels": 16 
    }
    
    all_combined_features = []
    
    # Process each fruit folder
    fruit_folders = [
        path.name
        for path in segmented_dir.iterdir()
        if path.is_dir() and path.name.startswith("F")
    ]
    
    for fruit_id in sorted(fruit_folders):
        print(f"\nProcessing fruit: {fruit_id}")
        fruit_dir = segmented_dir / fruit_id
        
        image_files = glob.glob(str(fruit_dir / "*.png"))
        print(f"Found {len(image_files)} images.")
        
        if len(image_files) == 0:
            continue
            
        fruit_static_features = []
        for img_path in sorted(image_files):
            filename = Path(img_path).name
            try:
                timestamp = parse_timestamp_from_filename(filename)
                
                feats = extract_features(img_path, fruit_id, timestamp, config)
                if feats.get("valid", False):
                    fruit_static_features.append(feats)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
        if len(fruit_static_features) > 0:
            df_static = pd.DataFrame(fruit_static_features)
            print(f"Extracted static features for {len(df_static)} valid frames.")
            
            # Calculate temporal features for this fruit
            df_temporal = calculate_temporal_features(df_static)
            all_combined_features.append(df_temporal)
            
    if len(all_combined_features) > 0:
        final_df = pd.concat(all_combined_features, ignore_index=True)
        # Drop the internal 'valid' column
        if 'valid' in final_df.columns:
            final_df = final_df.drop(columns=['valid'])
            
        final_df.to_csv(output_csv, index=False)
        print(f"\nSuccessfully saved all features to: {output_csv}")
        print(f"Total rows: {len(final_df)}")
    else:
        print("\nNo valid features extracted.")

if __name__ == "__main__":
    main()
