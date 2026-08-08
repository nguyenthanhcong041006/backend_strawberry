from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure stage3_preprocessing directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extracting_frames import main as extracting_frames_main
from crop_images import main as crop_images_main
from segmentation import main as segmentation_main
from frame_differencing import main as frame_differencing_main
from assign_id import main as assign_id_main
from eol import main as eol_main
from manifests import main as manifests_main
from consolidate_final import main as consolidate_final_main
from generate_final_labels import generate_final_labels as generate_final_labels_main
<<<<<<< HEAD
=======
from sensor_ocr import main as sensor_ocr_main
>>>>>>> 85deb16e504d50b57c73d77335b0ccec3805957a
from generate_metadata import generate_metadata as generate_metadata_main
from split_data import main as split_data_main
from sensor_mapping import remap_final_labels as sensor_mapping_main


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

def run_step(name, func):

    print(f"\n=== {name} ===")

    try:
        func()
        print(f"[OK] {name}")

    except Exception as e:
        print(f"[FAILED] {name}")
        print(e)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run the strawberry preprocessing pipeline.")
    parser.add_argument(
        "--sensor-csv",
        type=Path,
        default=None,
        help="Path to a real strawberry sensor CSV with timestamp, temperature_c, and humidity_pct.",
    )
    parser.add_argument(
        "--sensor-tolerance-minutes",
        type=int,
        default=45,
        help="Nearest-timestamp tolerance for matching frames to sensor readings.",
    )
    return parser.parse_args()


def load_pipeline_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    configs = load_pipeline_config()
    configured_sensor = configs.get("sensor_csv")
    sensor_csv = args.sensor_csv or (Path(configured_sensor) if configured_sensor else None)
    if sensor_csv is None:
        candidate = PROJECT_ROOT / "data" / "01_raw" / "strawberry" / "sensor_readings.csv"
        sensor_csv = candidate
    elif not sensor_csv.is_absolute():
        sensor_csv = PROJECT_ROOT / sensor_csv
    if not sensor_csv.exists():
        raise FileNotFoundError(
            "No real strawberry sensor CSV was provided. Pass --sensor-csv or set "
            "stage3_preprocessing/config.json -> sensor_csv to a real file."
        )

    # Step 0: Extract frames from videos
    #run_step("Extract Frames", extracting_frames_main)
    
    # Step 1: Crop images to focus on strawberries
    run_step("Crop Images", crop_images_main)
    
    # Step 2: Perform frame differencing and validate masks
    run_step("Frame Differencing", frame_differencing_main)

    # Step 3: Segment strawberries from the background
    run_step("Segmentation", segmentation_main)

    # Step 4: Assign unique IDs to each strawberry
    run_step("Assign IDs", assign_id_main)

    # Step 5: Generate end-of-life (EOL) anchors for strawberries
    run_step("Generate EOL Anchors", eol_main)

    # Step 6: Generate manifests for the dataset
    run_step("Generate Manifests", manifests_main)

    # Step 7: Consolidate final dataset and manifest
    run_step("Consolidate Final Dataset", consolidate_final_main)

    # Step 8: Label remaining useful life (RUL)
    run_step("Label RUL and Temporal Features", generate_final_labels_main)

<<<<<<< HEAD
    # Step 9: Map real sensor data into each fruit label file
    run_step(
        "Map Real Sensor Data",
        lambda: sensor_mapping_main(
            PROJECT_ROOT,
            sensor_csv,
            tolerance=f"{args.sensor_tolerance_minutes}min",
        ),
    )
=======
    # Step 9: OCR temperature and humidity from sensor display
    run_step("Sensor OCR Environment Data", sensor_ocr_main)
>>>>>>> 85deb16e504d50b57c73d77335b0ccec3805957a

    # Step 10: Generate consolidated metadata.csv
    run_step("Generate Metadata CSV", generate_metadata_main)

    # Step 11: Split data into training, validation, and test sets
    run_step("Split Data", split_data_main)

    print("\n[SUCCESS] Preprocessing pipeline completed successfully!")

if __name__ == "__main__":
    main()
