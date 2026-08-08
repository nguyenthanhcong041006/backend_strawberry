import os
import re
import shutil
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_FILE = PROJECT_ROOT / "src" / "strawberry" / "stage3_preprocessing" / "config.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    configs = json.load(f)

active_dataset = configs["active_dataset"]
dataset_cfg = configs["datasets"][active_dataset]

PROCESSED_ROOT = PROJECT_ROOT / "data" / "02_processed" / active_dataset

def is_segmented_folder(folder_name):

    if active_dataset == "strawberry":
        return re.match(
            r"^segmented_\d{2}-\d{2}-\d{4}$",
            folder_name
        ) is not None

    return folder_name == f"segmented_{active_dataset}"

def main():

    if active_dataset == "strawberry":

        segmented_folders = sorted(
        [
            folder
            for folder in PROCESSED_ROOT.iterdir()
            if folder.is_dir()
            and is_segmented_folder(folder.name)
        ]
    )

    else:
        segmented_folders = [PROCESSED_ROOT / f"segmented_{active_dataset}"]

    if not segmented_folders:
        print("No segmented folders found.")
        return

    for input_dir in segmented_folders:

        if active_dataset == "strawberry":

            date_str = input_dir.name.replace("segmented_","")

            output_dir = (PROCESSED_ROOT / f"assigned_{date_str}")

        else:
            date_str = active_dataset

            output_dir = (PROCESSED_ROOT / f"assigned_{active_dataset}")

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        assigned_count = 0

        print("\n" + "=" * 60)
        print(f"Processing date: {date_str}")
        print("=" * 60)

        for filename in os.listdir(input_dir):

            if not filename.lower().endswith(".png"):
                continue

            match = re.search(rf"^(.*?)_{active_dataset}_(\d+)\.png$", filename)

            if not match:
                print(f"Skip: {filename}")
                continue

            prefix = match.group(1)
            strawberry_id = match.group(2)

            formatted_id = f"{int(strawberry_id):02d}"

            new_filename = (
                f"{prefix}_F{formatted_id}.png"
            )

            target_folder = os.path.join(
                output_dir,
                f"F{formatted_id}"
            )

            os.makedirs(
                target_folder,
                exist_ok=True
            )

            src_path = os.path.join(
                input_dir,
                filename
            )

            dst_path = os.path.join(
                target_folder,
                new_filename
            )

            shutil.copy2(
                src_path,
                dst_path
            )

            assigned_count += 1

        print("=" * 40)
        print(f"Assigned {assigned_count} images")
        print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()