import cv2
from pathlib import Path
import re
import json

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"
with open(CONFIG_FILE, "r", encoding="utf-8") as file:
    configs = json.load(file)

# get config
active_dataset = configs["active_dataset"]
PROCESSED_DIR = PROJECT_ROOT / "data" / "02_processed" / active_dataset
config = configs["datasets"][active_dataset]
RAW_INPUT_DIR = PROJECT_ROOT / config["input_dir"]
OUTPUT_DIR = PROCESSED_DIR
TARGET_WIDTH = config["crop"]["width"]
TARGET_HEIGHT = config["crop"]["height"]
IMAGE_TYPE = config["image_type"]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def is_date_folder(folder_name):
    return bool(
        re.match(r"^\d{2}-\d{2}-\d{4}$", folder_name)
    )

# search image for strawberry
def is_strawberry_folder(folder_name):
    return bool(
        re.match(r"^frames_\d{2}-\d{2}-\d{4}$", folder_name)
    )



# collect input for strawberry
def collect_input_folders():
    input_by_date = {}

    if RAW_INPUT_DIR.exists():
        for folder in RAW_INPUT_DIR.iterdir():
            if folder.is_dir() and is_date_folder(folder.name):
                input_by_date.setdefault(folder.name, folder)

    if PROCESSED_DIR.exists():
        for folder in PROCESSED_DIR.iterdir():
            if folder.is_dir() and is_strawberry_folder(folder.name):
                date_str = folder.name.replace("frames_", "")
                input_by_date[date_str] = folder

    return sorted(input_by_date.items())



def center_crop(image, target_width, target_height):
    height, width = image.shape[:2]
    if width < target_width or height < target_height:
        raise ValueError(
            f"Image size {width}x{height} is smaller than target "
            f"{target_width}x{target_height}"
        )

    x_start = (width - target_width) // 2
    y_start = (height - target_height) // 2
    x_end = x_start + target_width
    y_end = y_start + target_height
    return image[y_start:y_end, x_start:x_end]


    
def process_strawberry():
    folders = collect_input_folders()

    if not folders:
        print("No strawberry frame folders found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_cropped = 0
    total_skipped = 0

    print("\n" + "=" * 50)
    print("Processing: STRAWBERRY dataset")
    print("=" * 50)

    for date_str, input_dir in folders:

        image_paths = [
            p for p in sorted(input_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not image_paths:
            print(f"No images in folder: {input_dir}")
            continue

        output_folder = OUTPUT_DIR / f"cropped_{date_str}"
        output_folder.mkdir(parents=True, exist_ok=True)

        cropped_count = 0
        skipped_count = 0

        for image_path in image_paths:

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skip unreadable image: {image_path.name}")
                skipped_count += 1
                continue

            try:
                cropped = center_crop(image, TARGET_WIDTH, TARGET_HEIGHT)
            except ValueError as e:
                print(f"Skip {image_path.name}: {e}")
                skipped_count += 1
                continue

            output_path = output_folder / image_path.name
            cv2.imwrite(str(output_path), cropped)

            cropped_count += 1
            print(f"Cropped {image_path.name}")

        print("-" * 40)
        print(f"{date_str}: cropped={cropped_count}, skipped={skipped_count}")

        total_cropped += cropped_count
        total_skipped += skipped_count

    print("\n" + "=" * 50)
    print(f"Done STRAWBERRY: cropped={total_cropped}, skipped={total_skipped}")

def main():
    if IMAGE_TYPE == "strawberry":
        process_strawberry()

    else:
        raise ValueError(f"Unknown IMAGE_TYPE: {IMAGE_TYPE}")

if __name__ == "__main__":
    main()
