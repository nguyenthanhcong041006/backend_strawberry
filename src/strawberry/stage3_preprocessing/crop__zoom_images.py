import cv2
from pathlib import Path
import re
import json

PROJECT_ROOT = Path(__file__).resolve().parents[3]





PROCESSED_DIR = PROJECT_ROOT / "data" / "02_processed" / "strawberry"

RAW_INPUT_DIR = Path(r"C:\fluttersrc\Strawberry-RUL-prediction\data\02_processed\strawberry\frames_18-03-2026")
OUTPUT_DIR = PROCESSED_DIR
TARGET_WIDTH = 1480
TARGET_HEIGHT = 858
IMAGE_TYPE = "strawberry"


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
    folders = [("18-03-2026", RAW_INPUT_DIR)]  # Only process the specific folder for now

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
