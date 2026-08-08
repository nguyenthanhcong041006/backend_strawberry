import argparse
import csv
import shutil
from pathlib import Path
import json


# Project root directory: .../Strawberry-RUL-prediction
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    configs = json.load(f)

active_dataset = configs["active_dataset"]

# File labels:
FINAL_METADATA_CSV = PROJECT_ROOT / "data" / "02_processed" / "strawberry" / "final" / "metadata.csv"
MANIFEST_LABELS_CSV = PROJECT_ROOT / "data" / "02_processed" / active_dataset / "manifests" / "labels.csv"
DEFAULT_LABELS_CSV = FINAL_METADATA_CSV if FINAL_METADATA_CSV.exists() else MANIFEST_LABELS_CSV

# folder after split:
# data/03_split/train, data/03_split/val, data/03_split/test
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "03_split" / active_dataset

# Fixed fruit-ID split:
# train: strawberry 1-4 (F01-F04), val: strawberry 5 (F05), test: strawberry 6 (F06)
DEFAULT_SPLIT_TO_IDS = {
    "train": ["F01", "F02", "F03", "F04"],
    "val": ["F05"],
    "test": ["F06"],
}


def read_labels(labels_csv):
    # read labels.csv into a list dictionary for processing using the available CSV library
    with labels_csv.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def get_id_column(rows):
    # Support both the old split_data schema and the current manifest schema.
    if "strawberry_id" in rows[0]:
        return "strawberry_id"

    if "fruit_id" in rows[0]:
        return "fruit_id"

    raise ValueError("labels.csv must contain a strawberry_id or fruit_id column")


def fruit_id_sort_key(fruit_id):
    prefix = fruit_id.rstrip("0123456789")
    number_text = fruit_id[len(prefix):]

    if number_text.isdigit():
        return prefix, int(number_text)

    return fruit_id, -1


def group_rows_by_fruit(rows, id_column):
    # group the frames/images by strawberry_id and split them left-hand, not randomly split them by image
    groups = {}

    for row in rows:
        strawberry_id = row[id_column]

        if strawberry_id not in groups:
            groups[strawberry_id] = []

        groups[strawberry_id].append(row)

    return groups


def split_fruit_ids(strawberry_ids):
    # Split by fixed fruit IDs to prevent image-level leakage and keep experiments repeatable.
    expected_ids = {
        strawberry_id
        for split_ids in DEFAULT_SPLIT_TO_IDS.values()
        for strawberry_id in split_ids
    }
    available_ids = set(strawberry_ids)

    if available_ids == expected_ids:
        return DEFAULT_SPLIT_TO_IDS

    # Dynamic fallback if available fruit IDs differ from F01-F06
    n = len(strawberry_ids)
    if n == 0:
        raise ValueError("No fruit IDs available to split.")
    
    val_idx = max(1, int(n * 0.7))
    test_idx = max(val_idx + 1, int(n * 0.85)) if n > 2 else val_idx
    if test_idx >= n:
        test_idx = n - 1 if n > 1 else n
    
    train_ids = strawberry_ids[:val_idx]
    val_ids = strawberry_ids[val_idx:test_idx] if test_idx > val_idx else [strawberry_ids[-1]]
    test_ids = strawberry_ids[test_idx:] if test_idx < n else ([strawberry_ids[-1]] if len(val_ids) > 1 else [])

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def get_image_paths(image_path_text, data_dir):
    image_path = Path(image_path_text)

    if image_path.is_absolute():
        source_image_path = image_path
    elif (PROJECT_ROOT / image_path).exists():
        source_image_path = PROJECT_ROOT / image_path
    elif (data_dir / image_path).exists():
        source_image_path = data_dir / image_path
    elif (data_dir.parent / image_path).exists():
        source_image_path = data_dir.parent / image_path
    else:
        source_image_path = data_dir / image_path.name

    try:
        relative_image_path = source_image_path.relative_to(data_dir)
    except ValueError:
        relative_image_path = Path(source_image_path.name)

    return source_image_path, relative_image_path


def copy_split_images(rows, split_name, data_dir, output_dir):
    split_dir = output_dir / split_name
    image_output_dir = split_dir / "images"
    labels_output_csv = split_dir / "labels.csv"

    image_output_dir.mkdir(parents=True, exist_ok=True)

    copied_rows = []

    for row in rows:
        source_image_path, relative_image_path = get_image_paths(
            row["image_path"],
            data_dir
        )

        if not source_image_path.exists():
            raise FileNotFoundError(
                f"Cannot find image: {source_image_path}"
            )

        target_image_path = (
            image_output_dir / relative_image_path
        )
        target_image_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(
            source_image_path,
            target_image_path
        )

        copied_row = dict(row)

        copied_row["image_path"] = str(
            Path("images") / relative_image_path
        ).replace("\\", "/")

        copied_rows.append(copied_row)

    with labels_output_csv.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(copied_rows)

    return len(copied_rows)


def write_split_summary(split_to_ids, split_to_count, output_dir):
    """Save the summary file to know which split contains which strawberry _id"""
    summary_csv = output_dir / "split_summary.csv"

    with summary_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "fruit_ids", "num_fruits", "num_images"],
        )
        writer.writeheader()

        for split_name, strawberry_ids in split_to_ids.items():
            writer.writerow({
                "split": split_name,
                "fruit_ids": " ".join(strawberry_ids),
                "num_fruits": len(strawberry_ids),
                "num_images": split_to_count[split_name],
            })


def main():
    parser = argparse.ArgumentParser(
        description=f"Split {active_dataset} dataset 4 train, 1 val, 1 test."
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help="The path to the labels.csv file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="output folder for train/val/test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deprecated. The split is now fixed by fruit ID and does not use a seed.",
    )
    args = parser.parse_args()

    labels_csv = args.labels_csv
    if FINAL_METADATA_CSV.exists():
        labels_csv = FINAL_METADATA_CSV
    elif not labels_csv.exists() and MANIFEST_LABELS_CSV.exists():
        labels_csv = MANIFEST_LABELS_CSV
    labels_csv = labels_csv.resolve()
    data_dir = labels_csv.parent
    output_dir = args.output_dir.resolve()

    rows = read_labels(labels_csv)

    if not rows:
        raise ValueError(f"File labels empty: {labels_csv}")

    id_column = get_id_column(rows)
    groups = group_rows_by_fruit(rows, id_column)
    strawberry_ids = sorted(groups.keys(), key=fruit_id_sort_key)

    split_to_ids = split_fruit_ids(strawberry_ids=strawberry_ids)

    split_to_count = {}

    for split_name, ids in split_to_ids.items():
        # Get all rows of strawberry_id belonging to the current split
        split_rows = []

        for strawberry_id in ids:
            split_rows.extend(groups[strawberry_id])

        # rearrange the labels so the output file is readable: first ID -> time of capture
        split_rows = sorted(
            split_rows,
            key=lambda row: (fruit_id_sort_key(row[id_column]), row["timestamp"]),
        )

        split_to_count[split_name] = copy_split_images(
            rows=split_rows,
            split_name=split_name,
            data_dir=data_dir,
            output_dir=output_dir,
        )

    write_split_summary(split_to_ids, split_to_count, output_dir)

    print("Split done")
    print(f"Labels input: {labels_csv}")
    print(f"Output dir: {output_dir}")

    for split_name, ids in split_to_ids.items():
        print(
            f"{split_name}: fruit_id={ids}, "
            f"images={split_to_count[split_name]}"
        )


if __name__ == "__main__":
    main()
