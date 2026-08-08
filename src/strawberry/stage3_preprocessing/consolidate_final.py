import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from time_compression import compress_recording_timestamp


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "02_processed" /"strawberry"
DEFAULT_OUTPUT_DIR = DEFAULT_PROCESSED_ROOT / "final"
DEFAULT_FRUIT_IDS = [f"F{i:02d}" for i in range(1, 7)]

ASSIGNED_DIR_RE = re.compile(r"^assigned_(\d{2}-\d{2}-\d{4})$")
IMAGE_RE = re.compile(
    r"^(frame-\d+)_(\d{2})-(\d{2})-(\d{2})_(F\d{2})\.png$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FinalImage:
    source_path: Path
    final_path: Path
    source_date: str
    fruit_id: str
    frame_id: str
    timestamp: datetime


def parse_assigned_date(folder_name: str):
    match = ASSIGNED_DIR_RE.match(folder_name)

    if not match:
        return None

    date_text = match.group(1)
    datetime.strptime(date_text, "%d-%m-%Y")

    return date_text


def parse_image_name(filename: str, date_text: str):
    match = IMAGE_RE.match(filename)

    if not match:
        return None

    frame_id, hour, minute, second, fruit_id = match.groups()
    timestamp = datetime.strptime(
        f"{date_text} {hour}:{minute}:{second}",
        "%d-%m-%Y %H:%M:%S",
    )

    return frame_id, fruit_id.upper(), timestamp


def build_final_name(frame_id: str, fruit_id: str, timestamp: datetime):
    timestamp_text = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

    return f"{timestamp_text}_{frame_id}_{fruit_id}.png"


def unique_path(path: Path, used_paths: set[Path]):
    if path not in used_paths:
        used_paths.add(path)
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 2

    while True:
        candidate = path.with_name(f"{stem}__dup{counter}{suffix}")

        if candidate not in used_paths:
            used_paths.add(candidate)
            return candidate

        counter += 1


def collect_images(processed_root: Path, output_dir: Path, fruit_ids: list[str]):
    assigned_dirs = []

    for folder in processed_root.iterdir():
        if not folder.is_dir():
            continue

        date_text = parse_assigned_date(folder.name)

        if date_text is not None:
            assigned_dirs.append((datetime.strptime(date_text, "%d-%m-%Y"), date_text, folder))

    assigned_dirs.sort(key=lambda item: item[0])

    records = []
    skipped = []
    used_paths = set()

    for _, date_text, assigned_dir in assigned_dirs:
        for fruit_id in fruit_ids:
            fruit_dir = assigned_dir / fruit_id

            if not fruit_dir.exists():
                skipped.append((str(fruit_dir), "missing_fruit_folder"))
                continue

            image_files = sorted(
                fruit_dir.glob("*.png"),
                key=lambda path: (
                    parse_image_name(path.name, date_text)[2]
                    if parse_image_name(path.name, date_text)
                    else datetime.max
                ),
            )

            for image_path in image_files:
                parsed = parse_image_name(image_path.name, date_text)

                if parsed is None:
                    skipped.append((str(image_path), "invalid_filename"))
                    continue

                frame_id, filename_fruit_id, timestamp = parsed

                compressed_timestamp = compress_recording_timestamp(timestamp)

                if filename_fruit_id != fruit_id:
                    skipped.append((str(image_path), "fruit_id_mismatch"))
                    continue

                final_name = build_final_name(
                    frame_id=frame_id,
                    fruit_id=fruit_id,
                    timestamp=compressed_timestamp,
                )
                final_path = unique_path(output_dir / fruit_id / final_name, used_paths)

                records.append(
                    FinalImage(
                        source_path=image_path,
                        final_path=final_path,
                        source_date=date_text,
                        fruit_id=fruit_id,
                        frame_id=frame_id,
                        timestamp=compressed_timestamp,
                    )
                )

    return records, skipped


def clean_output_dir(output_dir: Path, processed_root: Path):
    output_dir = output_dir.resolve()
    processed_root = processed_root.resolve()

    if output_dir == processed_root or processed_root not in output_dir.parents:
        raise ValueError(
            f"Refusing to clean output outside processed root: {output_dir}"
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)


def write_manifest(output_dir: Path, records: list[FinalImage], skipped: list[tuple[str, str]]):
    manifest_path = output_dir / "final_manifest.csv"

    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "fruit_id",
                "timestamp",
                "source_date",
                "frame_id",
                "source_path",
                "final_path",
                "original_filename",
                "final_filename",
            ],
        )
        writer.writeheader()

        for record in sorted(records, key=lambda item: (item.fruit_id, item.timestamp)):
            writer.writerow(
                {
                    "fruit_id": record.fruit_id,
                    "timestamp": record.timestamp.isoformat(sep=" "),
                    "source_date": record.source_date,
                    "frame_id": record.frame_id,
                    "source_path": str(record.source_path),
                    "final_path": str(record.final_path),
                    "original_filename": record.source_path.name,
                    "final_filename": record.final_path.name,
                }
            )

    if skipped:
        skipped_path = output_dir / "final_skipped.csv"

        with skipped_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["path", "reason"])
            writer.writeheader()

            for path, reason in skipped:
                writer.writerow({"path": path, "reason": reason})

    return manifest_path


def consolidate(processed_root: Path, output_dir: Path, fruit_ids: list[str], clean: bool, dry_run: bool):
    if clean and not dry_run:
        clean_output_dir(output_dir=output_dir, processed_root=processed_root)

    for fruit_id in fruit_ids:
        if not dry_run:
            (output_dir / fruit_id).mkdir(parents=True, exist_ok=True)

    records, skipped = collect_images(
        processed_root=processed_root,
        output_dir=output_dir,
        fruit_ids=fruit_ids,
    )

    if not dry_run:
        for record in records:
            record.final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.source_path, record.final_path)

        manifest_path = write_manifest(output_dir, records, skipped)
    else:
        manifest_path = output_dir / "final_manifest.csv"

    counts = {fruit_id: 0 for fruit_id in fruit_ids}

    for record in records:
        counts[record.fruit_id] += 1

    return records, skipped, counts, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect images from data/02_processed/assigned_* into "
            "data/02_processed/final/F01..F06 with full datetime filenames."
        )
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Folder containing assigned_DD-MM-YYYY directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Final output folder.",
    )
    parser.add_argument(
        "--fruit-ids",
        nargs="+",
        default=DEFAULT_FRUIT_IDS,
        help="Fruit IDs to collect.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output folder before copying.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without writing files.",
    )
    args = parser.parse_args()

    processed_root = args.processed_root.resolve()
    output_dir = args.output_dir.resolve()
    fruit_ids = [fruit_id.upper() for fruit_id in args.fruit_ids]

    records, skipped, counts, manifest_path = consolidate(
        processed_root=processed_root,
        output_dir=output_dir,
        fruit_ids=fruit_ids,
        clean=args.clean,
        dry_run=args.dry_run,
    )

    action = "Would copy" if args.dry_run else "Copied"
    print(f"{action} {len(records)} images into {output_dir}")

    for fruit_id in fruit_ids:
        print(f"{fruit_id}: {counts[fruit_id]} images")

    print(f"Manifest: {manifest_path}")

    if skipped:
        print(f"Skipped: {len(skipped)} entries")
        print(f"Skipped report: {output_dir / 'final_skipped.csv'}")


if __name__ == "__main__":
    main()
