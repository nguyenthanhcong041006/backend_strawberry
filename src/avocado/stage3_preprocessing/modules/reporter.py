import csv
from pathlib import Path


def generate_report(results: list[dict], report_path: Path) -> None:
    """
    Writes out the CSV report containing the frame path, fruit id, and mask/segmented paths.
    """
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "fruit_id", "mask_path", "segmented_path"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
