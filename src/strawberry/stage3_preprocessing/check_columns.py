from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# check columns in frame_manifest.csv
FRAME_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "02_processed"
    / "strawberry"
    / "manifests"
    / "frame_manifest.csv"
)

LABELS_CSV = (
    PROJECT_ROOT
    / "data"
    / "02_processed"
    / "strawberry"
    / "manifests"
    / "labels.csv"
)


def print_columns(csv_path):
    if not csv_path.exists():
        print(f"Missing file: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Columns in {csv_path.name}:")
    print(df.columns.tolist())


def main():
    print_columns(FRAME_MANIFEST)
    print_columns(LABELS_CSV)


if __name__ == "__main__":
    main()
