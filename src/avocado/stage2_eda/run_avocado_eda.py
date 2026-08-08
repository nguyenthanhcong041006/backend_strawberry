from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURES_CSV = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "eda" / "features" / "avocado_features.csv"
GRAPHS_DIR = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "eda" / "graphs"


def has_existing_outputs() -> bool:
    return FEATURES_CSV.exists() and any(GRAPHS_DIR.rglob("*.png"))


def run_step(script_path: Path) -> None:
    subprocess.run([sys.executable, str(script_path)], check=True)


def run(force: bool = False) -> None:
    if not force and has_existing_outputs():
        print(f"EDA outputs already exist in {GRAPHS_DIR.parent}. Skipping.")
        return

    extract_script = PROJECT_ROOT / "src" / "avocado" / "stage2_eda" / "extract_features.py"
    graphs_script = PROJECT_ROOT / "src" / "avocado" / "stage2_eda" / "generate_eda_graphs.py"

    run_step(extract_script)
    run_step(graphs_script)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run avocado EDA feature extraction and graph generation.")
    parser.add_argument("--force", action="store_true", help="Rebuild EDA outputs even if they already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
