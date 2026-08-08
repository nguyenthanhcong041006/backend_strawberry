"""Build a deterministic 100-image development sample outside Stage 1.5 flow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage1_5_image_inventory.dev_sample_builder import (  # noqa: E402
    DevSampleConfig,
    DevSampleDatasetBuilder,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the dev sample builder."""
    parser = argparse.ArgumentParser(
        description="Create a deterministic small image dataset for development testing."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_1_5_dev_sample.json"),
        help="Path to the dev sample JSON config.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the standalone dev sample builder."""
    args = parse_args()
    config = DevSampleConfig.from_json(args.config)
    result = DevSampleDatasetBuilder(config).build()
    print("Stage 1.5 dev sample created.")
    print(f"Output root: {result.output_root}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Total candidates: {result.total_candidates}")
    print(f"Requested sample size: {result.requested_sample_size}")
    print(f"Copied: {result.copied_count}")
    print(f"Skipped existing: {result.skipped_existing_count}")
    print(f"Random seed: {result.random_seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

