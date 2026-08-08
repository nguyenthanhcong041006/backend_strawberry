"""Command-line entry point for Stage 1.5 image inventory.

Keep this script thin: parse arguments, validate config, and call
Stage15Pipeline after validation passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage1_5_image_inventory.config import Stage15Config  # noqa: E402
from stage1_5_image_inventory.config_validator import Stage15ConfigValidator  # noqa: E402
from stage1_5_image_inventory.logging_utils import Stage15Logger  # noqa: E402
from stage1_5_image_inventory.pipeline import Stage15Pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse Stage 1.5 command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Stage 1.5 image inventory.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_1_5_image_inventory.yaml"),
        help="Path to the Stage 1.5 config YAML file.",
    )
    return parser.parse_args()


def main() -> int:
    """Run config loading, validation, logging, and the current pipeline."""
    args = parse_args()
    try:
        config = Stage15Config.from_yaml(args.config)
    except Exception as exc:
        print(f"Stage 1.5 config load failed: {exc}", file=sys.stderr)
        return 1

    logs_root = config.resolve_path(config.logging.get("logs_root", "logs/stage_1_5"))
    logger = Stage15Logger.create(logs_root)
    logger.info("Stage 1.5 phase started.")
    logger.info(f"Config path: {args.config}")

    validation_result = Stage15ConfigValidator().run(config)
    if validation_result.success:
        logger.info(validation_result.message)
    else:
        logger.error(validation_result.message)
    for warning in validation_result.warnings:
        logger.warning(warning)
    for error in validation_result.errors:
        logger.error(error)

    if not validation_result.success:
        logger.info("Stage 1.5 phase stopped before pipeline execution.")
        logger.close()
        print("Stage 1.5 config validation failed.")
        print(f"Log file: {logger.log_path}")
        return 1

    pipeline = Stage15Pipeline(config=config, logger=logger)
    run_result = pipeline.run()
    logger.info("Stage 1.5 pipeline checkpoint run completed.")
    logger.info(f"Tool results collected: {len(run_result.tool_results)}")
    logger.info(f"Discovered files: {run_result.total_files}")
    logger.info(f"Readable files: {run_result.readable_files}")
    logger.info(f"Unreadable files: {run_result.unreadable_files}")
    if run_result.inventory_csv_path:
        logger.info(f"Inventory CSV: {run_result.inventory_csv_path}")
    if run_result.report_path:
        logger.info(f"Summary report: {run_result.report_path}")
    if run_result.numeric_crosscheck_report_path:
        logger.info(f"Numeric coverage report: {run_result.numeric_crosscheck_report_path}")
    if run_result.debug_report_path:
        logger.info(f"Debug report: {run_result.debug_report_path}")
    logger.close()

    print("Stage 1.5 config validation passed.")
    print(f"Discovered files: {run_result.total_files}")
    print(f"Readable files: {run_result.readable_files}")
    print(f"Unreadable files: {run_result.unreadable_files}")
    if run_result.inventory_csv_path:
        print(f"Inventory CSV: {run_result.inventory_csv_path}")
    if run_result.report_path:
        print(f"Summary report: {run_result.report_path}")
    if run_result.numeric_crosscheck_report_path:
        print(f"Numeric coverage report: {run_result.numeric_crosscheck_report_path}")
    if run_result.debug_report_path:
        print(f"Debug report: {run_result.debug_report_path}")
    print(f"Log file: {logger.log_path}")
    print("Implemented checkpoint: full Stage 1.5 pipeline with numeric coverage cross-check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
