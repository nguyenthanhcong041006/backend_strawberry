from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainingConfig, load_training_config
from .engine import train_loocv_run
from .evaluate import summarize_run
from .lab_search import run_lab_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strawberry RUL training and lab utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run LOOCV training for strawberry models")
    train_parser.add_argument("--config", type=Path, default=None, help="Path to a JSON training config")
    train_parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model keys A B C D")

    eval_parser = subparsers.add_parser("evaluate", help="Summarize a completed training run")
    eval_parser.add_argument("--run-root", type=Path, required=True, help="Path to the run root directory")

    lab_parser = subparsers.add_parser("lab", help="Run the small-model lab sweep")
    lab_parser.add_argument("--config", type=Path, default=None, help="Optional JSON config for the lab")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = load_training_config(args.config)
        train_loocv_run(config, model_keys=args.models)
        return

    if args.command == "evaluate":
        summarize_run(args.run_root)
        return

    if args.command == "lab":
        run_lab_search(args.config)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

