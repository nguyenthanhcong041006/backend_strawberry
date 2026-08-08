from pathlib import Path
import argparse

from .runner import EDAConfig, EDARunner


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Run EDA pipeline')
    parser.add_argument('--root', type=str, default=None, help='Project root path')
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else None
    config = EDAConfig(project_root=root)
    runner = EDARunner(config)
    runner.run()


if __name__ == '__main__':
    main()
