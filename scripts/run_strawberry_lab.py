from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from strawberry.training.lab_search import run_lab_search


if __name__ == "__main__":
    run_lab_search(PROJECT_ROOT / "configs" / "strawberry_training.json")

