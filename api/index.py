import sys
from pathlib import Path

# Resolve project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRAWBERRY_DIR = PROJECT_ROOT / "src" / "strawberry"
SRC_DIR = PROJECT_ROOT / "src"

# Add directories to sys.path for proper import resolution on Vercel
for path in [str(STRAWBERRY_DIR), str(SRC_DIR), str(PROJECT_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from app import app
