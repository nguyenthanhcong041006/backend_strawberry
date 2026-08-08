import sys
from pathlib import Path

STRAWBERRY_DIR = Path(__file__).resolve().parent.parent

if str(STRAWBERRY_DIR) not in sys.path:
    sys.path.insert(0, str(STRAWBERRY_DIR))

from app import app