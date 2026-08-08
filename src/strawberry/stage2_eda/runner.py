from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .io import find_project_root


@dataclass
class EDAConfig:
    project_root: Optional[Path] = None

    def __post_init__(self):
        if self.project_root is not None:
            self.project_root = find_project_root(Path(self.project_root))

    @property
    def root(self) -> Path:
        return self.project_root or find_project_root()

    @property
    def manifest_dir(self) -> Path:
        return self.root / 'data' / '02_processed' / 'strawberry' / 'manifests_mock'

    @property
    def report_dir(self) -> Path:
        return self.root / 'output' / 'reports' / 'eda'

    @property
    def graph_dir(self) -> Path:
        return self.root / 'output' / 'graphs' / 'eda'


class EDARunner:
    """Wrapper that runs the EDA pipeline using the existing module functions.

    This keeps configuration explicit while reusing `run_from_manifests()`.
    """

    def __init__(self, config: EDAConfig):
        self.config = config

    def run(self) -> None:
        # Import here to avoid circular imports at module import time
        from .eda import run_from_manifests

        run_from_manifests(self.config.root)
