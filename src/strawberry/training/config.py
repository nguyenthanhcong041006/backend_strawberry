from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    backbone: str
    recurrent: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "A": ModelSpec(key="A", backbone="efficientnet_b0", recurrent="gru"),
    "B": ModelSpec(key="B", backbone="mobilenet_v2", recurrent="lstm"),
    "C": ModelSpec(key="C", backbone="efficientnet_b0", recurrent="lstm"),
    "D": ModelSpec(key="D", backbone="mobilenet_v2", recurrent="gru"),
}


@dataclass
class TrainingConfig:
    run_name: str = "strawberry_loocv"
    seq_len: int = 5
    max_gap_hours: float = 1.0
    fusion_mode: str = "late_env_branch"
    temporal_pooling: str = "last_mean_max"
    image_size: int = 224
    batch_size: int = 4
    epochs: int = 18
    patience: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_size: int = 160
    env_hidden_size: int = 64
    dropout: float = 0.25
    backbone_dropout: float = 0.10
    grad_clip: float = 1.0
    seed: int = 42
    amp: bool = True
    num_workers: int = 0
    pretrained_backbone: bool = True
    target_transform: str = "robust_zscore"
    env_feature_mode: str = "sensor"
    env_tolerance_minutes: int = 45
    checkpoint_root: str = "models/strawberry/runs"
    artifact_root: str = "output/runs/strawberry"
    model_keys: list[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    image_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None = None) -> "TrainingConfig":
        if not mapping:
            return cls()
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {key: value for key, value in mapping.items() if key in fields}
        if "model_keys" in filtered and isinstance(filtered["model_keys"], tuple):
            filtered["model_keys"] = list(filtered["model_keys"])
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def checkpoint_root_path(self, project_root: Path) -> Path:
        path = Path(self.checkpoint_root)
        return path if path.is_absolute() else project_root / path

    def artifact_root_path(self, project_root: Path) -> Path:
        path = Path(self.artifact_root)
        return path if path.is_absolute() else project_root / path


def load_training_config(config_path: str | Path | None = None) -> TrainingConfig:
    if config_path is None:
        return TrainingConfig()
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return TrainingConfig.from_mapping(payload)

