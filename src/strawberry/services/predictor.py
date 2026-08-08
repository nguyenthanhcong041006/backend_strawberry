from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np
import torch

from strawberry.training.models import build_model

from utils_app.image_utils import numpy_to_tensor, remove_alpha


class FruitRULPredictor:
    def __init__(self, config: dict, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.fruit_type = config.get("active_dataset", "strawberry")
        self.image_size = int(config.get("image", {}).get("crop_width", 224)) # fallback to crop_width or 224
        self.device = self._resolve_device(config.get("device", "cpu"))
        self.seq_len = int(config.get("model", {}).get("seq_len", 5))
        
        # Look for model_path in model dict or root config
        provided_model_path = config.get("model", {}).get("path") or config.get("model_path")
        self.model_path = self._resolve_model_path(provided_model_path)
        self.model: torch.nn.Module | None = None
        self.checkpoint: dict | None = None
        self.target_scaler: dict[str, float] | None = None
        self.env_stats: dict[str, np.ndarray] | None = None
        self._lock = Lock()

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)

    def _resolve_model_path(self, model_path: str | None) -> Path:
        if not model_path:
            # Fallback based on fruit type
            if self.fruit_type == "avocado":
                return self.project_root / "models" / "avocado" / "numeric_baselines" / "best_model.pth"
            return self.project_root / "models" / "strawberry" / "production" / "model_D" / "best_model.pt"
        path = Path(model_path)
        return path if path.is_absolute() else self.project_root / path

    def _load_checkpoint(self) -> dict:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unexpected checkpoint format: {self.model_path}")
        return checkpoint

    def _ensure_model(self) -> torch.nn.Module:
        if self.model is not None:
            return self.model

        with self._lock:
            if self.model is not None:
                return self.model

            checkpoint = self._load_checkpoint()
            self.checkpoint = checkpoint
            self.target_scaler = checkpoint.get("target_scaler")
            raw_env_stats = checkpoint.get("env_stats") or {}
            env_mean = np.asarray(raw_env_stats.get("mean", [0.0, 0.0]), dtype=np.float32)
            env_std = np.asarray(raw_env_stats.get("std", [1.0, 1.0]), dtype=np.float32)
            env_std = np.where((~np.isfinite(env_std)) | (env_std < 1e-6), 1.0, env_std)
            self.env_stats = {"mean": env_mean, "std": env_std}
            checkpoint_config = checkpoint.get("config", {})
            self.seq_len = int(checkpoint_config.get("seq_len", self.seq_len))
            model_key = str(
                checkpoint.get("model_key")
                or checkpoint_config.get("model_key")
                or self.config.get("model", {}).get("key", "D")
            )

            if self.fruit_type == "strawberry":
                model = build_model(
                    model_key,
                    hidden_size=int(checkpoint_config.get("hidden_size", self.config.get("model", {}).get("hidden_size", 160))),
                    env_hidden_size=int(checkpoint_config.get("env_hidden_size", self.config.get("model", {}).get("env_hidden_size", 64))),
                    dropout=float(checkpoint_config.get("dropout", self.config.get("model", {}).get("dropout", 0.25))),
                    backbone_dropout=float(checkpoint_config.get("backbone_dropout", self.config.get("model", {}).get("backbone_dropout", 0.1))),
                    temporal_pooling=str(checkpoint_config.get("temporal_pooling", self.config.get("model", {}).get("temporal_pooling", "last_mean_max"))),
                    fusion_mode=str(checkpoint_config.get("fusion_mode", self.config.get("model", {}).get("fusion_mode", "late_env_branch"))),
                    pretrained_backbone=False,
                ).to(self.device)
            else:
                # TODO: instantiate avocado models
                raise NotImplementedError(f"Model for {self.fruit_type} not fully implemented yet.")

            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint.get("model")
            if not isinstance(state_dict, dict):
                raise ValueError(f"Checkpoint does not contain a model_state_dict: {self.model_path}")
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            self.model = model
            return model

    def _build_env_sequence(
        self,
        *,
        temperature_c: float | None,
        humidity_pct: float | None,
    ) -> torch.Tensor:
        stats = self.env_stats or {
            "mean": np.zeros(2, dtype=np.float32),
            "std": np.ones(2, dtype=np.float32),
        }
        values = np.asarray(
            [
                np.nan if temperature_c is None else float(temperature_c),
                np.nan if humidity_pct is None else float(humidity_pct),
            ],
            dtype=np.float32,
        )
        missing = (~np.isfinite(values)).astype(np.float32)
        filled = np.where(missing > 0, stats["mean"], values)
        normalized = (filled - stats["mean"]) / stats["std"]
        normalized = np.where(missing > 0, 0.0, normalized)
        env_row = np.concatenate([normalized, missing], axis=0).astype(np.float32)
        return torch.from_numpy(env_row).to(self.device).view(1, 1, 4).repeat(1, self.seq_len, 1)

    def predict(
        self,
        segmented_fruit: np.ndarray,
        *,
        temperature_c: float | None = None,
        humidity_pct: float | None = None,
    ) -> tuple[float, float]:
        model = self._ensure_model()
        clean_image = remove_alpha(segmented_fruit)
        image_tensor = numpy_to_tensor(clean_image, self.image_size, self.device)
        images_seq = image_tensor.unsqueeze(0).unsqueeze(0).repeat(1, self.seq_len, 1, 1, 1)
        env_seq = self._build_env_sequence(temperature_c=temperature_c, humidity_pct=humidity_pct)

        with torch.no_grad():
            prediction = model(images_seq, env_seq)

        remaining_useful_life = float(prediction.squeeze().item())
        if self.target_scaler:
            center = float(self.target_scaler.get("center", 0.0))
            scale = float(self.target_scaler.get("scale", 1.0)) or 1.0
            remaining_useful_life = remaining_useful_life * scale + center
        confidence = self._estimate_prediction_confidence(remaining_useful_life)
        return remaining_useful_life, confidence

    @staticmethod
    def _estimate_prediction_confidence(remaining_useful_life: float) -> float:
        if not np.isfinite(remaining_useful_life):
            return 0.0
        # The regression checkpoint returns RUL only, so this is a bounded sanity score.
        if remaining_useful_life < 0:
            return 0.55
        return 0.9
