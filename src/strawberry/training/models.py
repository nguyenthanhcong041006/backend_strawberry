from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V2_Weights
from torchvision.models import efficientnet_b0, mobilenet_v2

from strawberry.shared.cbam import CBAM

from .config import MODEL_SPECS, ModelSpec


def _safe_backbone(name: str, pretrained: bool) -> nn.Module:
    try:
        if name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            return efficientnet_b0(weights=weights)
        if name == "mobilenet_v2":
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            return mobilenet_v2(weights=weights)
    except Exception:
        if name == "efficientnet_b0":
            return efficientnet_b0(weights=None)
        if name == "mobilenet_v2":
            return mobilenet_v2(weights=None)
    raise ValueError(f"Unsupported backbone: {name}")


class VisualEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        *,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        backbone = _safe_backbone(backbone_name, pretrained)
        if backbone_name == "efficientnet_b0":
            features = backbone.features
            channels = 1280
        else:
            features = backbone.features
            channels = 1280
        self.features = features
        self.cbam = CBAM(in_channels=channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = channels

        if freeze_backbone:
            for parameter in self.features.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = self.cbam(features)
        pooled = self.pool(features).flatten(1)
        return self.dropout(pooled)


class StrawberryRULModel(nn.Module):
    def __init__(
        self,
        spec: ModelSpec,
        *,
        hidden_size: int = 160,
        env_hidden_size: int = 64,
        dropout: float = 0.25,
        backbone_dropout: float = 0.1,
        temporal_pooling: str = "last_mean_max",
        fusion_mode: str = "late_env_branch",
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        valid_fusion_modes = {"image_only", "early_concat", "late_env_branch", "gated_env_branch"}
        valid_pooling = {"last", "last_mean_max", "attention"}
        if fusion_mode not in valid_fusion_modes:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        if temporal_pooling not in valid_pooling:
            raise ValueError(f"Unsupported temporal pooling: {temporal_pooling}")

        self.spec = spec
        self.fusion_mode = fusion_mode
        self.temporal_pooling = temporal_pooling
        self.hidden_size = int(hidden_size)
        self.env_hidden_size = int(env_hidden_size)
        self.env_dim = 4

        self.visual_encoder = VisualEncoder(
            spec.backbone,
            pretrained=pretrained_backbone,
            dropout=backbone_dropout,
            freeze_backbone=freeze_backbone,
        )
        rnn_input_dim = self.visual_encoder.feature_dim
        if fusion_mode == "early_concat":
            rnn_input_dim += self.env_dim

        rnn_cls = nn.GRU if spec.recurrent == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=rnn_input_dim,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=1,
        )

        if self.temporal_pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_size // 2, 1),
            )
            temporal_dim = self.hidden_size
        elif self.temporal_pooling == "last_mean_max":
            temporal_dim = self.hidden_size * 3
        else:
            temporal_dim = self.hidden_size

        self.env_encoder: nn.Module | None = None
        self.env_gate: nn.Module | None = None
        head_input_dim = temporal_dim
        if fusion_mode in {"late_env_branch", "gated_env_branch"}:
            self.env_encoder = nn.Sequential(
                nn.Linear(14, env_hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(env_hidden_size, env_hidden_size),
                nn.SiLU(),
            )
            head_input_dim += env_hidden_size
            if fusion_mode == "gated_env_branch":
                self.env_gate = nn.Sequential(
                    nn.Linear(env_hidden_size, temporal_dim),
                    nn.Sigmoid(),
                )
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, max(self.hidden_size, 128)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(max(self.hidden_size, 128), max(self.hidden_size // 2, 64)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(max(self.hidden_size // 2, 64), 1),
        )

    def _summarize_env(self, env_seq: torch.Tensor) -> torch.Tensor:
        values = env_seq[..., :2]
        missing = env_seq[..., 2:]
        valid = 1.0 - missing
        safe_values = values * valid
        counts = valid.sum(dim=1).clamp_min(1.0)
        mean = safe_values.sum(dim=1) / counts
        first = values[:, 0, :]
        last = values[:, -1, :]
        delta = last - first
        pos_inf = torch.full_like(values, float("inf"))
        neg_inf = torch.full_like(values, float("-inf"))
        min_vals = torch.where(valid.bool(), values, pos_inf).min(dim=1).values
        max_vals = torch.where(valid.bool(), values, neg_inf).max(dim=1).values
        min_vals = torch.where(torch.isfinite(min_vals), min_vals, torch.zeros_like(min_vals))
        max_vals = torch.where(torch.isfinite(max_vals), max_vals, torch.zeros_like(max_vals))
        missing_ratio = missing.mean(dim=1)
        return torch.cat([first, last, mean, delta, min_vals, max_vals, missing_ratio], dim=1)

    def _pool_temporal(self, outputs: torch.Tensor, hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.temporal_pooling == "last":
            return outputs[:, -1, :]
        if self.temporal_pooling == "attention":
            weights = torch.softmax(self.attention(outputs).squeeze(-1), dim=1)
            return torch.sum(outputs * weights.unsqueeze(-1), dim=1)
        last = outputs[:, -1, :]
        mean = outputs.mean(dim=1)
        max_vals = outputs.max(dim=1).values
        return torch.cat([last, mean, max_vals], dim=1)

    def forward(self, images_seq: torch.Tensor, env_seq: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = images_seq.shape
        flattened = images_seq.view(batch_size * seq_len, channels, height, width)
        frame_features = self.visual_encoder(flattened).view(batch_size, seq_len, -1)

        if self.fusion_mode == "early_concat" and env_seq is not None:
            rnn_input = torch.cat([frame_features, env_seq], dim=2)
        else:
            rnn_input = frame_features

        rnn_out, hidden = self.rnn(rnn_input)
        temporal_repr = self._pool_temporal(rnn_out, hidden)

        if self.fusion_mode in {"late_env_branch", "gated_env_branch"} and env_seq is not None and self.env_encoder is not None:
            env_repr = self.env_encoder(self._summarize_env(env_seq))
            if self.env_gate is not None:
                temporal_repr = temporal_repr * self.env_gate(env_repr)
            temporal_repr = torch.cat([temporal_repr, env_repr], dim=1)

        return self.head(temporal_repr).squeeze(-1)


def build_model(
    model_key: str,
    *,
    hidden_size: int = 160,
    env_hidden_size: int = 64,
    dropout: float = 0.25,
    backbone_dropout: float = 0.1,
    temporal_pooling: str = "last_mean_max",
    fusion_mode: str = "late_env_branch",
    pretrained_backbone: bool = True,
    freeze_backbone: bool = False,
) -> StrawberryRULModel:
    key = model_key.upper()
    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model key: {model_key}")
    spec = MODEL_SPECS[key]
    return StrawberryRULModel(
        spec,
        hidden_size=hidden_size,
        env_hidden_size=env_hidden_size,
        dropout=dropout,
        backbone_dropout=backbone_dropout,
        temporal_pooling=temporal_pooling,
        fusion_mode=fusion_mode,
        pretrained_backbone=pretrained_backbone,
        freeze_backbone=freeze_backbone,
    )
