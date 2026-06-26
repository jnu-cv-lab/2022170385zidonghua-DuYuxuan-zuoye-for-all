from __future__ import annotations

import torch
from torch import nn


class SkeletonTransformer(nn.Module):
    """A compact Transformer encoder for [B, T, 132] skeleton sequences."""

    def __init__(
        self,
        input_dim: int = 132,
        target_frames: int = 30,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        num_classes: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.target_frames = target_frames
        self.embedding = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, target_frames, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"输入应为 [B,T,132]，实际为 {tuple(x.shape)}")
        x = self.embedding(x)
        x = x + self.position_embedding[:, : x.shape[1], :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)
