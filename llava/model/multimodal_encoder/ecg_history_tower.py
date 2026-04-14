from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn

class ECGHistoryTower(nn.Module):

    def __init__(self, ecg_dim: int, repr_dim: int=512, dropout: float=0.1):
        super().__init__()
        self.ecg_dim = ecg_dim
        self.repr_dim = repr_dim
        self.time_decay_scale = nn.Parameter(torch.tensor(1.0))
        self.adapter = nn.Sequential(nn.Linear(ecg_dim, ecg_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ecg_dim, repr_dim), nn.LayerNorm(repr_dim))
        self.no_history_token = nn.Parameter(torch.randn(repr_dim) * 0.02)

    def forward(self, ecg_embeddings: Optional[torch.Tensor], days_before: Optional[torch.Tensor]) -> torch.Tensor:
        if ecg_embeddings is None or ecg_embeddings.size(0) == 0:
            return self.no_history_token
        lam = self.time_decay_scale.abs()
        weights = torch.exp(-lam * days_before.float() / 96.0)
        weights = weights / weights.sum().clamp(min=1e-08)
        pooled = (weights.unsqueeze(-1) * ecg_embeddings).sum(0)
        return self.adapter(pooled)

    def forward_batch(self, ecg_emb_list: List[Optional[torch.Tensor]], days_list: List[Optional[torch.Tensor]]) -> torch.Tensor:
        outs = [self.forward(e, d) for e, d in zip(ecg_emb_list, days_list)]
        return torch.stack(outs)

    @property
    def hidden_size(self) -> int:
        return self.repr_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.no_history_token.dtype

    @property
    def device(self) -> torch.device:
        return self.no_history_token.device
