from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

class EHRTower(nn.Module):

    def __init__(self, feature_cols: List[str], cat_feature_cols: List[str], cat_cardinalities: List[int], repr_dim: int=512, d_model: int=192, n_heads: int=4, n_layers: int=3, dropout: float=0.1):
        super().__init__()
        self.feature_cols = feature_cols
        self.cat_feature_cols = cat_feature_cols
        self.repr_dim = repr_dim
        self.d_model = d_model
        cat_set = set(cat_feature_cols)
        self.cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_set]
        self.cont_indices = [i for i, c in enumerate(feature_cols) if c not in cat_set]
        n_cont = len(self.cont_indices)
        self.cont_weight = nn.Parameter(torch.randn(n_cont, d_model) / d_model ** 0.5) if n_cont > 0 else None
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card + 1, d_model, padding_idx=0) for card in cat_cardinalities])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.projector = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, repr_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        tokens: list[torch.Tensor] = []
        if self.cont_weight is not None:
            x_cont = x[:, self.cont_indices]
            cont_tok = x_cont.unsqueeze(-1) * self.cont_weight
            tokens.append(cont_tok)
        for i, emb in enumerate(self.cat_embeddings):
            idx = x[:, self.cat_indices[i]].long().clamp(min=0)
            tokens.append(emb(idx).unsqueeze(1))
        feat_tok = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        x_tok = torch.cat([cls, feat_tok], dim=1)
        x_enc = self.transformer(x_tok)
        return self.projector(x_enc[:, 0])

    @property
    def hidden_size(self) -> int:
        return self.repr_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.cls_token.dtype

    @property
    def device(self) -> torch.device:
        return self.cls_token.device
