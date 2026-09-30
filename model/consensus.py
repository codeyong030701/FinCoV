"""
piano/consensus_emission.py

FinCoV Consensus Learning Module.
Predicts note-level consensus emission probabilities from the 7-D note representation.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        L = x.shape[1]
        if L <= self.pe.shape[1]:
            return self.dropout(x + self.pe[:, :L])

        # Dynamic fallback for sequences longer than max_len.
        pe = torch.zeros(L, x.shape[2], device=x.device)
        pos = torch.arange(L, device=x.device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, x.shape[2], 2, device=x.device).float() * (-math.log(10000.0) / x.shape[2]))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return self.dropout(x + pe.unsqueeze(0))


class ConsensusEmissionTransformer(nn.Module):
    """
    Supervised note-level emission predictor.

    Input : 7-D note features [B, L, D]
    Output: logits [B, L, K], where K=6 means pad + fingers 1..5.
    """
    def __init__(
        self,
        feat_dim: int = 16,
        num_classes: int = 6,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.d_model = d_model

        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.head = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        h = self.feat_proj(features)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        logits = self.head(h)
        logits[:, :, 0] = -1e9
        return logits
