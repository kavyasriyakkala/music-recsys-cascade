"""L1Ranker (MLP) and L2Ranker (DCN-v2) for the Yambda-cascade pipeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from yc.data.candidate_dataset import N_SCALAR_FEATURES


class L1Ranker(nn.Module):
    """Lightweight MLP ranker on 5 scalar features. Trims 1000 → top-100.

    Architecture:
        Linear(5, 64) → BatchNorm → ReLU →
        Linear(64, 64) → BatchNorm → ReLU →
        Linear(64, 1) → Sigmoid
    """

    def __init__(
        self,
        n_features: int = N_SCALAR_FEATURES,
        hidden_dim: int = 64,
        dropout: float = 0.0,  # unused; BatchNorm provides regularisation
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_features] → [B, 1]"""
        return self.net(x)


class _CrossLayer(nn.Module):
    """Single DCN-v2 cross layer: x_0 ⊙ (W x_i + b) + x_i."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=True)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.W(x) + x


class L2Ranker(nn.Module):
    """DCN-v2 ranker with mean-pooled history encoding. Ranks top-100.

    Input:
      - scalar_feats: [B, 5] — same cheap features as L1
      - candidate_idx: [B] — item2idx index for the candidate
      - history_ids:   [B, H] — item2idx indices (left-padded with 0)
      - history_mask:  [B, H] — True for real items

    Frozen ItemTower weights from the retriever provide item embeddings.

    Architecture (input_dim = 5 + 2*embed_dim):
      Cross network: 3 DCN-v2 layers on input_dim
      Deep network:  input_dim → 256 → 128 → 64
      Output:        concat(cross_out, deep_out) → Linear(input_dim+64, 1) → Sigmoid
    """

    def __init__(
        self,
        item_tower: Any,  # yc.models.two_tower.ItemTower, frozen
        n_scalar: int = N_SCALAR_FEATURES,
        embed_dim: int = 128,
        n_cross_layers: int = 3,
    ) -> None:
        super().__init__()
        self.item_tower = item_tower
        for p in self.item_tower.parameters():
            p.requires_grad_(False)

        self._embed_dim = embed_dim
        input_dim = n_scalar + 2 * embed_dim

        self.cross_layers = nn.ModuleList(
            [_CrossLayer(input_dim) for _ in range(n_cross_layers)]
        )

        self.deep = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.output = nn.Sequential(
            nn.Linear(input_dim + 64, 1),
            nn.Sigmoid(),
        )

    def encode_history(
        self,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool over history item embeddings. Subclasses may override."""
        item_embs = self.item_tower(history_ids)  # [B, H, D]
        mask_f = history_mask.float().unsqueeze(-1)  # [B, H, 1]
        pooled = (item_embs * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)  # [B, D]
        return pooled

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        """Cross + deep + output from concatenated input vector."""
        x0 = xi = x
        for layer in self.cross_layers:
            xi = layer(x0, xi)
        deep_out = self.deep(x)  # [B, 64]
        return self.output(torch.cat([xi, deep_out], dim=-1))  # [B, 1]

    def forward(
        self,
        scalar_feats: torch.Tensor,
        candidate_idx: torch.Tensor,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """→ [B, 1] probabilities."""
        with torch.no_grad():
            cand_emb = self.item_tower(candidate_idx)  # [B, D]
        hist_emb = self.encode_history(history_ids, history_mask, cand_emb)  # [B, D]
        x = torch.cat([scalar_feats, hist_emb, cand_emb], dim=-1)  # [B, input_dim]
        return self._score(x)
