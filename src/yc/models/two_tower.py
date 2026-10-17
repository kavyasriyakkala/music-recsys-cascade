"""Two-tower retriever: ItemTower, UserTower, TwoTowerRetriever."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemTower(nn.Module):
    """Maps item_idx -> L2-normalized item embedding.

    Combines a learned ID embedding with a projection of a precomputed audio
    embedding.  During training, the ID embedding is dropped with probability
    id_dropout_p so the model learns to rely on audio for cold-start items.
    """

    def __init__(
        self,
        n_items: int,
        embed_dim: int,
        audio_embeddings: dict[int, np.ndarray],
        id_dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        audio_dim: int
        if audio_embeddings:
            audio_dim = int(next(iter(audio_embeddings.values())).shape[0])
        else:
            audio_dim = embed_dim

        self.embed_dim = embed_dim
        self.id_dropout_p = id_dropout_p

        self.id_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.audio_proj = nn.Linear(audio_dim, embed_dim, bias=False)

        # Build audio buffer: shape (n_items+1, audio_dim); 0 for no-audio items.
        audio_buf = np.zeros((n_items + 1, audio_dim), dtype=np.float32)
        n_has_audio = 0
        for item_idx, vec in audio_embeddings.items():
            if 1 <= item_idx <= n_items:
                audio_buf[item_idx] = vec
                n_has_audio += 1

        self.register_buffer("audio_buffer", torch.from_numpy(audio_buf))

        n_no_audio = n_items - n_has_audio
        print(
            f"ItemTower init: {n_no_audio}/{n_items} items have no audio embedding"
        )
        self._n_no_audio = n_no_audio

    def forward(self, item_idx: torch.Tensor) -> torch.Tensor:
        id_emb = self.id_emb(item_idx)

        if self.training and self.id_dropout_p > 0.0:
            keep = (
                torch.rand(item_idx.shape, device=item_idx.device)
                > self.id_dropout_p
            ).float()
            id_emb = id_emb * keep.unsqueeze(-1)

        audio = self.audio_buffer[item_idx]
        out = id_emb + self.audio_proj(audio)

        # F.normalize returns zero for zero vectors when eps is set.
        return F.normalize(out, dim=-1, eps=1e-8)


class UserTower(nn.Module):
    """Maps (history, history_mask) -> L2-normalized user embedding.

    Shares the item embedding table with the ItemTower (weights tied).
    """

    def __init__(self, item_tower: ItemTower) -> None:
        super().__init__()
        self.item_tower = item_tower
        embed_dim = item_tower.embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(
        self, history: torch.Tensor, history_mask: torch.BoolTensor
    ) -> torch.Tensor:
        # Embed history items using the shared item tower.
        # Don't apply ID dropout here — history is queried, not trained.
        item_embs = self.item_tower.id_emb(history)  # [B, L, D]

        # Masked mean pooling.
        mask_f = history_mask.unsqueeze(-1).float()  # [B, L, 1]
        pooled = (item_embs * mask_f).sum(dim=1)  # [B, D]
        count = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = pooled / count  # [B, D]

        out = self.mlp(pooled)  # [B, D]

        # Zero out rows with no real history tokens before normalizing.
        has_content = history_mask.any(dim=1).unsqueeze(1).float()  # [B, 1]
        out = out * has_content
        return F.normalize(out, dim=-1, eps=1e-8)


class TwoTowerRetriever(nn.Module):
    """Wraps ItemTower and UserTower for training and retrieval."""

    def __init__(
        self,
        n_items: int,
        embed_dim: int,
        audio_embeddings: dict[int, np.ndarray],
        id_dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.item_tower = ItemTower(n_items, embed_dim, audio_embeddings, id_dropout_p)
        self.user_tower = UserTower(self.item_tower)

    def forward(
        self,
        history: torch.Tensor,
        history_mask: torch.BoolTensor,
        positive_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (user_emb, item_emb_positive); loss computed outside."""
        user_emb = self.user_tower(history, history_mask)
        item_emb = self.item_tower(positive_idx)
        return user_emb, item_emb

    def get_user_embedding(
        self, history: torch.Tensor, history_mask: torch.BoolTensor
    ) -> torch.Tensor:
        return self.user_tower(history, history_mask)

    def get_item_embedding(self, item_idx: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_idx)

    @torch.no_grad()
    def build_item_index(self, device: str = "cpu") -> torch.Tensor:
        """Compute all item embeddings in batches of 4096.

        Returns [n_items, D] tensor indexed so that row i corresponds to
        item_idx i+1 (row 0 = item_idx 1, ...).
        """
        self.eval()
        n_items = self.item_tower.id_emb.num_embeddings - 1  # exclude padding 0
        item_indices = torch.arange(1, n_items + 1, device=device)

        chunks: list[torch.Tensor] = []
        for start in range(0, n_items, 4096):
            batch = item_indices[start : start + 4096]
            chunks.append(self.item_tower(batch).cpu())

        return torch.cat(chunks, dim=0)
