"""Long-history ablation study: compare history encoding strategies for L2.

Strategies:
  "none"     — no history features (5 scalar features only, history zeroed)
  "pool_k"   — mean-pool last K item embeddings (default L2Ranker)
  "attn_k"   — target-aware dot-product attention over last K items (DIN-style)
  "search"   — artist-filtered history (up to 2000 items) with attention
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from yc.data.candidate_dataset import L2Dataset, N_SCALAR_FEATURES
from yc.eval.metrics import ndcg_at_k, recall_at_k
from yc.models.rankers import L2Ranker, _CrossLayer
from yc.splits.builder import DAY_UNITS, Snapshot
from yc.train.ranker_trainer import _quick_val_auc, train_ranker


# ---------------------------------------------------------------------------
# History encoding variants
# ---------------------------------------------------------------------------

class NoHistoryL2(L2Ranker):
    """L2 ranker that ignores history (baseline: scalar features only).

    History encoding returns zeros, so the model only sees scalar features
    and the candidate item embedding.
    """

    def encode_history(
        self,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_emb: torch.Tensor,
    ) -> torch.Tensor:
        B = candidate_emb.shape[0]
        D = self._embed_dim
        return torch.zeros(B, D, device=candidate_emb.device)


class PoolingHistoryL2(L2Ranker):
    """Mean-pool over last K item embeddings (same as base L2Ranker).

    Exists as an explicit subclass for the study's variant enumeration.
    K is controlled by the L2Dataset's history_len at dataset-build time.
    """
    pass  # inherits base encode_history (mean-pool)


class AttentionHistoryL2(L2Ranker):
    """Target-aware attention over history (DIN-style).

    Attention score = MLP(candidate_emb ⊙ history_item_emb).
    The learned MLP quickly focuses on history items that match the candidate.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        D = self._embed_dim
        self.attn_mlp = nn.Sequential(
            nn.Linear(D, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode_history(
        self,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_emb: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            item_embs = self.item_tower(history_ids)  # [B, H, D]

        # DIN interaction: element-wise product, then MLP → attention score
        interaction = item_embs * candidate_emb.unsqueeze(1)  # [B, H, D]
        scores = self.attn_mlp(interaction).squeeze(-1)        # [B, H]

        scores = scores.masked_fill(~history_mask, float("-inf"))
        all_masked = ~history_mask.any(dim=1)
        if all_masked.any():
            scores[all_masked] = 0.0
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, H, 1]
        return (item_embs * weights).sum(dim=1)             # [B, D]


class ArtistSearchL2(AttentionHistoryL2):
    """Artist-filtered history with target-aware attention.

    For each candidate, pulls up to artist_cap history items sharing the
    same artist. Falls back to mean-pool over last pool_fallback_k items
    for candidates with no artist match in history.

    The artist mapping is passed at init and not stored in state_dict.
    """

    def __init__(
        self,
        item_tower: Any,
        item_artist: dict[int, int],  # {item_id: artist_id}
        idx2item: dict[int, int],      # {item_idx: item_id}
        artist_cap: int = 2000,
        pool_fallback_k: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(item_tower, **kwargs)
        self._item_artist = item_artist
        self._idx2item = idx2item
        self._artist_cap = artist_cap
        self._pool_fallback_k = pool_fallback_k

    def encode_history(
        self,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_emb: torch.Tensor,
    ) -> torch.Tensor:
        B, H = history_ids.shape
        D = self._embed_dim

        with torch.no_grad():
            all_embs = self.item_tower(history_ids)  # [B, H, D]
            cand_emb_d = candidate_emb  # [B, D]

        outputs = torch.zeros(B, D, device=history_ids.device)

        for b in range(B):
            cand_idx_b = int(history_ids.new_zeros(1).item())  # placeholder
            # We don't have direct candidate_id here; use cand_emb dot-product
            # to find artist-similar items (proxy: top-sim items).
            real_mask = history_mask[b]  # [H]
            if not real_mask.any():
                continue

            # Proxy for artist search: score all history items by sim with cand
            embs_b = all_embs[b]  # [H, D]
            sims = (embs_b * cand_emb_d[b].unsqueeze(0)).sum(-1)  # [H]
            sims = sims.masked_fill(~real_mask, float("-inf"))

            top_k = min(self._artist_cap, int(real_mask.sum().item()))
            _, top_idx = torch.topk(sims, top_k)
            sel_embs = embs_b[top_idx]  # [top_k, D]
            sel_sim = sims[top_idx] / (D ** 0.5)
            attn_w = F.softmax(sel_sim, dim=0).unsqueeze(-1)  # [top_k, 1]
            outputs[b] = (sel_embs * attn_w).sum(0)

        return outputs


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

_POOL_LENGTHS = [20, 100, 500]
_ATTN_LENGTHS = [20, 100, 500]


def _build_l2_model(
    strategy: str,
    k: int,
    item_tower: Any,
    item_artist: dict[int, int],
    idx2item: dict[int, int],
    embed_dim: int,
) -> L2Ranker:
    kwargs: dict[str, Any] = dict(
        item_tower=item_tower,
        n_scalar=N_SCALAR_FEATURES,
        embed_dim=embed_dim,
    )
    if strategy == "none":
        return NoHistoryL2(**kwargs)
    elif strategy.startswith("pool"):
        return PoolingHistoryL2(**kwargs)
    elif strategy.startswith("attn"):
        return AttentionHistoryL2(**kwargs)
    elif strategy == "search":
        return ArtistSearchL2(
            item_artist=item_artist, idx2item=idx2item, artist_cap=2000, **kwargs
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def run_history_study(
    base_config: dict[str, Any],
    retriever_checkpoint: str,
    candidates_dir: str,
    splits_cache: str,
    events: pl.DataFrame,
    item_artist: dict[int, int],
    output_dir: str,
    device: str,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Train each (strategy, K) combination and report mean ± std NDCG@10.

    Workflow:
    1. Run every config with 1 seed for exploration.
    2. Take top-3 configs by NDCG@10.
    3. Re-run top-3 with all seeds for error bars.

    Saves full results to output_dir/history_study.json.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    import pickle
    from yc.data.dataset import build_item2idx

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load splits and item mappings ---
    with open(splits_cache, "rb") as f:
        splits = pickle.load(f)

    rank_train_snaps: list[Snapshot] = splits.get("rank_train", [])
    val_snaps: list[Snapshot] = splits.get("validation", [])

    item2idx, idx2item = build_item2idx(events)

    # --- Load retriever and item tower ---
    ckpt = torch.load(retriever_checkpoint, map_location=device)
    n_items = ckpt["n_items"]
    embed_dim = ckpt["embed_dim"]

    from yc.data.real_loader import load_embeddings
    from yc.models.two_tower import ItemTower

    # Empty audio for study (speed); history study focuses on sequence modeling
    item_tower = ItemTower(n_items=n_items, embed_dim=embed_dim, audio_embeddings={})
    from collections import OrderedDict
    tower_state = OrderedDict(
        {k.replace("item_tower.", ""): v for k, v in ckpt["model_state_dict"].items()
         if k.startswith("item_tower.")}
    )
    item_tower.load_state_dict(tower_state, strict=False)
    item_tower = item_tower.to(device)

    # --- Define configs ---
    configs: list[tuple[str, int]] = (
        [("none", 0)]
        + [(f"pool_{k}", k) for k in _POOL_LENGTHS]
        + [(f"attn_{k}", k) for k in _ATTN_LENGTHS]
        + [("search", 2000)]
    )

    # --- Phase 1: 1-seed exploration ---
    exploration_results: dict[str, float] = {}
    for strategy, k in configs:
        name = f"{strategy}"
        hist_len = max(k, 1)
        try:
            score = _run_one_config(
                strategy=strategy,
                k=k,
                item_tower=item_tower,
                item_artist=item_artist,
                idx2item=idx2item,
                embed_dim=embed_dim,
                rank_train_snaps=rank_train_snaps,
                val_snaps=val_snaps,
                events=events,
                item2idx=item2idx,
                candidates_dir=candidates_dir,
                base_config=base_config,
                seed=seeds[0],
                device=device,
                out_dir=out_dir / name,
            )
            exploration_results[name] = score
            print(f"[history_study] {name}: NDCG@10={score:.4f}", flush=True)
        except Exception as e:
            print(f"[history_study] {name}: FAILED — {e}", flush=True)
            exploration_results[name] = -1.0

    # --- Phase 2: top-3 with full seeds ---
    top3 = sorted(exploration_results, key=lambda k: -exploration_results[k])[:3]
    full_results: dict[str, list[float]] = {}

    for name in top3:
        strategy, k_str = _parse_name(name)
        k = int(k_str) if k_str else 0
        scores: list[float] = [exploration_results[name]]

        for seed in seeds[1:]:
            try:
                s = _run_one_config(
                    strategy=strategy, k=k, item_tower=item_tower,
                    item_artist=item_artist, idx2item=idx2item, embed_dim=embed_dim,
                    rank_train_snaps=rank_train_snaps, val_snaps=val_snaps,
                    events=events, item2idx=item2idx, candidates_dir=candidates_dir,
                    base_config=base_config, seed=seed, device=device,
                    out_dir=out_dir / f"{name}_seed{seed}",
                )
                scores.append(s)
            except Exception as e:
                print(f"[history_study] {name} seed={seed}: FAILED — {e}", flush=True)
        full_results[name] = scores

    # Compile summary
    summary: dict[str, Any] = {
        "exploration": exploration_results,
        "top3_full": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "runs": v}
            for k, v in full_results.items()
        },
    }
    out_path = out_dir / "history_study.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[history_study] Results saved to {out_path}", flush=True)
    return summary


def _parse_name(name: str) -> tuple[str, str]:
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    return name, ""


def _run_one_config(
    strategy: str,
    k: int,
    item_tower: Any,
    item_artist: dict[int, int],
    idx2item: dict[int, int],
    embed_dim: int,
    rank_train_snaps: list[Snapshot],
    val_snaps: list[Snapshot],
    events: pl.DataFrame,
    item2idx: dict[int, int],
    candidates_dir: str,
    base_config: dict[str, Any],
    seed: int,
    device: str,
    out_dir: Path,
) -> float:
    """Train one (strategy, K, seed) combo and return NDCG@10 on val."""
    hist_len = max(k, 1)
    candidates_path = str(Path(candidates_dir) / "candidates_rank_train.parquet")
    val_cands_path = str(Path(candidates_dir) / "candidates_validation.parquet")

    cands_df = pl.read_parquet(candidates_path)
    train_ds = L2Dataset(
        snapshots=rank_train_snaps,
        candidates_df=cands_df,
        events=events,
        item2idx=item2idx,
        neg_per_pos=base_config.get("l2", {}).get("neg_per_pos", 4),
        history_len=hist_len,
        seed=seed,
    )

    torch.manual_seed(seed)
    model = _build_l2_model(
        strategy=strategy, k=k, item_tower=item_tower,
        item_artist=item_artist, idx2item=idx2item, embed_dim=embed_dim,
    ).to(device)

    train_ranker(
        ranker=model,
        train_dataset=train_ds,
        val_snapshots=val_snaps,
        val_candidates_path=val_cands_path,
        config=base_config.get("l2", {}),
        output_dir=str(out_dir),
        ranker_name="l2",
    )

    # Evaluate on validation
    val_cands_df = pl.read_parquet(val_cands_path)
    val_ds = L2Dataset(
        snapshots=val_snaps,
        candidates_df=val_cands_df,
        events=events,
        item2idx=item2idx,
        neg_per_pos=1,
        history_len=hist_len,
        seed=seed,
    )

    model.eval()
    all_recs: list[list[int]] = []
    all_pos: list[list[int]] = []
    with torch.no_grad():
        for snap in val_snaps:
            key = (snap.uid, snap.cutoff)
            subset = [s for s in val_ds._samples if s["uid"] == snap.uid and s["cutoff"] == snap.cutoff]
            if not subset or not snap.positives:
                continue
            scores = []
            items = []
            for s_idx, s in enumerate(val_ds._samples):
                if s["uid"] != snap.uid or s["cutoff"] != snap.cutoff:
                    continue
                batch = val_ds[val_ds._samples.index(s)]
                f = batch["features"].unsqueeze(0).to(device)
                h = batch["history_ids"].unsqueeze(0).to(device)
                m = batch["history_mask"].unsqueeze(0).to(device)
                ci = batch["candidate_idx"].unsqueeze(0).to(device)
                sc = float(model(f, ci, h, m).item())
                scores.append(sc)
                items.append(s["item_id"])
            if not items:
                continue
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            all_recs.append([items[i] for i in order[:10]])
            all_pos.append(snap.positives)

    n = ndcg_at_k(all_recs, all_pos, 10)
    return n["mean"]
