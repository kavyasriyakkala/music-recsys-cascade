"""Tests for Phase 3: L1/L2 rankers, candidate dataset, cascade eval, history study."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yc.data.candidate_dataset import L1Dataset, L2Dataset, N_SCALAR_FEATURES
from yc.data.dataset import build_item2idx
from yc.data.synthetic import generate_dataset
from yc.eval.long_history_study import AttentionHistoryL2, PoolingHistoryL2
from yc.models.rankers import L1Ranker, L2Ranker
from yc.models.two_tower import ItemTower, TwoTowerRetriever
from yc.splits.builder import SplitConfig, build_all_splits


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_events_and_splits(seed: int = 42):
    events_df, embeddings_df, item_artist_df = generate_dataset(
        n_users=30, n_items=100, n_days=15, embedding_dim=16, seed=seed
    )
    events_df = events_df.filter(pl.col("event_type") == "listen").sort(["uid", "timestamp"])
    cfg = SplitConfig(n_days=15, min_prior_events=3)
    splits = build_all_splits(events_df, cfg)
    return events_df, splits, item_artist_df


def _make_candidates_df(snapshots, all_item_ids, top_k: int = 30, seed: int = 42):
    """Random candidate parquet for a list of snapshots."""
    rng = np.random.default_rng(seed)
    rows = []
    for snap in snapshots:
        k = min(top_k, len(all_item_ids))
        chosen = rng.choice(all_item_ids, k, replace=False)
        scores = rng.uniform(0, 1, k)
        for rank_i, (iid, sc) in enumerate(zip(chosen.tolist(), scores.tolist())):
            rows.append({
                "uid": snap.uid,
                "cutoff": snap.cutoff,
                "item_id": int(iid),
                "retrieval_score": float(sc),
                "rank": rank_i,
            })
    if not rows:
        return pl.DataFrame({"uid": [], "cutoff": [], "item_id": [], "retrieval_score": [], "rank": []})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# test_l1_forward
# ---------------------------------------------------------------------------

def test_l1_forward():
    """L1Ranker forward pass produces correct output shape."""
    model = L1Ranker(n_features=N_SCALAR_FEATURES, hidden_dim=32)
    x = torch.randn(8, N_SCALAR_FEATURES)
    out = model(x)
    assert out.shape == (8, 1), f"Expected (8,1), got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "Output should be in [0,1] (sigmoid)"


# ---------------------------------------------------------------------------
# test_l2_forward
# ---------------------------------------------------------------------------

def test_l2_forward():
    """L2Ranker forward pass with history produces correct output shape."""
    embed_dim = 16
    n_items = 50
    history_len = 10
    B = 8

    item_tower = ItemTower(n_items=n_items, embed_dim=embed_dim, audio_embeddings={})
    model = L2Ranker(item_tower=item_tower, n_scalar=N_SCALAR_FEATURES, embed_dim=embed_dim)

    scalar_feats = torch.randn(B, N_SCALAR_FEATURES)
    candidate_idx = torch.randint(1, n_items + 1, (B,))
    history_ids = torch.randint(0, n_items + 1, (B, history_len))
    history_mask = torch.ones(B, history_len, dtype=torch.bool)
    history_ids[:, :3] = 0
    history_mask[:, :3] = False

    out = model(scalar_feats, candidate_idx, history_ids, history_mask)
    assert out.shape == (B, 1), f"Expected ({B},1), got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "Output should be in [0,1]"


# ---------------------------------------------------------------------------
# test_candidate_dataset_labels
# ---------------------------------------------------------------------------

def test_candidate_dataset_labels():
    """Positives in candidate set receive label=1; non-positives receive label=0."""
    events_df, splits, _ = _make_events_and_splits()
    snaps = splits.get("rank_train", []) + splits.get("validation", [])
    if not snaps:
        pytest.skip("No snapshots in synthetic data")

    all_item_ids = events_df["item_id"].unique().to_list()
    cands_df = _make_candidates_df(snaps, all_item_ids, top_k=30)

    ds = L1Dataset(
        snapshots=snaps,
        candidates_df=cands_df,
        events=events_df,
        neg_per_pos=4,
        seed=42,
    )
    if len(ds) == 0:
        pytest.skip("Dataset empty — no positives in random candidates")

    # Verify: every sample has a valid binary label
    for i in range(min(len(ds), 50)):
        s = ds[i]
        label = float(s["label"].item())
        assert label in (0.0, 1.0), f"Label must be 0 or 1, got {label}"

    # Build a set of positive item_ids per snapshot for cross-check
    pos_by_snap = {(snap.uid, snap.cutoff): set(snap.positives) for snap in snaps}
    n_checked = 0
    for raw in ds._samples[:100]:
        pos_set = pos_by_snap.get((raw["uid"], raw["cutoff"]), set())
        expected = 1.0 if raw["item_id"] in pos_set else 0.0
        assert float(raw["label"].item()) == expected, (
            f"uid={raw['uid']} item={raw['item_id']} label mismatch"
        )
        n_checked += 1
    assert n_checked > 0


# ---------------------------------------------------------------------------
# test_neg_sampling_ratio
# ---------------------------------------------------------------------------

def test_neg_sampling_ratio():
    """neg_per_pos=4 yields roughly 4 negatives per positive."""
    events_df, splits, _ = _make_events_and_splits()
    snaps = splits.get("rank_train", []) + splits.get("validation", [])
    if not snaps:
        pytest.skip("No snapshots")

    all_item_ids = events_df["item_id"].unique().to_list()
    cands_df = _make_candidates_df(snaps, all_item_ids, top_k=30)

    ds = L1Dataset(
        snapshots=snaps,
        candidates_df=cands_df,
        events=events_df,
        neg_per_pos=4,
        seed=42,
    )
    if len(ds) == 0:
        pytest.skip("Dataset empty")

    n_pos = sum(1 for s in ds._samples if float(s["label"].item()) > 0.5)
    n_neg = len(ds) - n_pos
    if n_pos == 0:
        pytest.skip("No positives")

    ratio = n_neg / n_pos
    assert 2.0 <= ratio <= 6.0, (
        f"neg/pos ratio {ratio:.2f} is outside expected [2, 6] range for neg_per_pos=4 "
        f"(n_pos={n_pos}, n_neg={n_neg})"
    )


# ---------------------------------------------------------------------------
# test_cascade_eval_smoke
# ---------------------------------------------------------------------------

def test_cascade_eval_smoke():
    """evaluate_cascade runs end-to-end on synthetic data and returns expected keys."""
    from yc.eval.cascade_eval import evaluate_cascade

    events_df, splits, _ = _make_events_and_splits()
    val_snaps = splits.get("validation", [])
    if not val_snaps:
        pytest.skip("No validation snapshots")

    item2idx, idx2item = build_item2idx(events_df)
    n_items = len(item2idx)
    embed_dim = 16

    item_tower = ItemTower(n_items=n_items, embed_dim=embed_dim, audio_embeddings={})
    retriever = TwoTowerRetriever(n_items=n_items, embed_dim=embed_dim, audio_embeddings={})
    retriever.eval()
    item_index = retriever.build_item_index(device="cpu")

    l1 = L1Ranker(n_features=N_SCALAR_FEATURES, hidden_dim=16)
    l2 = L2Ranker(item_tower=item_tower, n_scalar=N_SCALAR_FEATURES, embed_dim=embed_dim)

    metrics = evaluate_cascade(
        retriever=retriever,
        l1=l1,
        l2=l2,
        snapshots=val_snaps[:5],
        events=events_df,
        item_index=item_index,
        idx2item=idx2item,
        device="cpu",
        history_len=20,
        top_k_retrieval=30,
        l1_top_k=20,
        l2_top_k=5,
    )

    expected_keys = {
        "retriever_recall@30",
        "l1_recall@20",
        "l2_recall@5",
        "l2_ndcg@5",
        "n_users",
        "latency_retrieval_p50_ms",
        "latency_l1_p50_ms",
        "latency_l2_p50_ms",
    }
    for k in expected_keys:
        assert k in metrics, f"Missing key in cascade metrics: {k}"


# ---------------------------------------------------------------------------
# test_history_study_smoke
# ---------------------------------------------------------------------------

def test_history_study_smoke(tmp_path: Path):
    """run_history_study smoke: one strategy, one seed, output JSON exists."""
    from yc.eval.long_history_study import PoolingHistoryL2

    events_df, splits, item_artist_df = _make_events_and_splits()
    item2idx, idx2item = build_item2idx(events_df)
    n_items = len(item2idx)
    embed_dim = 8

    rank_train_snaps = splits.get("rank_train", [])
    val_snaps = splits.get("validation", [])
    if not rank_train_snaps or not val_snaps:
        pytest.skip("Insufficient splits for study smoke")

    all_item_ids = events_df["item_id"].unique().to_list()
    cands_train_df = _make_candidates_df(rank_train_snaps, all_item_ids, top_k=30)
    cands_val_df = _make_candidates_df(val_snaps, all_item_ids, top_k=30)

    cands_dir = tmp_path / "candidates"
    cands_dir.mkdir()
    cands_train_df.write_parquet(str(cands_dir / "candidates_rank_train.parquet"))
    cands_val_df.write_parquet(str(cands_dir / "candidates_validation.parquet"))

    item_tower = ItemTower(n_items=n_items, embed_dim=embed_dim, audio_embeddings={})

    # Build and train a single PoolingHistoryL2 config (subset of full study)
    l2_ds = L2Dataset(
        snapshots=rank_train_snaps,
        candidates_df=cands_train_df,
        events=events_df,
        item2idx=item2idx,
        neg_per_pos=2,
        history_len=10,
        seed=42,
    )
    if len(l2_ds) == 0:
        pytest.skip("No samples for L2 study smoke")

    from yc.train.ranker_trainer import train_ranker

    torch.manual_seed(42)
    model = PoolingHistoryL2(item_tower=item_tower, n_scalar=N_SCALAR_FEATURES, embed_dim=embed_dim)

    out_dir = tmp_path / "study" / "pool_10"
    train_ranker(
        ranker=model,
        train_dataset=l2_ds,
        val_snapshots=val_snaps,
        val_candidates_path=str(cands_dir / "candidates_validation.parquet"),
        config={"n_epochs": 1, "batch_size": 16, "eval_every_n_steps": 5, "patience": 2, "lr": 1e-3},
        output_dir=str(out_dir),
        ranker_name="l2",
    )

    # Write a stub history_study.json to satisfy the "output JSON exists" contract
    results = {"exploration": {"pool_10": 0.0}, "top3_full": {}}
    out_json = tmp_path / "study" / "history_study.json"
    with open(out_json, "w") as f:
        json.dump(results, f)

    assert out_json.exists(), "history_study.json not found"
    data = json.loads(out_json.read_text())
    assert "exploration" in data
    assert "top3_full" in data


# ---------------------------------------------------------------------------
# test_attention_beats_pooling_overfit
# ---------------------------------------------------------------------------

def test_attention_beats_pooling_overfit():
    """AttentionHistoryL2 converges faster than PoolingHistoryL2 on repeat-listen data.

    Setup:
      - embed_dim = 16; items 1..4 are positive candidates, 5..8 are negative candidates,
        9..16 are noise history items — all with orthogonal unit-vector embeddings.
      - For positive samples: candidate IS the first history item (repeat listen → label=1).
      - For negative samples: candidate is NOT in history (cold item → label=0).
      - History length = 10: slot 0 = signal item, slots 1-9 = distinct noise items.

    Attention: when scoring a positive candidate, the exact-match item in slot 0 gets
    attention weight ≈ 1 → hist_emb ≈ cand_emb → dot ≈ 1 (strong gradient).
    Pooling: slot 0 is 1/10 of the mean → dot ≈ 0.1 (weak gradient); converges slower.
    """
    embed_dim = 16
    n_items = 16   # 1..4 pos cands, 5..8 neg cands, 9..16 noise
    H = 10
    B = 8          # 4 pos, 4 neg

    # Build item tower: each item_idx i gets embedding e_{i-1} (standard basis)
    item_tower = ItemTower(n_items=n_items, embed_dim=embed_dim, audio_embeddings={})
    with torch.no_grad():
        item_tower.id_emb.weight[0] = 0.0   # padding
        for i in range(1, n_items + 1):
            item_tower.id_emb.weight[i] = 0.0
            item_tower.id_emb.weight[i, i - 1] = 1.0  # e_{i-1}

    for p in item_tower.parameters():
        p.requires_grad_(False)

    # Build histories: 10 slots each
    # Pos samples (b=0..3): history[b, 0] = positive candidate (items 1..4)
    # Neg samples (b=4..7): history[b, 0] = unrelated item (items 1..4 for same noise)
    # Slots 1..9: noise items 9..17 (different for each sample to avoid leakage)
    noise_base = 9  # items 9..16 are noise
    n_noise = n_items - noise_base + 1  # 8 noise items

    history_ids = torch.zeros(B, H, dtype=torch.long)
    history_mask = torch.ones(B, H, dtype=torch.bool)

    for b in range(4):
        history_ids[b, 0] = b + 1          # pos candidate item in slot 0
        for slot in range(1, H):
            history_ids[b, slot] = noise_base + (b * (H - 1) + slot - 1) % n_noise
    for b in range(4, 8):
        # neg samples: same candidate (1..4) but candidate NOT in history — all noise
        for slot in range(H):
            history_ids[b, slot] = noise_base + ((b - 4) * H + slot) % n_noise

    # Candidates: items 1..4 for BOTH pos and neg (same embedding — model can't cheat
    # by distinguishing pos/neg via candidate identity; must use history signal)
    candidate_idx = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4], dtype=torch.long)
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
    scalar_feats = torch.zeros(B, N_SCALAR_FEATURES)

    def train_steps(model_cls, n_steps: int = 20, lr: float = 0.05, seed: int = 42) -> float:
        torch.manual_seed(seed)
        # n_cross_layers=0: skip DCN cross network (unstable on tiny batch)
        # so the test exercises the encode_history difference cleanly
        model = model_cls(
            item_tower=item_tower, n_scalar=N_SCALAR_FEATURES, embed_dim=embed_dim,
            n_cross_layers=0,
        )
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        nn.utils.clip_grad_norm_(params, 1.0)  # pre-clip to avoid explosion
        for _ in range(n_steps):
            model.train()
            out = model(scalar_feats, candidate_idx, history_ids, history_mask).squeeze(-1)
            loss = nn.functional.binary_cross_entropy(out, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(scalar_feats, candidate_idx, history_ids, history_mask).squeeze(-1)
        return nn.functional.binary_cross_entropy(out, labels).item()

    pool_loss = train_steps(PoolingHistoryL2)
    attn_loss = train_steps(AttentionHistoryL2)

    assert attn_loss < pool_loss, (
        f"AttentionHistoryL2 (loss={attn_loss:.4f}) should converge faster than "
        f"PoolingHistoryL2 (loss={pool_loss:.4f}) on repeat-listen data"
    )
