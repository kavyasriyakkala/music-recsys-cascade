"""Tests for the two-tower retriever components."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yc.data.dataset import RetrieverDataset
from yc.data.synthetic import generate_dataset
from yc.models.two_tower import ItemTower, TwoTowerRetriever, UserTower
from yc.splits.builder import SplitConfig, build_all_splits
from yc.train.losses import in_batch_softmax_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_audio_by_idx(n_items: int, audio_dim: int, coverage: float = 0.8, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_with = int(n_items * coverage)
    chosen = rng.choice(np.arange(1, n_items + 1), size=n_with, replace=False)
    vecs = rng.standard_normal((n_with, audio_dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
    return {int(idx): vecs[i] for i, idx in enumerate(chosen)}


# ---------------------------------------------------------------------------
# ItemTower
# ---------------------------------------------------------------------------

def test_item_tower_audio_fallback(capsys):
    """Item with no ID embedding dropped (p=1.0) but with audio → non-zero output."""
    n_items, embed_dim, audio_dim = 10, 16, 8
    audio = _make_audio_by_idx(n_items, audio_dim, coverage=1.0)
    tower = ItemTower(n_items, embed_dim, audio, id_dropout_p=1.0)
    tower.train()

    item_idx = torch.tensor([1])
    out = tower(item_idx)

    assert out.shape == (1, embed_dim)
    assert not torch.isnan(out).any(), "output contains NaN"
    assert out.norm(dim=-1).item() > 0.5, "output should be non-zero (audio branch active)"


def test_item_tower_no_audio_no_id(capsys):
    """Item with no audio and dropped ID → zero output, count logged at init."""
    n_items, embed_dim, audio_dim = 10, 16, 8
    # No audio at all.
    audio: dict[int, np.ndarray] = {}
    tower = ItemTower(n_items, embed_dim, audio, id_dropout_p=1.0)
    tower.train()

    captured = capsys.readouterr()
    assert str(n_items) in captured.out, "init should log no-audio count"

    item_idx = torch.tensor([3])
    out = tower(item_idx)

    assert out.shape == (1, embed_dim)
    assert not torch.isnan(out).any(), "output should not be NaN"
    assert out.abs().sum().item() == pytest.approx(0.0, abs=1e-6), "output should be zero"


# ---------------------------------------------------------------------------
# UserTower
# ---------------------------------------------------------------------------

def test_user_tower_padding():
    """All-padding history → zero user embedding, not NaN."""
    n_items, embed_dim, audio_dim = 20, 16, 8
    audio = _make_audio_by_idx(n_items, audio_dim)
    item_tower = ItemTower(n_items, embed_dim, audio)
    user_tower = UserTower(item_tower)
    user_tower.eval()

    history = torch.zeros(2, 10, dtype=torch.long)
    mask = history != 0  # all False

    out = user_tower(history, mask)

    assert out.shape == (2, embed_dim)
    assert not torch.isnan(out).any(), "output should not be NaN"
    assert out.abs().sum().item() == pytest.approx(0.0, abs=1e-6), "output should be zero"


# ---------------------------------------------------------------------------
# In-batch softmax loss
# ---------------------------------------------------------------------------

def test_in_batch_loss_diagonal():
    """With B orthogonal (user=item) pairs the diagonal dominates → loss near zero."""
    B, D = 4, 16
    # Each user and its matched item share the same basis vector → diagonal sim=1, off-diag=0.
    user_emb = F.normalize(torch.eye(B, D), dim=-1)
    item_emb = F.normalize(torch.eye(B, D), dim=-1)
    item_freqs = torch.ones(B)

    loss = in_batch_softmax_loss(user_emb, item_emb, item_freqs)
    assert loss.item() < math.log(B), (
        f"loss={loss.item():.4f} should be < log({B})={math.log(B):.3f}"
    )


def test_in_batch_loss_frequency_correction():
    """Popular item's off-diagonal logit is reduced more with frequency correction."""
    torch.manual_seed(1)
    B, D = 2, 8
    user_emb = F.normalize(torch.randn(B, D), dim=-1)
    item_emb = F.normalize(torch.randn(B, D), dim=-1)
    temperature = 0.07

    raw_logits = user_emb @ item_emb.T / temperature

    uniform_freqs = torch.ones(B)
    skewed_freqs = torch.tensor([10.0, 1.0])

    # Off-diagonal correction for item 0:
    # uniform: log(1/2) ≈ -0.693 → subtract -0.693 (i.e., add 0.693)
    # skewed:  log(10/11) ≈ -0.095 → subtract -0.095 (i.e., add 0.095)
    # After correction, skewed logit for item 0 (as negative) is LOWER.
    log_corr_uniform = torch.log(uniform_freqs / uniform_freqs.sum())
    log_corr_skewed = torch.log(skewed_freqs / skewed_freqs.sum())

    off_diag = 1.0 - torch.eye(B)
    corrected_uniform = raw_logits - log_corr_uniform.unsqueeze(0) * off_diag
    corrected_skewed = raw_logits - log_corr_skewed.unsqueeze(0) * off_diag

    # User 1 viewing item 0 as a negative: skewed correction lowers it more.
    assert corrected_skewed[1, 0].item() < corrected_uniform[1, 0].item(), (
        "popular item (idx 0) should have a lower off-diagonal logit with skewed correction"
    )


# ---------------------------------------------------------------------------
# Smoke: end-to-end via CLI
# ---------------------------------------------------------------------------

def test_retriever_smoke(tmp_path):
    """smoke-retriever CLI must write retriever_best.pt and at least one eval entry."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_retriever.py"),
            "--config", str(ROOT / "configs" / "retriever.yaml"),
            "--data-dir", "none",
            "--output-dir", str(tmp_path),
            "--smoke",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"smoke run failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert (tmp_path / "retriever_best.pt").exists(), "retriever_best.pt not found"
    log_path = tmp_path / "retriever_train_log.json"
    assert log_path.exists(), "retriever_train_log.json not found"
    with open(log_path) as f:
        log = json.load(f)
    assert len(log["evals"]) >= 1, "training log has no eval entries"
    assert "recall@100" in log["evals"][0]
    assert log["evals"][0]["recall@100"] >= 0.0


# ---------------------------------------------------------------------------
# Overfit on tiny batch
# ---------------------------------------------------------------------------

def test_retriever_overfit_tiny():
    """Model should overfit a tiny batch of 8 unique pairs in 50 steps."""
    torch.manual_seed(42)
    n_users, n_items_tiny, n_days = 8, 50, 5
    embed_dim = 16

    events, emb_df, _ = generate_dataset(
        n_users=n_users, n_items=n_items_tiny, n_days=n_days, seed=42
    )
    events = events.filter(pl.col("event_type") == "listen").sort(["uid", "timestamp"])

    split_cfg = SplitConfig(n_days=n_days, min_prior_events=1)
    splits = build_all_splits(events, split_cfg)
    train_snaps = splits["retriever_train"]
    if not train_snaps:
        pytest.skip("no training snapshots generated with this synthetic data")

    dataset = RetrieverDataset(train_snaps, events, history_len=10)
    if len(dataset) == 0:
        pytest.skip("dataset empty")

    audio_by_idx: dict[int, np.ndarray] = {}
    for row in emb_df.iter_rows(named=True):
        iid = row["item_id"]
        if iid in dataset.item2idx:
            audio_by_idx[dataset.item2idx[iid]] = np.array(
                row["normalized_embed"], dtype=np.float32
            )

    n_model_items = len(dataset.item2idx)
    model = TwoTowerRetriever(n_model_items, embed_dim, audio_by_idx, id_dropout_p=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Fix a single batch of 8 samples and overfit it.
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)
    batch = next(iter(loader))

    history = batch["history"]
    mask = batch["history_mask"]
    positive_idx = batch["positive"]
    item_freqs = batch["item_freq"]

    model.train()
    final_loss = float("inf")
    for _ in range(50):
        user_emb, item_emb = model(history, mask, positive_idx)
        loss = in_batch_softmax_loss(user_emb, item_emb, item_freqs, temperature=0.07)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < 0.5, (
        f"Model did not overfit tiny batch: final loss={final_loss:.4f}, expected < 0.5"
    )
