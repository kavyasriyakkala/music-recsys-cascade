"""Tests for eval metrics: brute-force match, shuffled-labels canary, determinism."""

from __future__ import annotations

import random

import numpy as np
import pytest

from yc.data.synthetic import generate_dataset
from yc.eval.metrics import (
    _brute_ndcg_at_k,
    _brute_recall_at_k,
    ndcg_at_k,
    recall_at_k,
)
from yc.models.baselines import PopularityBaseline
from yc.splits.builder import DAY_UNITS, SplitConfig, build_snapshots


# ---------------------------------------------------------------------------
# Brute-force match tests
# ---------------------------------------------------------------------------

def _random_inputs(seed: int, n: int = 20, n_items: int = 100, k: int = 10):
    rng = random.Random(seed)
    recommended = [[rng.randint(0, n_items - 1) for _ in range(k * 2)] for _ in range(n)]
    relevant = [[rng.randint(0, n_items - 1) for _ in range(rng.randint(1, 5))] for _ in range(n)]
    return recommended, relevant


def test_recall_matches_brute_force() -> None:
    k = 10
    recommended, relevant = _random_inputs(seed=7, n=20, n_items=100, k=k)

    fast = recall_at_k(recommended, relevant, k=k)
    brute = [_brute_recall_at_k(r, set(rel), k) for r, rel in zip(recommended, relevant) if rel]
    expected_mean = float(np.mean(brute))

    assert abs(fast["mean"] - expected_mean) < 1e-9, (
        f"recall@{k} mismatch: fast={fast['mean']} brute={expected_mean}"
    )
    np.testing.assert_allclose(fast["per_user"], brute, atol=1e-9)


def test_ndcg_matches_brute_force() -> None:
    k = 10
    recommended, relevant = _random_inputs(seed=13, n=20, n_items=100, k=k)

    fast = ndcg_at_k(recommended, relevant, k=k)
    brute = [_brute_ndcg_at_k(r, set(rel), k) for r, rel in zip(recommended, relevant) if rel]
    expected_mean = float(np.mean(brute))

    assert abs(fast["mean"] - expected_mean) < 1e-9
    np.testing.assert_allclose(fast["per_user"], brute, atol=1e-9)


# ---------------------------------------------------------------------------
# Shuffled-labels canary
# ---------------------------------------------------------------------------

def test_shuffled_labels_canary() -> None:
    """Popularity recall@k on shuffled labels should collapse toward chance.

    chance_recall = k / n_candidate_items:
      each of the k recommended slots hits a positive with probability
      1 / n_candidate_items if labels are random, so expected hits = k * (1/N).
    """
    events, _, _ = generate_dataset(
        n_users=50, n_items=500, n_days=25, n_latent=3,
        embedding_dim=8, embedding_coverage=0.8, seed=1,
    )
    cfg = SplitConfig(n_days=25, min_prior_events=5)
    cutoff = 12 * DAY_UNITS
    snaps = build_snapshots(events, [cutoff], cfg)

    if not snaps:
        pytest.skip("No snapshots generated — dataset too small for these params")

    k = 10
    n_items = events["item_id"].n_unique()

    pop = PopularityBaseline()
    pop.fit(events, cutoff)
    uids = [s.uid for s in snaps]
    recs = pop.predict(uids, k=k)

    # Shuffle positive labels.
    rng = random.Random(42)
    all_positives = [s.positives for s in snaps]
    flat = [iid for pos in all_positives for iid in pos]
    rng.shuffle(flat)
    idx = 0
    shuffled: list[list[int]] = []
    for pos in all_positives:
        shuffled.append(flat[idx: idx + len(pos)])
        idx += len(pos)

    rec_lists = [recs[s.uid] for s in snaps]
    result = recall_at_k(rec_lists, shuffled, k=k)
    recall_shuffled = result["mean"]

    # Expected recall if recommendations are random: each of k slots hits
    # a positive with probability 1/n_candidate_items, so chance = k/n_items.
    chance_recall = k / n_items
    assert recall_shuffled <= 2 * chance_recall, (
        f"Shuffled-label recall {recall_shuffled:.4f} > 2x chance {chance_recall:.4f} — "
        "labels may not be shuffled correctly or n_items is very small"
    )


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

def test_determinism() -> None:
    """Same seed produces identical events, splits, and metric outputs."""
    def _run(seed: int):
        events, _, _ = generate_dataset(
            n_users=30, n_items=200, n_days=20, n_latent=3,
            embedding_dim=8, seed=seed,
        )
        cfg = SplitConfig(n_days=20, min_prior_events=3)
        cutoff = 8 * DAY_UNITS
        snaps = build_snapshots(events, [cutoff], cfg)
        pop = PopularityBaseline()
        pop.fit(events, cutoff)
        uids = [s.uid for s in snaps]
        recs = pop.predict(uids, k=10)
        rec_lists = [recs[s.uid] for s in snaps]
        pos_lists = [s.positives for s in snaps]
        return recall_at_k(rec_lists, pos_lists, k=10)["mean"]

    r1 = _run(99)
    r2 = _run(99)
    assert r1 == r2, f"Non-deterministic: {r1} != {r2}"

    r3 = _run(100)
    # Different seeds may coincidentally produce the same value; just run to confirm no crash.
    assert isinstance(r3, float)
