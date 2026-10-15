"""Tests for split correctness: leakage, label window, non-overlap, eligibility, novel flags."""

from __future__ import annotations

import random

import polars as pl
import pytest

from yc.data.synthetic import generate_dataset
from yc.splits.builder import (
    DAY_UNITS,
    LeakageError,
    Snapshot,
    SplitConfig,
    build_all_splits,
    build_snapshots,
    check_leakage,
    label_day_sets,
)

# Tiny config for fast tests.
SMALL_CFG = SplitConfig(n_days=25, min_prior_events=5, label_hours=24, listen_plus_threshold=0.5)


@pytest.fixture(scope="module")
def small_events() -> pl.DataFrame:
    events, _, _ = generate_dataset(
        n_users=50, n_items=500, n_days=25, n_latent=3,
        embedding_dim=8, embedding_coverage=0.8, seed=0,
    )
    return events


@pytest.fixture(scope="module")
def all_splits(small_events: pl.DataFrame) -> dict:
    return build_all_splits(small_events, SMALL_CFG)


# ---------------------------------------------------------------------------
# Leakage tests
# ---------------------------------------------------------------------------

def test_leakage(all_splits: dict) -> None:
    """No history event should have timestamp >= cutoff."""
    for name, snaps in all_splits.items():
        # Sample for CI speed; full check runs via `make check-leakage`
        sample = random.Random(42).sample(snaps, min(500, len(snaps)))
        check_leakage(sample)  # must not raise


def test_leakage_catches_injection() -> None:
    """Leakage check must raise LeakageError when a history event equals cutoff."""
    cutoff = 1000
    bad_snapshot = Snapshot(
        uid=0,
        cutoff=cutoff,
        positives=[42],
        novel_positives=[42],
        _history_timestamps=[800, 900, cutoff],  # cutoff == deliberate leak
    )
    with pytest.raises(LeakageError):
        check_leakage([bad_snapshot])


# ---------------------------------------------------------------------------
# Label window test
# ---------------------------------------------------------------------------

def test_label_window(small_events: pl.DataFrame) -> None:
    """Every positive must have timestamp in [cutoff, cutoff + label_units)."""
    label_units = SMALL_CFG.label_hours * 3600 // 5
    threshold_pct = SMALL_CFG.listen_plus_threshold * 100

    cutoff = 10 * DAY_UNITS
    snaps = build_snapshots(small_events, [cutoff], SMALL_CFG)

    label_window = small_events.filter(
        (pl.col("timestamp") >= cutoff)
        & (pl.col("timestamp") < cutoff + label_units)
        & (pl.col("event_type") == "listen")
        & (pl.col("played_ratio_pct") > threshold_pct)
    )
    valid_items_per_user: dict[int, set[int]] = {}
    for row in label_window.select(["uid", "item_id"]).iter_rows():
        uid, iid = row
        valid_items_per_user.setdefault(uid, set()).add(iid)

    for snap in snaps:
        valid = valid_items_per_user.get(snap.uid, set())
        for iid in snap.positives:
            assert iid in valid, (
                f"uid={snap.uid} item={iid} not in label window [cutoff={cutoff}, +{label_units})"
            )


# ---------------------------------------------------------------------------
# Window non-overlap test
# ---------------------------------------------------------------------------

def test_window_non_overlap(all_splits: dict) -> None:
    """Label-day sets for all five splits must be pairwise disjoint."""
    day_sets = label_day_sets(all_splits)
    names = list(day_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = day_sets[a] & day_sets[b]
            assert not overlap, f"Splits '{a}' and '{b}' share label days: {overlap}"


# ---------------------------------------------------------------------------
# Min prior events eligibility
# ---------------------------------------------------------------------------

def test_min_prior_events(small_events: pl.DataFrame) -> None:
    """Users with fewer than min_prior_events before cutoff must not appear."""
    cutoff = 10 * DAY_UNITS
    snaps = build_snapshots(small_events, [cutoff], SMALL_CFG)

    prior_counts = (
        small_events.filter(pl.col("timestamp") < cutoff)
        .group_by("uid")
        .agg(pl.len().alias("n_prior"))
    )
    count_map = dict(zip(prior_counts["uid"].to_list(), prior_counts["n_prior"].to_list()))

    for snap in snaps:
        n = count_map.get(snap.uid, 0)
        assert n >= SMALL_CFG.min_prior_events, (
            f"uid={snap.uid} has only {n} prior events but appeared in snapshots"
        )


# ---------------------------------------------------------------------------
# Novel positives test
# ---------------------------------------------------------------------------

def test_novel_positives(small_events: pl.DataFrame) -> None:
    """Novel flag must be exactly those positives the user never listened before cutoff."""
    cutoff = 12 * DAY_UNITS
    snaps = build_snapshots(small_events, [cutoff], SMALL_CFG)

    prior_listens = (
        small_events.filter(
            (pl.col("timestamp") < cutoff) & (pl.col("event_type") == "listen")
        )
        .group_by("uid")
        .agg(pl.col("item_id").alias("prior_items"))
    )
    prior_map: dict[int, set[int]] = {}
    for row in prior_listens.iter_rows(named=True):
        prior_map[row["uid"]] = set(row["prior_items"])

    for snap in snaps:
        prior = prior_map.get(snap.uid, set())
        expected_novel = {iid for iid in snap.positives if iid not in prior}
        assert set(snap.novel_positives) == expected_novel, (
            f"uid={snap.uid}: novel_positives mismatch"
        )
