"""Day-based snapshot builder for Yambda-cascade.

All timestamps are in 5-second units (global; UNVERIFIED for real data).
A snapshot captures: user, cutoff time, positive items in [cutoff, cutoff+1day),
and which of those are novel (never listened before cutoff).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

DAY_UNITS: int = 86400 // 5  # 17280


class LeakageError(Exception):
    """Raised when a snapshot's history contains an event at or after cutoff."""


@dataclass
class SplitConfig:
    n_days: int
    min_prior_events: int = 50
    label_hours: int = 24
    # Yambda dataset card: a listen is positive if played_ratio_pct > 50
    listen_plus_threshold: float = 0.5


@dataclass
class Snapshot:
    uid: int
    cutoff: int  # 5-sec timestamp units
    positives: list[int] = field(default_factory=list)
    novel_positives: list[int] = field(default_factory=list)
    # history timestamps kept so leakage checks can run without re-reading data
    _history_timestamps: list[int] = field(default_factory=list, repr=False)


def check_leakage(snapshots: list[Snapshot]) -> None:
    """Raise LeakageError if any history event has timestamp >= cutoff."""
    for snap in snapshots:
        if snap._history_timestamps:
            max_ts = max(snap._history_timestamps)
            if max_ts >= snap.cutoff:
                raise LeakageError(
                    f"uid={snap.uid} cutoff={snap.cutoff}: "
                    f"history contains timestamp {max_ts} >= cutoff"
                )


def build_snapshots(
    events: pl.DataFrame,
    cutoff_timestamps: list[int],
    config: SplitConfig,
) -> list[Snapshot]:
    """Build one snapshot per eligible user per cutoff.

    Eligibility: >= config.min_prior_events events strictly before cutoff,
    and >= 1 positive in the label window.
    Uses searchsorted on sorted timestamps — no post-hoc filtering.
    """
    label_units = int(config.label_hours * 3600 / 5)
    threshold_pct = config.listen_plus_threshold * 100

    snapshots: list[Snapshot] = []

    # Pre-sort events by (uid, timestamp) once.
    events_sorted = events.sort(["uid", "timestamp"])

    for cutoff in cutoff_timestamps:
        label_end = cutoff + label_units

        # Split into history (< cutoff) and label window [cutoff, label_end).
        history = events_sorted.filter(pl.col("timestamp") < cutoff)
        label_window = events_sorted.filter(
            (pl.col("timestamp") >= cutoff) & (pl.col("timestamp") < label_end)
        )

        # Positives: listen events in label window with played_ratio_pct above threshold.
        positives_df = label_window.filter(
            (pl.col("event_type") == "listen")
            & (pl.col("played_ratio_pct") > threshold_pct)
        )

        # Count prior events per user.
        prior_counts = (
            history.group_by("uid")
            .agg(pl.len().alias("n_prior"))
            .filter(pl.col("n_prior") >= config.min_prior_events)
        )
        eligible_uids = set(prior_counts["uid"].to_list())

        # Positive items per user in label window.
        pos_per_user: dict[int, list[int]] = {}
        if positives_df.height > 0:
            for row in positives_df.select(["uid", "item_id"]).iter_rows():
                uid, iid = row
                if uid in eligible_uids:
                    pos_per_user.setdefault(uid, [])
                    if iid not in pos_per_user[uid]:
                        pos_per_user[uid].append(iid)

        # Items each user listened to (any event_type==listen) before cutoff.
        listen_history = history.filter(pl.col("event_type") == "listen")
        prior_listens: dict[int, set[int]] = {}
        if listen_history.height > 0:
            for row in listen_history.select(["uid", "item_id"]).iter_rows():
                uid, iid = row
                prior_listens.setdefault(uid, set()).add(iid)

        # History timestamps per user (for leakage checks).
        hist_ts_per_user: dict[int, list[int]] = {}
        if history.height > 0:
            for row in history.select(["uid", "timestamp"]).iter_rows():
                uid, ts = row
                hist_ts_per_user.setdefault(uid, []).append(ts)

        for uid, positives in pos_per_user.items():
            prior_items = prior_listens.get(uid, set())
            novel = [iid for iid in positives if iid not in prior_items]
            snapshots.append(
                Snapshot(
                    uid=uid,
                    cutoff=cutoff,
                    positives=positives,
                    novel_positives=novel,
                    _history_timestamps=hist_ts_per_user.get(uid, []),
                )
            )

    return snapshots


def build_all_splits(
    events: pl.DataFrame,
    config: SplitConfig,
) -> dict[str, list[Snapshot]]:
    """Build all five splits.

    Window boundaries (measured back from the final day N = config.n_days - 1):
      retriever_train:  label days 0 .. N-16
      retriever_val:    label day  N-15
      rank_train:       label days N-14 .. N-3   (12 days, multi-snapshot)
      validation:       label day  N-2
      test:             label day  N

    cutoff for label day d = d * DAY_UNITS (start-of-day).
    build_snapshots is called once per split with the full list of cutoffs for
    that window; for rank_train that is 12 cutoffs × eligible users snapshots.
    """
    N = config.n_days - 1  # index of the last day

    def day_cutoff(d: int) -> int:
        return d * DAY_UNITS

    retriever_train_cutoffs = [day_cutoff(d) for d in range(0, N - 15)]
    retriever_val_cutoffs = [day_cutoff(N - 15)]
    rank_train_cutoffs = [day_cutoff(d) for d in range(N - 14, N - 2)]
    validation_cutoffs = [day_cutoff(N - 2)]
    test_cutoffs = [day_cutoff(N)]

    return {
        "retriever_train": build_snapshots(events, retriever_train_cutoffs, config),
        "retriever_val": build_snapshots(events, retriever_val_cutoffs, config),
        "rank_train": build_snapshots(events, rank_train_cutoffs, config),
        "validation": build_snapshots(events, validation_cutoffs, config),
        "test": build_snapshots(events, test_cutoffs, config),
    }


def label_day_sets(splits: dict[str, list[Snapshot]]) -> dict[str, set[int]]:
    """Return the set of label-day indices for each split (for overlap checks)."""

    def _label_day(snap: Snapshot) -> int:
        return snap.cutoff // DAY_UNITS

    return {
        name: {_label_day(s) for s in snaps} for name, snaps in splits.items()
    }
