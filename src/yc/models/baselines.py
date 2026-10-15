"""Popularity, user-most-listened, and user-most-recent baselines.

All baselines enforce that only events with timestamp < cutoff_ts are used
inside fit() — the caller does not need to pre-filter.
"""

from __future__ import annotations

import polars as pl

DAY_UNITS: int = 86400 // 5  # 17280


class PopularityBaseline:
    """Global item popularity over the last window_days before cutoff."""

    def __init__(self) -> None:
        self._ranked: list[int] = []

    def fit(
        self,
        events: pl.DataFrame,
        cutoff_ts: int,
        window_days: int = 7,
    ) -> None:
        window_start = cutoff_ts - window_days * DAY_UNITS
        window = events.filter(
            (pl.col("timestamp") >= window_start)
            & (pl.col("timestamp") < cutoff_ts)
            & (pl.col("event_type") == "listen")
        )
        counts = (
            window.group_by("item_id")
            .agg(pl.len().alias("cnt"))
            .sort(["cnt", "item_id"], descending=[True, False])
        )
        self._ranked = counts["item_id"].to_list()

    def predict(self, uids: list[int], k: int) -> dict[int, list[int]]:
        top_k = self._ranked[:k]
        return {uid: top_k for uid in uids}


class UserMostListenedBaseline:
    """For each user, items they listened to most often before cutoff."""

    def __init__(self) -> None:
        self._user_ranked: dict[int, list[int]] = {}
        self._global_fallback: list[int] = []

    def fit(self, events: pl.DataFrame, cutoff_ts: int) -> None:
        history = events.filter(
            (pl.col("timestamp") < cutoff_ts) & (pl.col("event_type") == "listen")
        )
        if history.height == 0:
            return

        counts = (
            history.group_by(["uid", "item_id"])
            .agg(pl.len().alias("cnt"))
            .sort(["uid", "cnt", "item_id"], descending=[False, True, False])
        )
        for uid in counts["uid"].unique(maintain_order=True).to_list():
            user_rows = counts.filter(pl.col("uid") == uid)
            self._user_ranked[uid] = user_rows["item_id"].to_list()

        global_counts = (
            history.group_by("item_id")
            .agg(pl.len().alias("cnt"))
            .sort(["cnt", "item_id"], descending=[True, False])
        )
        self._global_fallback = global_counts["item_id"].to_list()

    def predict(self, uids: list[int], k: int) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        for uid in uids:
            ranked = self._user_ranked.get(uid, self._global_fallback)
            result[uid] = ranked[:k]
        return result


class UserMostRecentBaseline:
    """For each user, the items they most recently listened to before cutoff."""

    def __init__(self) -> None:
        self._user_ranked: dict[int, list[int]] = {}
        self._global_fallback: list[int] = []

    def fit(self, events: pl.DataFrame, cutoff_ts: int) -> None:
        history = events.filter(
            (pl.col("timestamp") < cutoff_ts) & (pl.col("event_type") == "listen")
        )
        if history.height == 0:
            return

        # Sort descending by timestamp; keep first occurrence of each item per user.
        recent = (
            history.sort(["timestamp", "item_id"], descending=[True, False])
            .unique(subset=["uid", "item_id"], keep="first")
            .sort(["uid", "timestamp", "item_id"], descending=[False, True, False])
        )
        for uid in recent["uid"].unique(maintain_order=True).to_list():
            user_rows = recent.filter(pl.col("uid") == uid)
            self._user_ranked[uid] = user_rows["item_id"].to_list()

        global_recent = (
            history.sort(["timestamp", "item_id"], descending=[True, False])
            .unique(subset=["item_id"], keep="first")
        )
        self._global_fallback = global_recent["item_id"].to_list()

    def predict(self, uids: list[int], k: int) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        for uid in uids:
            ranked = self._user_ranked.get(uid, self._global_fallback)
            result[uid] = ranked[:k]
        return result
