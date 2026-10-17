"""RetrieverDataset and LazySnapshotIterator for the two-tower retriever."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, IterableDataset

if TYPE_CHECKING:
    from yc.splits.builder import Snapshot, SplitConfig


def build_item2idx(
    events: pl.DataFrame,
) -> tuple[dict[int, int], dict[int, int]]:
    """Build item2idx and idx2item from events; index 0 is reserved for padding."""
    all_item_ids = sorted(events["item_id"].unique().to_list())
    item2idx: dict[int, int] = {iid: idx + 1 for idx, iid in enumerate(all_item_ids)}
    idx2item: dict[int, int] = {v: k for k, v in item2idx.items()}
    return item2idx, idx2item


def _build_user_event_index(
    events: pl.DataFrame,
) -> dict[int, list[tuple[int, int]]]:
    """Return {uid: [(timestamp, item_id), ...]} sorted by timestamp."""
    index: dict[int, list[tuple[int, int]]] = {}
    for uid, ts, iid in events.sort("timestamp").select(
        ["uid", "timestamp", "item_id"]
    ).iter_rows():
        index.setdefault(uid, []).append((int(ts), int(iid)))
    return index


class RetrieverDataset(Dataset):
    """Training dataset for the two-tower retriever.

    Each sample is (user_history, positive_item_id).  Item ids are mapped to
    1-based indices; 0 is reserved for padding.
    """

    def __init__(
        self,
        snapshots: list[Snapshot],
        events: pl.DataFrame,
        history_len: int = 100,
    ) -> None:
        super().__init__()
        self.history_len = history_len

        # Build item2idx: sorted for determinism; 0 reserved for padding.
        all_item_ids = sorted(events["item_id"].unique().to_list())
        self.item2idx: dict[int, int] = {
            iid: idx + 1 for idx, iid in enumerate(all_item_ids)
        }
        self.idx2item: dict[int, int] = {v: k for k, v in self.item2idx.items()}

        # Pre-group events by user.
        user_event_index = _build_user_event_index(events)

        # Count item frequencies as positives (for SBC loss correction).
        _freq: dict[int, int] = {}

        # Build flat sample list: (padded_history_indices, positive_idx).
        self._samples: list[tuple[list[int], int]] = []

        for snap in snapshots:
            uid = snap.uid
            cutoff = snap.cutoff
            user_evts = user_event_index.get(uid, [])

            hist_items = [
                iid for ts, iid in user_evts if ts < cutoff
            ][-history_len:]

            hist_indices = [self.item2idx.get(iid, 0) for iid in hist_items]
            padded = [0] * (history_len - len(hist_indices)) + hist_indices

            for pos_iid in snap.positives:
                pos_idx = self.item2idx.get(pos_iid, 0)
                if pos_idx == 0:
                    continue
                self._samples.append((padded, pos_idx))
                _freq[pos_idx] = _freq.get(pos_idx, 0) + 1

        self._item_freq = _freq

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        hist, pos = self._samples[idx]
        hist_t = torch.tensor(hist, dtype=torch.long)
        return {
            "history": hist_t,
            "history_mask": hist_t != 0,
            "positive": torch.tensor(pos, dtype=torch.long),
            "item_freq": torch.tensor(
                float(self._item_freq.get(pos, 1)), dtype=torch.float32
            ),
        }


class LazySnapshotIterator(IterableDataset):
    """IterableDataset for retriever training that builds snapshots one day at a time.

    Avoids materializing all training-window snapshots at once: only one cutoff
    day's worth of Snapshot objects are in memory at any point.
    Shuffles cutoff order and within-day snapshot order each epoch.
    """

    def __init__(
        self,
        events: pl.DataFrame,
        config: SplitConfig,
        cutoff_timestamps: list[int],
        item2idx: dict[int, int],
        item_freq: dict[int, int],
        history_len: int = 100,
    ) -> None:
        super().__init__()
        self.events = events
        self.config = config
        self.cutoffs = cutoff_timestamps
        self.item2idx = item2idx
        self.item_freq = item_freq
        self.history_len = history_len

    def __iter__(self):
        from yc.splits.builder import build_snapshots

        user_event_index = _build_user_event_index(self.events)
        rng = np.random.default_rng()

        cutoffs = list(self.cutoffs)
        rng.shuffle(cutoffs)

        for cutoff in cutoffs:
            day_snaps = build_snapshots(self.events, [cutoff], self.config)
            rng.shuffle(day_snaps)

            for snap in day_snaps:
                uid = snap.uid
                user_evts = user_event_index.get(uid, [])
                hist_items = [
                    iid for ts, iid in user_evts if ts < cutoff
                ][-self.history_len :]
                hist_indices = [self.item2idx.get(iid, 0) for iid in hist_items]
                padded = [0] * (self.history_len - len(hist_indices)) + hist_indices
                hist_t = torch.tensor(padded, dtype=torch.long)
                mask_t = hist_t != 0

                for pos_iid in snap.positives:
                    pos_idx = self.item2idx.get(pos_iid, 0)
                    if pos_idx == 0:
                        continue
                    yield {
                        "history": hist_t,
                        "history_mask": mask_t,
                        "positive": torch.tensor(pos_idx, dtype=torch.long),
                        "item_freq": torch.tensor(
                            float(self.item_freq.get(pos_idx, 1)), dtype=torch.float32
                        ),
                    }

            del day_snaps
