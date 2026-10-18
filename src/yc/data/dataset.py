"""RetrieverDataset and LazySnapshotIterator for the two-tower retriever."""

from __future__ import annotations

import time
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

    __init__ builds only a flat index (snapshot_idx, pos_item_id); history
    tensors are constructed lazily in __getitem__ to avoid materialising
    O(n_snapshots × history_len) integers in memory upfront.
    """

    def __init__(
        self,
        snapshots: list[Snapshot],
        events: pl.DataFrame,
        history_len: int = 100,
    ) -> None:
        super().__init__()
        t0 = time.time()
        print(f"Building dataset index from {len(snapshots)} snapshots...", flush=True)

        self.history_len = history_len
        self.snapshots = snapshots

        # Build item2idx: sorted for determinism; 0 reserved for padding.
        all_item_ids = sorted(events["item_id"].unique().to_list())
        self.item2idx: dict[int, int] = {
            iid: idx + 1 for idx, iid in enumerate(all_item_ids)
        }
        self.idx2item: dict[int, int] = {v: k for k, v in self.item2idx.items()}

        # User event index for fast per-user lookup in __getitem__.
        self._user_event_index = _build_user_event_index(events)

        # Flat index: (snapshot_idx, pos_item_id) — no history tensors yet.
        self._flat: list[tuple[int, int]] = []
        _freq: dict[int, int] = {}

        for snap_idx, snap in enumerate(snapshots):
            for pos_iid in snap.positives:
                pos_idx = self.item2idx.get(pos_iid, 0)
                if pos_idx == 0:
                    continue
                self._flat.append((snap_idx, pos_iid))
                _freq[pos_idx] = _freq.get(pos_idx, 0) + 1

        self._item_freq = _freq
        print(
            f"Dataset index ready: {len(self)} samples in {time.time() - t0:.1f}s",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self._flat)

    def __getitem__(self, idx: int) -> dict:
        snap_idx, pos_iid = self._flat[idx]
        snap = self.snapshots[snap_idx]

        user_evts = self._user_event_index.get(snap.uid, [])
        hist_items = [iid for ts, iid in user_evts if ts < snap.cutoff][
            -self.history_len :
        ]
        hist_indices = [self.item2idx.get(iid, 0) for iid in hist_items]
        padded = [0] * (self.history_len - len(hist_indices)) + hist_indices

        pos_idx = self.item2idx.get(pos_iid, 0)
        hist_t = torch.tensor(padded, dtype=torch.long)
        return {
            "history": hist_t,
            "history_mask": hist_t != 0,
            "positive": torch.tensor(pos_idx, dtype=torch.long),
            "item_freq": torch.tensor(
                float(self._item_freq.get(pos_idx, 1)), dtype=torch.float32
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
        t0 = time.time()
        n_cutoffs = len(cutoff_timestamps)
        print(f"Building dataset index from {n_cutoffs} cutoff days...", flush=True)
        self.events = events
        self.config = config
        self.cutoffs = cutoff_timestamps
        self.item2idx = item2idx
        self.item_freq = item_freq
        self.history_len = history_len
        print(
            f"Dataset index ready: {n_cutoffs} cutoff days (lazy) in "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )

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
