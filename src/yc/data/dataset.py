"""RetrieverDataset: wraps Snapshot lists into a torch Dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from yc.splits.builder import Snapshot


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
