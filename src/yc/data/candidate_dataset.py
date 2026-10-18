"""CandidateDataset: (user, candidate_item, label) triples for ranker training.

Features are computed at init from the pre-generated candidates parquet and
the events DataFrame; nothing is fetched lazily during __getitem__.

L1Dataset: returns 5 scalar features only (cheap features).
L2Dataset: returns 5 scalars + history_ids/mask + candidate_idx (for L2Ranker).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from yc.splits.builder import DAY_UNITS, Snapshot

# Number of scalar features shared by L1 and L2
N_SCALAR_FEATURES: int = 5
# Feature order: retrieval_score, log_popularity, user_prior_listens,
#                days_since_last_listen, is_novel


class CandidateDataset(Dataset):
    """Base training dataset for L1 and L2 rankers.

    Each sample is a (user, candidate_item, label) triple with 5 scalar features.
    Positives come from snapshot.positives; negatives are retrieved candidates
    that are NOT positives, sampled at rate neg_per_pos:1.

    Skips snapshots with no positives in the candidate set.
    """

    def __init__(
        self,
        snapshots: list[Snapshot],
        candidates_df: pl.DataFrame,
        events: pl.DataFrame,
        neg_per_pos: int = 4,
        seed: int = 42,
    ) -> None:
        # --- Global item popularity (listen count across all events) ---
        self._global_pop: dict[int, int] = {
            int(r[0]): int(r[1])
            for r in events.group_by("item_id")
            .agg(pl.len().alias("c"))
            .iter_rows()
        }

        # --- User event index: {uid: [(ts, item_id), ...]} sorted by ts ---
        # Events are pre-sorted by (uid, timestamp); we preserve that order.
        self._user_events: dict[int, list[tuple[int, int]]] = {}
        for uid, ts, iid in events.select(["uid", "timestamp", "item_id"]).iter_rows():
            uid, ts, iid = int(uid), int(ts), int(iid)
            self._user_events.setdefault(uid, []).append((ts, iid))

        # --- Candidate lookup: {(uid, cutoff): [(item_id, retrieval_score)]} ---
        cand_lookup: dict[tuple[int, int], list[tuple[int, float]]] = {}
        for row in candidates_df.select(
            ["uid", "cutoff", "item_id", "retrieval_score"]
        ).iter_rows(named=True):
            key = (int(row["uid"]), int(row["cutoff"]))
            cand_lookup.setdefault(key, []).append(
                (int(row["item_id"]), float(row["retrieval_score"]))
            )

        rng = np.random.default_rng(seed)

        # Each sample: {"uid", "cutoff", "item_id", "features": Tensor[5], "label": Tensor}
        self._samples: list[dict] = []

        for snap in snapshots:
            key = (snap.uid, snap.cutoff)
            if key not in cand_lookup:
                continue

            all_cands = cand_lookup[key]
            pos_set = set(snap.positives)
            positives = [(iid, sc) for iid, sc in all_cands if iid in pos_set]
            negatives = [(iid, sc) for iid, sc in all_cands if iid not in pos_set]

            if not positives or not negatives:
                continue

            n_neg = min(neg_per_pos * len(positives), len(negatives))
            neg_indices = rng.choice(len(negatives), n_neg, replace=False)
            sampled_neg = [negatives[int(i)] for i in neg_indices]

            # Per-item stats for this user before cutoff
            user_evs = self._user_events.get(snap.uid, [])
            item_counts: dict[int, int] = {}
            item_last_ts: dict[int, int] = {}
            for ts, iid in user_evs:
                if ts >= snap.cutoff:
                    break  # sorted by ts; stop once we pass cutoff
                item_counts[iid] = item_counts.get(iid, 0) + 1
                if ts > item_last_ts.get(iid, -1):
                    item_last_ts[iid] = ts

            for iid, sc in positives:
                self._samples.append(
                    self._make_sample(iid, sc, 1.0, item_counts, item_last_ts, snap)
                )
            for iid, sc in sampled_neg:
                self._samples.append(
                    self._make_sample(iid, sc, 0.0, item_counts, item_last_ts, snap)
                )

    def _make_sample(
        self,
        iid: int,
        sc: float,
        label: float,
        item_counts: dict[int, int],
        item_last_ts: dict[int, int],
        snap: Snapshot,
    ) -> dict:
        counts = item_counts.get(iid, 0)
        last_ts = item_last_ts.get(iid)
        dsll = (
            float(min((snap.cutoff - last_ts) / DAY_UNITS, 365.0))
            if last_ts is not None
            else 365.0
        )
        feats = torch.tensor(
            [
                sc,
                math.log1p(self._global_pop.get(iid, 0)),
                float(counts),
                dsll,
                float(counts == 0),  # is_novel
            ],
            dtype=torch.float32,
        )
        return {
            "uid": snap.uid,
            "cutoff": snap.cutoff,
            "item_id": iid,
            "features": feats,
            "label": torch.tensor(label, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self._samples[idx]
        return {"features": s["features"], "label": s["label"]}


class L1Dataset(CandidateDataset):
    """Returns 5 scalar features (identical to base CandidateDataset)."""
    pass


class L2Dataset(CandidateDataset):
    """Returns 5 scalar features + history tensors + candidate_idx.

    History is the sequence of item2idx indices for the user's prior events
    (strictly before cutoff), capped to history_len and left-padded with zeros.
    """

    def __init__(
        self,
        snapshots: list[Snapshot],
        candidates_df: pl.DataFrame,
        events: pl.DataFrame,
        item2idx: dict[int, int],
        neg_per_pos: int = 4,
        history_len: int = 100,
        seed: int = 42,
    ) -> None:
        super().__init__(snapshots, candidates_df, events, neg_per_pos, seed)
        self._item2idx = item2idx
        self._history_len = history_len

        # Build shared history index once per unique (uid, cutoff) pair.
        # Avoids storing full history per sample.
        self._history_index: dict[tuple[int, int], list[int]] = {}
        pairs = {(s["uid"], s["cutoff"]) for s in self._samples}
        for uid, cutoff in pairs:
            user_evs = self._user_events.get(uid, [])
            indices = [
                item2idx[iid]
                for ts, iid in user_evs
                if ts < cutoff and iid in item2idx
            ]
            self._history_index[(uid, cutoff)] = indices[-history_len:]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self._samples[idx]
        history = self._history_index.get((s["uid"], s["cutoff"]), [])
        H = self._history_len
        n_real = len(history)
        padded = [0] * (H - n_real) + history  # left-pad with 0
        mask = [False] * (H - n_real) + [True] * n_real

        candidate_idx = self._item2idx.get(s["item_id"], 0)

        return {
            "features": s["features"],
            "label": s["label"],
            "history_ids": torch.tensor(padded, dtype=torch.long),
            "history_mask": torch.tensor(mask, dtype=torch.bool),
            "candidate_idx": torch.tensor(candidate_idx, dtype=torch.long),
        }
