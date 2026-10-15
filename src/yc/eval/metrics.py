"""Recall@K and NDCG@K for the Yambda-cascade evaluation harness.

Each metric returns a per-user array so paired bootstrap CIs can be added later.
A brute-force reference implementation is provided for test verification only —
it is maximally legible, not fast.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from yc.splits.builder import Snapshot


# ---------------------------------------------------------------------------
# Brute-force reference (used only in tests)
# ---------------------------------------------------------------------------

def _brute_recall_at_k(rec: list[int], rel: set[int], k: int) -> float:
    if not rel:
        return 0.0
    hits = sum(1 for iid in rec[:k] if iid in rel)
    return hits / len(rel)


def _brute_ndcg_at_k(rec: list[int], rel: set[int], k: int) -> float:
    if not rel:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, iid in enumerate(rec[:k])
        if iid in rel
    )
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Vectorised implementations
# ---------------------------------------------------------------------------

def recall_at_k(
    recommended: list[list[int]],
    relevant: list[list[int]],
    k: int,
) -> dict[str, Any]:
    """Compute Recall@K.

    Returns {"mean": float, "per_user": np.ndarray}.
    Users with no relevant items are excluded from the mean.
    """
    per_user: list[float] = []
    for rec, rel in zip(recommended, relevant):
        rel_set = set(rel)
        if not rel_set:
            continue
        hits = sum(1 for iid in rec[:k] if iid in rel_set)
        per_user.append(hits / len(rel_set))
    arr = np.array(per_user, dtype=np.float64)
    return {"mean": float(arr.mean()) if len(arr) > 0 else 0.0, "per_user": arr}


def ndcg_at_k(
    recommended: list[list[int]],
    relevant: list[list[int]],
    k: int,
) -> dict[str, Any]:
    """Compute NDCG@K.

    Returns {"mean": float, "per_user": np.ndarray}.
    """
    per_user: list[float] = []
    for rec, rel in zip(recommended, relevant):
        rel_set = set(rel)
        if not rel_set:
            continue
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, iid in enumerate(rec[:k])
            if iid in rel_set
        )
        ideal_hits = min(len(rel_set), k)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        per_user.append(dcg / idcg if idcg > 0 else 0.0)
    arr = np.array(per_user, dtype=np.float64)
    return {"mean": float(arr.mean()) if len(arr) > 0 else 0.0, "per_user": arr}


# ---------------------------------------------------------------------------
# End-to-end evaluator
# ---------------------------------------------------------------------------

def evaluate(
    snapshots: list[Snapshot],
    recommendations: dict[int, list[int]],
    ks: list[int] = None,
) -> dict[str, Any]:
    """Evaluate recommendations against snapshots.

    Returns metrics broken down by all-target and novel-only, at each K.
    Users with no positives are skipped (they should not be in snapshots,
    but we guard anyway).
    """
    if ks is None:
        ks = [10, 50, 100]

    rec_lists: list[list[int]] = []
    all_positives: list[list[int]] = []
    novel_positives: list[list[int]] = []

    for snap in snapshots:
        if not snap.positives:
            continue
        rec = recommendations.get(snap.uid, [])
        rec_lists.append(rec)
        all_positives.append(snap.positives)
        novel_positives.append(snap.novel_positives)

    results: dict[str, Any] = {}
    for k in ks:
        results[f"recall@{k}"] = recall_at_k(rec_lists, all_positives, k)
        results[f"ndcg@{k}"] = ndcg_at_k(rec_lists, all_positives, k)
        results[f"novel_recall@{k}"] = recall_at_k(rec_lists, novel_positives, k)
        results[f"novel_ndcg@{k}"] = ndcg_at_k(rec_lists, novel_positives, k)

    return results
