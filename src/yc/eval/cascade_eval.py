"""End-to-end cascade evaluation: retriever → L1 → L2."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from yc.eval.metrics import ndcg_at_k, recall_at_k
from yc.splits.builder import DAY_UNITS, Snapshot


def evaluate_cascade(
    retriever: Any,   # TwoTowerRetriever
    l1: nn.Module,
    l2: nn.Module,
    snapshots: list[Snapshot],
    events: pl.DataFrame,
    item_index: torch.Tensor,
    idx2item: dict[int, int],
    device: str = "cuda",
    history_len: int = 500,
    top_k_retrieval: int = 1000,
    l1_top_k: int = 100,
    l2_top_k: int = 10,
) -> dict[str, Any]:
    """Run retriever → L1 → L2 for each snapshot and report metrics + latency.

    Returns per-stage recall ceilings, HitRate@10, Recall@10, NDCG@10,
    and per-stage latency (p50/p95 ms/user).
    """
    from yc.data.dataset import build_item2idx  # item2idx from events

    item_ids = sorted(events["item_id"].unique().to_list())
    item2idx = {iid: i + 1 for i, iid in enumerate(item_ids)}

    retriever.eval()
    l1.eval()
    l2.eval()

    lat_ret: list[float] = []
    lat_l1: list[float] = []
    lat_l2: list[float] = []

    ret_recs: list[list[int]] = []
    l1_recs: list[list[int]] = []
    l2_recs: list[list[int]] = []
    all_pos: list[list[int]] = []
    all_novel: list[list[int]] = []

    global_pop: dict[int, int] = {
        int(r[0]): int(r[1])
        for r in events.group_by("item_id").agg(pl.len().alias("c")).iter_rows()
    }

    with torch.no_grad():
        for snap in snapshots:
            if not snap.positives:
                continue

            # --- Retrieval ---
            t0 = time.perf_counter()
            user_hist = (
                events.filter(
                    (pl.col("uid") == snap.uid) & (pl.col("timestamp") < snap.cutoff)
                )
                .tail(history_len)
                .select(["item_id"])
            )
            iids = user_hist["item_id"].to_list()
            indices = [item2idx[iid] for iid in iids if iid in item2idx]
            if not indices:
                continue

            n_real = len(indices)
            padded = [0] * (history_len - n_real) + indices[-history_len:]
            hist_t = torch.tensor(padded, dtype=torch.long, device=device).unsqueeze(0)
            mask_t = torch.tensor(
                [False] * (history_len - n_real) + [True] * min(n_real, history_len),
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)

            user_emb = retriever.user_tower(hist_t, mask_t)  # [1, D]
            sims = (user_emb @ item_index.to(device).T).squeeze(0)  # [n_items]
            top_k_r = min(top_k_retrieval, sims.shape[0])
            top_scores, top_idx = torch.topk(sims, top_k_r)
            # item_index[i] corresponds to item_idx i+1
            ret_candidates = [
                idx2item[int(i) + 1] for i in top_idx.tolist() if (int(i) + 1) in idx2item
            ]
            ret_scores_map = {c: float(s) for c, s in zip(ret_candidates, top_scores.tolist())}
            lat_ret.append((time.perf_counter() - t0) * 1000)

            # --- L1 scoring ---
            t1 = time.perf_counter()
            l1_feats = _build_scalar_feats(ret_candidates, ret_scores_map, global_pop, item2idx)
            l1_feats_t = torch.tensor(l1_feats, dtype=torch.float32, device=device)
            l1_scores = l1(l1_feats_t).squeeze(-1)
            l1_order = torch.argsort(l1_scores, descending=True)
            l1_top = l1_order[:l1_top_k]
            l1_candidates = [ret_candidates[i] for i in l1_top.tolist()]
            lat_l1.append((time.perf_counter() - t1) * 1000)

            # --- L2 scoring ---
            t2 = time.perf_counter()
            l1_ret_scores = {c: ret_scores_map[c] for c in l1_candidates}
            l2_feats = _build_scalar_feats(l1_candidates, l1_ret_scores, global_pop, item2idx)
            l2_feats_t = torch.tensor(l2_feats, dtype=torch.float32, device=device)

            l2_hist_ids = torch.tensor(padded, dtype=torch.long, device=device).unsqueeze(0)
            l2_hist_mask = mask_t

            # L2 forward (one candidate at a time to avoid memory pressure)
            l2_scores_list: list[float] = []
            for fi in range(l2_feats_t.shape[0]):
                cand_id = l1_candidates[fi]
                cand_idx = item2idx.get(cand_id, 0)
                cand_idx_t = torch.tensor([cand_idx], dtype=torch.long, device=device)
                s = l2(
                    l2_feats_t[fi].unsqueeze(0),
                    cand_idx_t,
                    l2_hist_ids,
                    l2_hist_mask,
                )
                l2_scores_list.append(float(s.item()))
            l2_order = sorted(range(len(l2_scores_list)), key=lambda i: -l2_scores_list[i])
            l2_top = [l1_candidates[i] for i in l2_order[:l2_top_k]]
            lat_l2.append((time.perf_counter() - t2) * 1000)

            ret_recs.append(ret_candidates)
            l1_recs.append(l1_candidates)
            l2_recs.append(l2_top)
            all_pos.append(snap.positives)
            all_novel.append(snap.novel_positives)

    def _p(arr: list[float], p: int) -> float:
        return float(np.percentile(arr, p)) if arr else 0.0

    ret_r = recall_at_k(ret_recs, all_pos, top_k_retrieval)
    l1_r = recall_at_k(l1_recs, all_pos, l1_top_k)
    l2_r = recall_at_k(l2_recs, all_pos, l2_top_k)
    l2_n = ndcg_at_k(l2_recs, all_pos, l2_top_k)
    l2_hr = recall_at_k([[r[0]] if r else [] for r in l2_recs], all_pos, 1)  # HR@1

    return {
        f"retriever_recall@{top_k_retrieval}": ret_r["mean"],
        f"l1_recall@{l1_top_k}": l1_r["mean"],
        f"l2_recall@{l2_top_k}": l2_r["mean"],
        f"l2_ndcg@{l2_top_k}": l2_n["mean"],
        "n_users": len(l2_recs),
        "latency_retrieval_p50_ms": _p(lat_ret, 50),
        "latency_retrieval_p95_ms": _p(lat_ret, 95),
        "latency_l1_p50_ms": _p(lat_l1, 50),
        "latency_l1_p95_ms": _p(lat_l1, 95),
        "latency_l2_p50_ms": _p(lat_l2, 50),
        "latency_l2_p95_ms": _p(lat_l2, 95),
    }


def _build_scalar_feats(
    candidates: list[int],
    ret_scores: dict[int, float],
    global_pop: dict[int, int],
    item2idx: dict[int, int],
) -> list[list[float]]:
    """Build [retrieval_score, log_pop, prior_listens=0, dsll=365, is_novel=1] per candidate.

    This is a lightweight approximation used inside evaluate_cascade (no per-user
    history computation). For full feature computation use CandidateDataset.
    """
    rows = []
    for iid in candidates:
        rows.append([
            ret_scores.get(iid, 0.0),
            math.log1p(global_pop.get(iid, 0)),
            0.0,  # prior_listens (unknown in cascade eval)
            365.0,  # days_since_last_listen
            1.0,  # is_novel
        ])
    return rows
