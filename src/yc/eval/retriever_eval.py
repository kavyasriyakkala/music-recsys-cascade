"""Evaluation utilities for the two-tower retriever."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import polars as pl
import torch

from yc.data.dataset import _build_user_event_index
from yc.models.two_tower import TwoTowerRetriever

if TYPE_CHECKING:
    from yc.splits.builder import Snapshot


def _encode_users_batched(
    model: TwoTowerRetriever,
    snapshots: list[Snapshot],
    events: pl.DataFrame,
    item2idx: dict[int, int],
    history_len: int,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Return [N, D] user embeddings for all snapshots."""
    user_event_index = _build_user_event_index(events)
    histories: list[list[int]] = []

    for snap in snapshots:
        user_evts = user_event_index.get(snap.uid, [])
        hist_items = [iid for ts, iid in user_evts if ts < snap.cutoff][-history_len:]
        hist_indices = [item2idx.get(iid, 0) for iid in hist_items]
        padded = [0] * (history_len - len(hist_indices)) + hist_indices
        histories.append(padded)

    all_user_embs: list[torch.Tensor] = []
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for start in range(0, len(histories), batch_size):
            batch_hist = torch.tensor(
                histories[start : start + batch_size], dtype=torch.long, device=device
            )
            mask = batch_hist != 0
            emb = model.get_user_embedding(batch_hist, mask)
            all_user_embs.append(emb.cpu())

    if was_training:
        model.train()

    return torch.cat(all_user_embs, dim=0)


def evaluate_retriever(
    model: TwoTowerRetriever,
    snapshots: list[Snapshot],
    item_index: torch.Tensor,
    idx2item: dict[int, int],
    events: pl.DataFrame,
    item2idx: dict[int, int],
    history_len: int = 100,
    ks: list[int] | None = None,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict:
    """Exact matmul retrieval evaluation.

    Computes Recall@K (all positives) and novel-only Recall@K for each k in ks.
    item_index[i] corresponds to item_idx i+1.
    """
    if ks is None:
        ks = [100, 500, 1000]

    item_index_dev = item_index.to(device)
    user_embs = _encode_users_batched(
        model, snapshots, events, item2idx, history_len, device, batch_size
    )

    recall_sums: dict[int, float] = {k: 0.0 for k in ks}
    novel_sums: dict[int, float] = {k: 0.0 for k in ks}
    n_users = len(snapshots)

    for i, snap in enumerate(snapshots):
        u_emb = user_embs[i].unsqueeze(0).to(device)
        scores = (u_emb @ item_index_dev.T).squeeze(0)  # [n_items]

        pos_idxs = {item2idx[iid] for iid in snap.positives if iid in item2idx}
        novel_idxs = {item2idx[iid] for iid in snap.novel_positives if iid in item2idx}

        if not pos_idxs:
            n_users -= 1
            continue

        max_k = max(ks)
        top_idxs = scores.topk(min(max_k, scores.size(0))).indices + 1  # item_idx

        top_set = set(top_idxs.tolist())
        for k in ks:
            top_k = set(scores.topk(min(k, scores.size(0))).indices.tolist())
            top_k_item_idxs = {t + 1 for t in top_k}
            recall_sums[k] += len(pos_idxs & top_k_item_idxs) / len(pos_idxs)
            if novel_idxs:
                novel_sums[k] += len(novel_idxs & top_k_item_idxs) / len(novel_idxs)

    result: dict = {"n_users": n_users}
    for k in ks:
        result[f"recall@{k}"] = round(recall_sums[k] / max(n_users, 1), 6)
        result[f"novel_recall@{k}"] = round(novel_sums[k] / max(n_users, 1), 6)

    return result


def compare_ann_vs_exact(
    model: TwoTowerRetriever,
    snapshots: list[Snapshot],
    item_index: torch.Tensor,
    idx2item: dict[int, int],
    events: pl.DataFrame,
    item2idx: dict[int, int],
    history_len: int = 100,
    n_lists: int = 100,
    n_probe: int = 10,
    k: int = 100,
    device: str = "cpu",
) -> dict:
    """Compare faiss-cpu IVFFlat ANN retrieval vs exact matmul.

    Reports recall@k (exact), recall@k (ANN), ann_recall_of_exact,
    and latency_ms_per_user for both methods.
    """
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required for ANN comparison: pip install faiss-cpu"
        ) from exc

    item_np = item_index.numpy().astype("float32")
    d = item_np.shape[1]

    # Build IVFFlat index with inner product metric (embeddings are L2-normalized).
    quantizer = faiss.IndexFlatIP(d)
    ann_index = faiss.IndexIVFFlat(quantizer, d, min(n_lists, len(item_np)), faiss.METRIC_INNER_PRODUCT)
    ann_index.train(item_np)
    ann_index.add(item_np)
    ann_index.nprobe = n_probe

    user_embs = _encode_users_batched(
        model, snapshots, events, item2idx, history_len, device, batch_size=256
    )
    user_np = user_embs.numpy().astype("float32")

    # Exact retrieval timing.
    t0 = time.perf_counter()
    item_index_dev = item_index.to(device)
    exact_hits: list[set[int]] = []
    for i in range(len(snapshots)):
        u = user_embs[i].unsqueeze(0).to(device)
        scores = (u @ item_index_dev.T).squeeze(0)
        top_k = set((scores.topk(min(k, scores.size(0))).indices + 1).tolist())
        exact_hits.append(top_k)
    exact_ms = (time.perf_counter() - t0) * 1000 / len(snapshots)

    # ANN retrieval timing.
    t0 = time.perf_counter()
    _, ann_raw = ann_index.search(user_np, k)
    ann_ms = (time.perf_counter() - t0) * 1000 / len(snapshots)

    # Compute metrics.
    recall_exact = 0.0
    recall_ann = 0.0
    ann_recall_of_exact = 0.0
    n_valid = 0

    for i, snap in enumerate(snapshots):
        pos_idxs = {item2idx[iid] for iid in snap.positives if iid in item2idx}
        if not pos_idxs:
            continue
        ann_hits = {int(x) + 1 for x in ann_raw[i] if x >= 0}
        recall_exact += len(pos_idxs & exact_hits[i]) / len(pos_idxs)
        recall_ann += len(pos_idxs & ann_hits) / len(pos_idxs)
        if exact_hits[i]:
            ann_recall_of_exact += len(exact_hits[i] & ann_hits) / len(exact_hits[i])
        n_valid += 1

    n = max(n_valid, 1)
    return {
        "k": k,
        "recall_exact": round(recall_exact / n, 6),
        "recall_ann": round(recall_ann / n, 6),
        "ann_recall_of_exact": round(ann_recall_of_exact / n, 6),
        "exact_latency_ms_per_user": round(exact_ms, 3),
        "ann_latency_ms_per_user": round(ann_ms, 3),
        "n_users": n_valid,
    }
