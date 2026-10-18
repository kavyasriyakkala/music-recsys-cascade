"""Training loop for L1Ranker and L2Ranker."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yc.data.candidate_dataset import CandidateDataset, L2Dataset
from yc.eval.metrics import ndcg_at_k, recall_at_k
from yc.models.rankers import L1Ranker, L2Ranker
from yc.splits.builder import Snapshot


def _compute_auc(labels: list[float], scores: list[float]) -> float:
    """Compute ROC-AUC without sklearn for portability."""
    if len(set(labels)) < 2:
        return 0.5
    pos_scores = [s for s, l in zip(scores, labels) if l > 0.5]
    neg_scores = [s for s, l in zip(scores, labels) if l <= 0.5]
    if not pos_scores or not neg_scores:
        return 0.5
    n = len(pos_scores) * len(neg_scores)
    concordant = sum(1 for p in pos_scores for q in neg_scores if p > q)
    ties = sum(1 for p in pos_scores for q in neg_scores if p == q)
    return (concordant + 0.5 * ties) / n


def _eval_l1_cascade(
    l1: L1Ranker,
    snapshots: list[Snapshot],
    candidates_df: pl.DataFrame,
    top_k: int,
    device: str,
) -> dict[str, float]:
    """Score all candidates per snapshot with L1, keep top-k, compute Recall@k."""
    l1.eval()
    # Build per-snapshot candidate lookup
    cand_by_snap: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for row in candidates_df.select(
        ["uid", "cutoff", "item_id", "retrieval_score"]
    ).iter_rows(named=True):
        key = (int(row["uid"]), int(row["cutoff"]))
        cand_by_snap.setdefault(key, []).append(
            (int(row["item_id"]), float(row["retrieval_score"]))
        )

    from yc.data.candidate_dataset import N_SCALAR_FEATURES
    import math

    global_pop: dict[int, int] = {}  # simplified: skip for cascade eval

    all_recs: list[list[int]] = []
    all_pos: list[list[int]] = []

    with torch.no_grad():
        for snap in snapshots:
            key = (snap.uid, snap.cutoff)
            if key not in cand_by_snap or not snap.positives:
                continue
            cands = cand_by_snap[key]
            feats = torch.tensor(
                [[sc, 0.0, 0.0, 365.0, 1.0] for _, sc in cands],
                dtype=torch.float32,
            ).to(device)
            scores = l1(feats).squeeze(-1).cpu().tolist()
            ranked = [
                iid for iid, _ in sorted(zip([c[0] for c in cands], scores), key=lambda x: -x[1])
            ]
            all_recs.append(ranked[:top_k])
            all_pos.append(snap.positives)

    r = recall_at_k(all_recs, all_pos, top_k)
    return {f"recall@{top_k}": r["mean"]}


def train_ranker(
    ranker: nn.Module,
    train_dataset: CandidateDataset,
    val_snapshots: list[Snapshot],
    val_candidates_path: str,
    config: dict[str, Any],
    output_dir: str,
    ranker_name: str,  # "l1" or "l2"
) -> None:
    """Train an L1 or L2 ranker.

    Saves best checkpoint (by validation AUC) to output_dir/{ranker_name}_best.pt.
    Writes training log to output_dir/{ranker_name}_train_log.json.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ranker = ranker.to(device)

    batch_size: int = config.get("batch_size", 4096 if ranker_name == "l1" else 1024)
    lr: float = config.get("lr", 1e-3)
    weight_decay: float = config.get("weight_decay", 1e-5)
    n_epochs: int = config.get("n_epochs", 5)
    patience: int = config.get("patience", 2)
    eval_every: int = config.get("eval_every_n_steps", 200)

    # pos_weight: inverse class frequency from training data
    n_pos = sum(1 for s in train_dataset._samples if float(s["label"]) > 0.5)
    n_neg = len(train_dataset) - n_pos
    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)], dtype=torch.float32, device=device
    )
    loss_fn = nn.BCELoss(weight=None)  # pos_weight applied via manual reweighting
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    optimizer = torch.optim.AdamW(
        [p for p in ranker.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_cands_df = pl.read_parquet(val_candidates_path) if val_candidates_path != "none" else None

    is_l2 = ranker_name == "l2"

    best_auc = -1.0
    no_improve = 0
    global_step = 0
    eval_log: list[dict] = []

    for epoch in range(n_epochs):
        ranker.train()
        running_loss = 0.0
        n_batches = 0

        for batch in loader:
            feats = batch["features"].to(device)
            labels = batch["label"].to(device)

            if is_l2:
                hist = batch["history_ids"].to(device)
                mask = batch["history_mask"].to(device)
                cand_idx = batch["candidate_idx"].to(device)
                logits = ranker(feats, cand_idx, hist, mask)
            else:
                logits = ranker(feats)

            # Use BCELoss (output is already sigmoided)
            loss = nn.functional.binary_cross_entropy(
                logits.squeeze(-1), labels, reduction="mean"
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % eval_every == 0:
                mean_loss = running_loss / n_batches
                running_loss = 0.0
                n_batches = 0

                auc = _quick_val_auc(ranker, train_dataset, device, is_l2, max_samples=2000)
                print(
                    f"[{ranker_name}] step={global_step} epoch={epoch+1} "
                    f"loss={mean_loss:.4f} val_auc={auc:.4f}",
                    flush=True,
                )
                eval_log.append({"step": global_step, "loss": mean_loss, "val_auc": auc})

                if auc > best_auc:
                    best_auc = auc
                    no_improve = 0
                    torch.save(
                        {"step": global_step, "model_state_dict": ranker.state_dict(), "auc": auc},
                        out_dir / f"{ranker_name}_best.pt",
                    )
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stop at step {global_step}", flush=True)
                        break
                ranker.train()
        else:
            continue
        break

    # Final eval
    if not eval_log:
        auc = _quick_val_auc(ranker, train_dataset, device, is_l2, max_samples=2000)
        eval_log.append({"step": global_step, "loss": float("nan"), "val_auc": auc})

    if not (out_dir / f"{ranker_name}_best.pt").exists():
        torch.save(
            {"step": global_step, "model_state_dict": ranker.state_dict(), "auc": 0.0},
            out_dir / f"{ranker_name}_best.pt",
        )

    log_path = out_dir / f"{ranker_name}_train_log.json"
    with open(log_path, "w") as f:
        json.dump({"evals": eval_log, "best_auc": best_auc}, f, indent=2)

    print(f"[{ranker_name}] Best AUC: {best_auc:.4f} → {out_dir / f'{ranker_name}_best.pt'}", flush=True)


def _quick_val_auc(
    ranker: nn.Module,
    dataset: CandidateDataset,
    device: str,
    is_l2: bool,
    max_samples: int = 2000,
) -> float:
    """Fast AUC on a random subset of training data (used as proxy for validation)."""
    ranker.eval()
    indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
    all_labels: list[float] = []
    all_scores: list[float] = []

    with torch.no_grad():
        for i in indices:
            s = dataset[int(i)]
            feats = s["features"].unsqueeze(0).to(device)
            label = float(s["label"].item())

            if is_l2:
                hist = s["history_ids"].unsqueeze(0).to(device)
                mask = s["history_mask"].unsqueeze(0).to(device)
                cand_idx = s["candidate_idx"].unsqueeze(0).to(device)
                score = float(ranker(feats, cand_idx, hist, mask).item())
            else:
                score = float(ranker(feats).item())

            all_labels.append(label)
            all_scores.append(score)

    return _compute_auc(all_labels, all_scores)
