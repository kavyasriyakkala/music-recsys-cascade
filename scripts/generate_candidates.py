"""Generate top-K candidate sets for rank_train, validation, and test splits.

Uses a frozen retriever checkpoint to encode each user's history and retrieve
top-K items via exact inner-product search.  Saves one parquet file per split.

Usage (real data):
    python scripts/generate_candidates.py \\
        --retriever-checkpoint /scratch/ky2751/outputs/retriever/retriever_best.pt \\
        --data-dir /scratch/ky2751/yambda/50m \\
        --splits-cache /scratch/ky2751/outputs/retriever/splits.pkl \\
        --output-dir /scratch/ky2751/outputs/candidates \\
        --top-k 1000 \\
        --device cuda \\
        --batch-size 256

Usage (smoke):
    python scripts/generate_candidates.py --smoke \\
        --retriever-checkpoint none \\
        --data-dir none \\
        --splits-cache none \\
        --output-dir /tmp/candidates_smoke
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _build_smoke_data(output_dir: Path, top_k: int, seed: int) -> None:
    """Generate random candidate parquets from synthetic events."""
    from yc.data.synthetic import generate_dataset
    from yc.splits.builder import DAY_UNITS, SplitConfig, build_snapshots, build_all_splits

    events_df, embeddings_df, item_artist_df = generate_dataset(
        n_users=60, n_items=200, n_days=20, embedding_dim=16, seed=seed
    )
    events_df = events_df.filter(pl.col("event_type") == "listen").sort(["uid", "timestamp"])

    cfg = SplitConfig(n_days=20, min_prior_events=3)
    splits = build_all_splits(events_df, cfg)

    all_item_ids = sorted(events_df["item_id"].unique().to_list())
    rng = np.random.default_rng(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}

    split_map = {
        "rank_train": splits.get("rank_train", []),
        "validation": splits.get("validation", []),
        "test": splits.get("test", []),
    }

    for split_name, snaps in split_map.items():
        rows: list[dict] = []
        for snap in snaps:
            k = min(top_k, len(all_item_ids))
            chosen = rng.choice(all_item_ids, k, replace=False)
            scores = rng.uniform(0, 1, k)
            order = np.argsort(-scores)
            for rank_i, i in enumerate(order):
                rows.append({
                    "uid": snap.uid,
                    "cutoff": snap.cutoff,
                    "item_id": int(chosen[i]),
                    "retrieval_score": float(scores[i]),
                    "rank": rank_i,
                })
            pos_set = set(snap.positives)
            n_recall = sum(1 for r in rows[-k:] if r["item_id"] in pos_set)
            denom = len(pos_set)

        df = pl.DataFrame(rows) if rows else pl.DataFrame({"uid": [], "cutoff": [], "item_id": [], "retrieval_score": [], "rank": []})
        out_path = output_dir / f"candidates_{split_name}.parquet"
        df.write_parquet(out_path)
        stats[split_name] = {"n_snapshots": len(snaps), "n_rows": len(rows)}
        print(f"  {split_name}: {len(snaps)} snapshots, {len(rows)} rows → {out_path}", flush=True)

    stats_path = output_dir / "candidate_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats → {stats_path}", flush=True)


def _run_real(
    retriever_ckpt: str,
    data_dir: str,
    splits_cache: str,
    output_dir: Path,
    top_k: int,
    device: str,
    batch_size: int,
) -> None:
    from yc.data.real_loader import load_real_events
    from yc.data.dataset import build_item2idx
    from yc.models.two_tower import TwoTowerRetriever, ItemTower
    from yc.splits.builder import Snapshot
    from yc.eval.metrics import recall_at_k

    print("Loading events...", flush=True)
    events = load_real_events(data_dir)

    print("Loading retriever checkpoint...", flush=True)
    ckpt = torch.load(retriever_ckpt, map_location=device)
    n_items = ckpt["n_items"]
    embed_dim = ckpt["embed_dim"]
    history_len = ckpt.get("history_len", 500)

    item_tower = ItemTower(n_items=n_items, embed_dim=embed_dim, audio_embeddings={})
    from collections import OrderedDict
    tower_state = OrderedDict(
        {k.replace("item_tower.", ""): v for k, v in ckpt["model_state_dict"].items()
         if k.startswith("item_tower.")}
    )
    item_tower.load_state_dict(tower_state, strict=False)

    user_tower_state = OrderedDict(
        {k.replace("user_tower.", ""): v for k, v in ckpt["model_state_dict"].items()
         if k.startswith("user_tower.")}
    )

    from yc.models.two_tower import TwoTowerRetriever
    retriever = TwoTowerRetriever(n_items=n_items, embed_dim=embed_dim, audio_embeddings={})
    retriever.load_state_dict(ckpt["model_state_dict"])
    retriever = retriever.to(device)
    retriever.eval()

    item2idx, idx2item = build_item2idx(events)

    print("Building item index...", flush=True)
    item_index = retriever.build_item_index(device=device).to(device)

    with open(splits_cache, "rb") as f:
        splits = pickle.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_stats: dict[str, dict] = {}

    for split_name in ["rank_train", "validation", "test"]:
        snaps: list[Snapshot] = splits.get(split_name, [])
        if not snaps:
            print(f"  {split_name}: no snapshots, skipping", flush=True)
            continue

        print(f"  Generating {split_name} candidates ({len(snaps)} snapshots)...", flush=True)
        rows: list[dict] = []
        all_recs: list[list[int]] = []
        all_pos: list[list[int]] = []

        with torch.no_grad():
            for snap in snaps:
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
                    dtype=torch.bool, device=device,
                ).unsqueeze(0)

                user_emb = retriever.user_tower(hist_t, mask_t)
                sims = (user_emb @ item_index.T).squeeze(0)
                k = min(top_k, sims.shape[0])
                top_scores, top_idx = torch.topk(sims, k)

                candidates = [idx2item[int(i) + 1] for i in top_idx.tolist() if (int(i) + 1) in idx2item]
                for rank_i, (iid, sc) in enumerate(zip(candidates, top_scores.tolist())):
                    rows.append({
                        "uid": snap.uid,
                        "cutoff": snap.cutoff,
                        "item_id": iid,
                        "retrieval_score": float(sc),
                        "rank": rank_i,
                    })
                all_recs.append(candidates)
                all_pos.append(snap.positives)

        df = pl.DataFrame(rows)
        out_path = output_dir / f"candidates_{split_name}.parquet"
        df.write_parquet(out_path)

        r = recall_at_k(all_recs, all_pos, top_k)
        r_novel_snaps = [s for s in snaps if s.novel_positives]
        r_novel = recall_at_k(
            [all_recs[i] for i, s in enumerate(snaps) if s.novel_positives],
            [s.novel_positives for s in snaps if s.novel_positives],
            top_k,
        ) if r_novel_snaps else {"mean": 0.0}

        all_stats[split_name] = {
            "n_snapshots": len(snaps),
            "n_rows": len(rows),
            f"recall@{top_k}": r["mean"],
            f"novel_recall@{top_k}": r_novel["mean"],
        }
        print(
            f"  {split_name}: recall@{top_k}={r['mean']:.4f} "
            f"novel={r_novel['mean']:.4f} → {out_path}",
            flush=True,
        )

    stats_path = output_dir / "candidate_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Stats → {stats_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever-checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--splits-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if args.smoke:
        print("=== smoke: generating random candidates ===", flush=True)
        _build_smoke_data(out_dir, top_k=50, seed=args.seed)
    else:
        _run_real(
            retriever_ckpt=args.retriever_checkpoint,
            data_dir=args.data_dir,
            splits_cache=args.splits_cache,
            output_dir=out_dir,
            top_k=args.top_k,
            device=args.device,
            batch_size=args.batch_size,
        )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
