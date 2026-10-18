"""Train L1 and L2 rankers.

Usage (real data):
    python scripts/train_rankers.py \\
        --config configs/ranker.yaml \\
        --retriever-checkpoint /scratch/ky2751/outputs/retriever/retriever_best.pt \\
        --candidates-dir /scratch/ky2751/outputs/candidates \\
        --splits-cache /scratch/ky2751/outputs/retriever/splits.pkl \\
        --data-dir /scratch/ky2751/yambda/50m \\
        --output-dir /scratch/ky2751/outputs/rankers

Usage (smoke):
    python scripts/train_rankers.py \\
        --config configs/ranker.yaml \\
        --retriever-checkpoint none \\
        --candidates-dir none \\
        --splits-cache none \\
        --data-dir none \\
        --output-dir /tmp/ranker_smoke \\
        --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_config(config_path: str) -> dict:
    import yaml  # type: ignore
    with open(config_path) as f:
        return yaml.safe_load(f)


def _run_smoke(config: dict, output_dir: Path) -> None:
    """Smoke: synthetic data, random retriever, CPU, <3 minutes."""
    import numpy as np
    from yc.data.synthetic import generate_dataset
    from yc.data.candidate_dataset import L1Dataset, L2Dataset
    from yc.data.dataset import build_item2idx
    from yc.models.rankers import L1Ranker, L2Ranker
    from yc.models.two_tower import ItemTower, TwoTowerRetriever
    from yc.splits.builder import SplitConfig, build_all_splits, DAY_UNITS
    from yc.train.ranker_trainer import train_ranker
    from yc.eval.cascade_eval import evaluate_cascade

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=== smoke: generating synthetic data ===", flush=True)
    events_df, embeddings_df, item_artist_df = generate_dataset(
        n_users=60, n_items=200, n_days=20, embedding_dim=16, seed=seed
    )
    events_df = events_df.filter(pl.col("event_type") == "listen").sort(["uid", "timestamp"])

    cfg = SplitConfig(n_days=20, min_prior_events=3)
    splits = build_all_splits(events_df, cfg)

    rank_train_snaps = splits.get("rank_train", [])
    val_snaps = splits.get("validation", [])

    print(f"rank_train: {len(rank_train_snaps)} snaps, val: {len(val_snaps)} snaps", flush=True)

    item2idx, idx2item = build_item2idx(events_df)
    n_items = len(item2idx)
    embed_dim = 16

    # Generate random candidates (no trained retriever in smoke)
    from generate_candidates import _build_smoke_data as _gen  # noqa: F401
    candidates_dir = output_dir / "candidates"
    print("Generating smoke candidates...", flush=True)
    _gen(candidates_dir, top_k=50, seed=seed)

    cands_train_path = str(candidates_dir / "candidates_rank_train.parquet")
    cands_val_path = str(candidates_dir / "candidates_validation.parquet")

    cands_train_df = pl.read_parquet(cands_train_path)
    cands_val_df = pl.read_parquet(cands_val_path)

    # Create random item tower
    item_tower = ItemTower(n_items=n_items, embed_dim=embed_dim, audio_embeddings={})

    # --- L1 ---
    print("\n=== Training L1 ===", flush=True)
    l1_cfg = dict(config.get("l1", {}))
    l1_cfg.update({"n_epochs": 2, "batch_size": 64, "eval_every_n_steps": 10})

    l1_ds = L1Dataset(
        snapshots=rank_train_snaps,
        candidates_df=cands_train_df,
        events=events_df,
        neg_per_pos=l1_cfg.get("neg_per_pos", 4),
        seed=seed,
    )
    print(f"L1Dataset: {len(l1_ds)} samples", flush=True)
    if len(l1_ds) == 0:
        print("WARNING: L1Dataset is empty — no snapshots with positives in candidates", flush=True)
    else:
        l1_model = L1Ranker(hidden_dim=32)
        train_ranker(
            ranker=l1_model,
            train_dataset=l1_ds,
            val_snapshots=val_snaps,
            val_candidates_path=cands_val_path,
            config=l1_cfg,
            output_dir=str(output_dir / "l1"),
            ranker_name="l1",
        )
        print("L1 training complete.", flush=True)

    # --- L2 ---
    print("\n=== Training L2 ===", flush=True)
    l2_cfg = dict(config.get("l2", {}))
    l2_cfg.update({"n_epochs": 2, "batch_size": 32, "eval_every_n_steps": 10})

    history_len = l2_cfg.get("history_len", 10)
    l2_ds = L2Dataset(
        snapshots=rank_train_snaps,
        candidates_df=cands_train_df,
        events=events_df,
        item2idx=item2idx,
        neg_per_pos=l2_cfg.get("neg_per_pos", 4),
        history_len=history_len,
        seed=seed,
    )
    print(f"L2Dataset: {len(l2_ds)} samples", flush=True)
    if len(l2_ds) == 0:
        print("WARNING: L2Dataset is empty — no snapshots with positives in candidates", flush=True)
    else:
        l2_model = L2Ranker(item_tower=item_tower, n_scalar=5, embed_dim=embed_dim)
        train_ranker(
            ranker=l2_model,
            train_dataset=l2_ds,
            val_snapshots=val_snaps,
            val_candidates_path=cands_val_path,
            config=l2_cfg,
            output_dir=str(output_dir / "l2"),
            ranker_name="l2",
        )
        print("L2 training complete.", flush=True)

    print("\n=== smoke-rankers complete ===", flush=True)


def _run_real(config: dict, args: argparse.Namespace, output_dir: Path) -> None:
    import pickle
    from yc.data.real_loader import load_real_events, load_artist_mapping
    from yc.data.candidate_dataset import L1Dataset, L2Dataset
    from yc.data.dataset import build_item2idx
    from yc.models.rankers import L1Ranker, L2Ranker
    from yc.models.two_tower import ItemTower
    from yc.train.ranker_trainer import train_ranker

    print("Loading events...", flush=True)
    events = load_real_events(args.data_dir)

    with open(args.splits_cache, "rb") as f:
        splits = pickle.load(f)

    rank_train_snaps = splits.get("rank_train", [])
    val_snaps = splits.get("validation", [])

    item2idx, idx2item = build_item2idx(events)

    cands_train_df = pl.read_parquet(str(Path(args.candidates_dir) / "candidates_rank_train.parquet"))
    cands_val_path = str(Path(args.candidates_dir) / "candidates_validation.parquet")

    ckpt = torch.load(args.retriever_checkpoint, map_location="cpu")
    n_items = ckpt["n_items"]
    embed_dim = ckpt["embed_dim"]

    item_tower = ItemTower(n_items=n_items, embed_dim=embed_dim, audio_embeddings={})
    from collections import OrderedDict
    tower_state = OrderedDict(
        {k.replace("item_tower.", ""): v for k, v in ckpt["model_state_dict"].items()
         if k.startswith("item_tower.")}
    )
    item_tower.load_state_dict(tower_state, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    item_tower = item_tower.to(device)

    l1_cfg = config.get("l1", {})
    l2_cfg = config.get("l2", {})

    print("Training L1...", flush=True)
    l1_ds = L1Dataset(
        snapshots=rank_train_snaps,
        candidates_df=cands_train_df,
        events=events,
        neg_per_pos=l1_cfg.get("neg_per_pos", 4),
        seed=config.get("seed", 42),
    )
    train_ranker(
        ranker=L1Ranker(),
        train_dataset=l1_ds,
        val_snapshots=val_snaps,
        val_candidates_path=cands_val_path,
        config=l1_cfg,
        output_dir=str(output_dir / "l1"),
        ranker_name="l1",
    )

    print("Training L2...", flush=True)
    history_len = l2_cfg.get("history_len", 100)
    l2_ds = L2Dataset(
        snapshots=rank_train_snaps,
        candidates_df=cands_train_df,
        events=events,
        item2idx=item2idx,
        neg_per_pos=l2_cfg.get("neg_per_pos", 4),
        history_len=history_len,
        seed=config.get("seed", 42),
    )
    train_ranker(
        ranker=L2Ranker(item_tower=item_tower, n_scalar=5, embed_dim=embed_dim),
        train_dataset=l2_ds,
        val_snapshots=val_snaps,
        val_candidates_path=cands_val_path,
        config=l2_cfg,
        output_dir=str(output_dir / "l2"),
        ranker_name="l2",
    )
    print("Done.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--retriever-checkpoint", required=True)
    parser.add_argument("--candidates-dir", required=True)
    parser.add_argument("--splits-cache", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        _run_smoke(config, out_dir)
    else:
        _run_real(config, args, out_dir)


if __name__ == "__main__":
    main()
