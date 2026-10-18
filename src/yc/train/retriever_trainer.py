"""Training loop for the two-tower retriever."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import pickle

from yc.data.dataset import LazySnapshotIterator, RetrieverDataset, build_item2idx
from yc.data.real_loader import load_embeddings, load_real_events
from yc.eval.retriever_eval import evaluate_retriever
from yc.models.two_tower import TwoTowerRetriever
from yc.splits.builder import DAY_UNITS, SplitConfig, build_snapshots
from yc.train.losses import in_batch_softmax_loss


def _load_config(config_path: str) -> dict:
    import yaml  # type: ignore

    with open(config_path) as f:
        return yaml.safe_load(f)


def _cosine_with_warmup(warmup_steps: int, total_steps: int):
    def _lr(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-6)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _lr


def train_retriever(
    config_path: str,
    data_dir: str,
    output_dir: str,
    smoke: bool = False,
) -> None:
    """Full training loop for TwoTowerRetriever.

    In smoke mode: uses synthetic data, 2 epochs, batch_size=64, eval every 50 steps.
    Must complete in <2 minutes on CPU.
    """
    cfg = _load_config(config_path)
    seed: int = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    if smoke:
        from yc.data.synthetic import generate_dataset

        events_df, embeddings_df, _ = generate_dataset(
            n_users=60,
            n_items=600,
            n_days=20,
            embedding_dim=64,
            seed=seed,
        )
        embeddings: dict[int, np.ndarray] = {
            int(row["item_id"]): np.array(row["normalized_embed"], dtype=np.float32)
            for row in embeddings_df.iter_rows(named=True)
        }
        events_df = events_df.filter(pl.col("event_type") == "listen").sort(
            ["uid", "timestamp"]
        )
        n_days = int(events_df["timestamp"].max() // DAY_UNITS) + 1
        min_prior = 5
        n_epochs = 2
        batch_size = 64
        eval_every = 10
        warmup_steps = 5
    else:
        print("Step 1: Loading events...", flush=True)
        events_df = load_real_events(data_dir)
        print(f"Loaded {len(events_df)} events", flush=True)

        print("Step 2: Loading embeddings...", flush=True)
        item_ids_in_events = set(events_df["item_id"].unique().to_list())
        embeddings = load_embeddings(data_dir, item_ids_needed=item_ids_in_events)
        print("Embeddings loaded", flush=True)

        n_days = 1504
        min_prior = cfg.get("min_prior_events", 50)
        n_epochs = cfg.get("n_epochs", 10)
        batch_size = cfg.get("batch_size", 2048)
        eval_every = cfg.get("eval_every_n_steps", 500)
        warmup_steps = 500

    temperature: float = cfg.get("temperature", 0.07)
    embed_dim: int = cfg.get("embed_dim", 128)
    history_len: int = cfg.get("history_len", 100)
    id_dropout_p: float = cfg.get("id_dropout_p", 0.1)
    lr: float = cfg.get("lr", 1e-3)
    weight_decay: float = cfg.get("weight_decay", 1e-5)
    patience: int = cfg.get("patience", 3)
    top_k_eval: list[int] = cfg.get("top_k_eval", [100, 500, 1000])
    window: int = cfg.get("retriever_train_window_days", 60)

    # --- item2idx from events (cheap; no snapshots needed yet) ---
    print("Step 3: Building item index and splits...", flush=True)
    item2idx, idx2item = build_item2idx(events_df)
    n_items = len(item2idx)

    # --- Cutoff timestamps ---
    N = n_days - 1
    train_end = N - 16
    train_start = max(0, train_end - window + 1)
    train_cutoffs = [d * DAY_UNITS for d in range(train_start, train_end + 1)]

    split_cfg = SplitConfig(n_days=n_days, min_prior_events=min_prior)

    # Load val snapshots from cache if available; otherwise build and cache.
    splits_cache = out_dir / "splits.pkl"
    if splits_cache.exists():
        with open(splits_cache, "rb") as _f:
            _cached = pickle.load(_f)
        val_snaps = _cached["retriever_val"]
        print("Loaded splits from cache", flush=True)
    else:
        val_cutoffs = [(N - 15) * DAY_UNITS]
        val_snaps = build_snapshots(events_df, val_cutoffs, split_cfg)
        with open(splits_cache, "wb") as _f:
            pickle.dump({"retriever_val": val_snaps}, _f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Built splits, saved to cache", flush=True)

    print(
        f"Training: {len(train_cutoffs)} cutoff days (lazy), "
        f"val snapshots: {len(val_snaps)}",
        flush=True,
    )

    # Item frequency from events for SBC loss correction.
    item_freq_iid = {
        int(r[0]): int(r[1])
        for r in events_df.group_by("item_id")
        .agg(pl.len().alias("freq"))
        .iter_rows()
    }
    item_freq = {item2idx[iid]: f for iid, f in item_freq_iid.items() if iid in item2idx}

    # --- Lazy training dataset ---
    print("Step 4: Building dataset...", flush=True)
    lazy_train = LazySnapshotIterator(
        events=events_df,
        config=split_cfg,
        cutoff_timestamps=train_cutoffs,
        item2idx=item2idx,
        item_freq=item_freq,
        history_len=history_len,
    )

    # Map audio embeddings to item_idx.
    audio_by_idx: dict[int, np.ndarray] = {
        item2idx[iid]: arr for iid, arr in embeddings.items() if iid in item2idx
    }

    loader = DataLoader(
        lazy_train,
        batch_size=batch_size,
        shuffle=False,  # LazySnapshotIterator shuffles internally
        num_workers=0,
        drop_last=True,
    )

    print("Starting training...", flush=True)

    # --- Model ---
    model = TwoTowerRetriever(
        n_items=n_items,
        embed_dim=embed_dim,
        audio_embeddings=audio_by_idx,
        id_dropout_p=id_dropout_p,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # IterableDataset has no len(); estimate steps from window × users × positives / batch.
    steps_per_epoch_est = max(len(train_cutoffs) * max(len(val_snaps), 10) // batch_size, 1)
    total_steps = n_epochs * steps_per_epoch_est
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_cosine_with_warmup(warmup_steps, total_steps),
    )

    # --- Training ---
    best_recall = -1.0
    no_improve_count = 0
    eval_log: list[dict] = []
    global_step = 0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in loader:
            history = batch["history"].to(device)
            mask = batch["history_mask"].to(device)
            positive_idx = batch["positive"].to(device)
            item_freqs = batch["item_freq"].to(device)

            user_emb, item_emb = model(history, mask, positive_idx)
            loss = in_batch_softmax_loss(user_emb, item_emb, item_freqs, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % eval_every == 0:
                mean_loss = running_loss / n_batches
                running_loss = 0.0
                n_batches = 0

                item_index = model.build_item_index(device=device)
                val_metrics = evaluate_retriever(
                    model=model,
                    snapshots=val_snaps,
                    item_index=item_index,
                    idx2item=idx2item,
                    events=events_df,
                    item2idx=item2idx,
                    history_len=history_len,
                    ks=top_k_eval,
                    device=device,
                )
                r100 = val_metrics.get("recall@100", 0.0)
                print(
                    f"step={global_step} epoch={epoch+1} "
                    f"loss={mean_loss:.4f} recall@100={r100:.4f}"
                )
                eval_log.append(
                    {"step": global_step, "loss": mean_loss, **val_metrics}
                )

                if r100 > best_recall:
                    best_recall = r100
                    no_improve_count = 0
                    torch.save(
                        {
                            "step": global_step,
                            "model_state_dict": model.state_dict(),
                            "recall@100": r100,
                            "n_items": n_items,
                            "embed_dim": embed_dim,
                            "history_len": history_len,
                        },
                        out_dir / "retriever_best.pt",
                    )
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(
                            f"Early stopping at step {global_step} "
                            f"(no improvement for {patience} eval points)"
                        )
                        break
                model.train()
            else:
                running_loss += 0.0  # already accounted for above

        else:
            continue
        break  # inner loop broke due to early stopping

    # Always do a final eval so the log has at least one entry.
    if not eval_log:
        item_index = model.build_item_index(device=device)
        val_metrics = evaluate_retriever(
            model=model,
            snapshots=val_snaps,
            item_index=item_index,
            idx2item=idx2item,
            events=events_df,
            item2idx=item2idx,
            history_len=history_len,
            ks=top_k_eval,
            device=device,
        )
        eval_log.append({"step": global_step, "loss": float("nan"), **val_metrics})

    # Save final checkpoint if best was never written (e.g., no eval point hit).
    if not (out_dir / "retriever_best.pt").exists():
        torch.save(
            {
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "recall@100": 0.0,
                "n_items": n_items,
                "embed_dim": embed_dim,
                "history_len": history_len,
            },
            out_dir / "retriever_best.pt",
        )

    log_path = out_dir / "retriever_train_log.json"
    with open(log_path, "w") as f:
        json.dump({"evals": eval_log}, f, indent=2)

    print(f"\nBest recall@100: {best_recall:.4f}")
    print(f"Checkpoint: {out_dir / 'retriever_best.pt'}")
    print(f"Training log: {log_path}")
