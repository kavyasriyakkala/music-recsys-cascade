"""One-time setup: build all splits and save to disk.

Usage:
    python scripts/precompute_splits.py \
        --data-dir /scratch/ky2751/yambda/50m \
        --output-dir /scratch/ky2751/outputs/retriever
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yc.data.real_loader import load_real_events
from yc.splits.builder import DAY_UNITS, SplitConfig, build_snapshots

N_DAYS = 1504
RETRIEVER_TRAIN_WINDOW_DAYS = 60
MIN_PRIOR_EVENTS = 50


def precompute_splits(data_dir: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "splits.pkl"

    if out_path.exists():
        print(f"{out_path} already exists — delete it to regenerate.", flush=True)
        return

    print("Loading events...", flush=True)
    events = load_real_events(data_dir)
    print(f"Loaded {len(events)} events", flush=True)

    cfg = SplitConfig(
        n_days=N_DAYS,
        min_prior_events=MIN_PRIOR_EVENTS,
        retriever_train_window_days=RETRIEVER_TRAIN_WINDOW_DAYS,
    )
    N = N_DAYS - 1
    train_end = N - 16
    train_start = max(0, train_end - RETRIEVER_TRAIN_WINDOW_DAYS + 1)

    print("Building retriever_train snapshots...", flush=True)
    train_snaps = build_snapshots(
        events,
        [d * DAY_UNITS for d in range(train_start, train_end + 1)],
        cfg,
    )
    print(f"retriever_train: {len(train_snaps)} snapshots", flush=True)

    print("Building retriever_val snapshots...", flush=True)
    val_snaps = build_snapshots(events, [(N - 15) * DAY_UNITS], cfg)
    print(f"retriever_val: {len(val_snaps)} snapshots", flush=True)

    splits = {"retriever_train": train_snaps, "retriever_val": val_snaps}

    print(f"Saving splits to {out_path} ...", flush=True)
    with open(out_path, "wb") as f:
        pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = out_path.stat().st_size / 1024**2
    print(f"Saved ({size_mb:.0f} MB)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    precompute_splits(args.data_dir, args.output_dir)
