"""One-time setup: filter embeddings.parquet to only items in 50M events.

Creates data_dir/embeddings_50m.parquet (much smaller than the full 13GB file),
which load_embeddings() will use automatically on subsequent training runs.

Usage:
    python scripts/prefilter_embeddings.py --data-dir /scratch/ky2751/yambda/50m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import polars as pl


def _find_events_path(data_dir: Path) -> Path:
    for p in [
        data_dir / "multi_event.parquet",
        data_dir / "flat" / "50m" / "multi_event.parquet",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find multi_event.parquet in {data_dir}."
    )


def prefilter_embeddings(data_dir: str) -> None:
    data = Path(data_dir)
    out_path = data / "embeddings_50m.parquet"

    if out_path.exists():
        print(f"{out_path} already exists — delete it to regenerate.", flush=True)
        return

    emb_path = data / "embeddings.parquet"
    if not emb_path.exists():
        raise FileNotFoundError(f"embeddings.parquet not found in {data}")

    print("Reading item_ids from events...", flush=True)
    events_path = _find_events_path(data)
    item_ids = (
        pl.scan_parquet(events_path)
        .select("item_id")
        .unique()
        .collect()["item_id"]
        .to_list()
    )
    print(f"Found {len(item_ids)} unique item_ids in events", flush=True)

    print(f"Filtering embeddings.parquet → {out_path.name} ...", flush=True)
    (
        pl.scan_parquet(emb_path)
        .filter(pl.col("item_id").is_in(item_ids))
        .select(["item_id", "normalized_embed"])
        .sink_parquet(out_path)
    )

    n = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    size_mb = out_path.stat().st_size / 1024**2
    print(
        f"Saved {n} embeddings to {out_path} ({size_mb:.0f} MB)",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    prefilter_embeddings(args.data_dir)
