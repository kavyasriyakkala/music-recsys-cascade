"""Focused loaders for real Yambda-50M data used by the retriever pipeline.

Distinct from loader.py (which handles the full multi-format load). These
functions fail loudly on missing files, schema mismatches, or type errors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

_REQUIRED_COLUMNS: dict[str, pl.PolarsDataType] = {
    "uid": pl.UInt32,
    "item_id": pl.UInt32,
    "timestamp": pl.UInt32,
    "is_organic": pl.UInt8,
    "event_type": pl.Utf8,
    "played_ratio_pct": pl.UInt16,
    "track_length_seconds": pl.UInt32,
}


def _find_events_path(data_dir: Path) -> Path:
    for p in [
        data_dir / "multi_event.parquet",
        data_dir / "flat" / "50m" / "multi_event.parquet",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find multi_event.parquet in {data_dir}. "
        f"Tried: multi_event.parquet, flat/50m/multi_event.parquet"
    )


def load_real_events(data_dir: str | Path) -> pl.DataFrame:
    """Load and return listen-only events sorted by (uid, timestamp).

    Validates schema; fails loudly on missing columns or bad dtypes.
    Filters to event_type == 'listen' only.
    """
    data_dir = Path(data_dir)
    path = _find_events_path(data_dir)
    df = pl.read_parquet(path)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"events schema mismatch — missing columns: {missing}. "
            f"Present: {df.columns}"
        )

    casts = [
        pl.col(c).cast(t)
        for c, t in _REQUIRED_COLUMNS.items()
        if df.schema[c] != t
    ]
    if casts:
        df = df.with_columns(casts)

    df = df.filter(pl.col("event_type") == "listen")
    return df.sort(["uid", "timestamp"])


def load_embeddings(
    data_dir: str | Path,
    event_item_ids: set[int] | None = None,
) -> dict[int, np.ndarray]:
    """Load embeddings.parquet and return {item_id: normalized_embed array}.

    If event_item_ids is provided, filters to items present in events (saves memory).
    Uses the normalized_embed column.
    """
    data_dir = Path(data_dir)
    path = data_dir / "embeddings.parquet"
    if not path.exists():
        raise FileNotFoundError(f"embeddings.parquet not found in {data_dir}")

    df = pl.read_parquet(path)
    if event_item_ids is not None:
        df = df.filter(pl.col("item_id").is_in(list(event_item_ids)))

    if "normalized_embed" not in df.columns:
        raise ValueError(
            f"embeddings.parquet missing 'normalized_embed' column. "
            f"Present: {df.columns}"
        )

    result: dict[int, np.ndarray] = {}
    for row in df.select(["item_id", "normalized_embed"]).iter_rows():
        iid, vec = row
        result[int(iid)] = np.array(vec, dtype=np.float32)

    if result:
        dim = next(iter(result.values())).shape[0]
        total_items = len(event_item_ids) if event_item_ids else "?"
        print(
            f"Loaded {len(result)} embeddings (dim={dim}); "
            f"coverage over events: {len(result)}/{total_items}"
        )

    return result


def load_artist_mapping(data_dir: str | Path) -> dict[int, int]:
    """Load artist_item_mapping.parquet and return {item_id: artist_id}."""
    data_dir = Path(data_dir)
    path = data_dir / "artist_item_mapping.parquet"
    if not path.exists():
        raise FileNotFoundError(f"artist_item_mapping.parquet not found in {data_dir}")

    df = pl.read_parquet(path)
    return {int(r[0]): int(r[1]) for r in df.select(["item_id", "artist_id"]).iter_rows()}
