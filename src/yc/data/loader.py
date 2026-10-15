"""Real data loader for Yambda-50M parquet files.

Fails loudly on missing files, schema mismatches, or sort violations.
Never silently substitutes defaults.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)

# Canonical schema for the multi_event table.
# Source: Yambda-50M dataset card (yandex/yambda on HuggingFace).
_REQUIRED_COLUMNS: dict[str, pl.PolarsDataType] = {
    "uid": pl.UInt32,
    "item_id": pl.UInt32,
    "timestamp": pl.UInt32,
    "is_organic": pl.UInt8,
    "event_type": pl.Utf8,
    "played_ratio_pct": pl.UInt16,
    "track_length_seconds": pl.UInt32,
}

# If the embeddings table has more rows than this threshold, we avoid loading
# all embed vectors into RAM (each float32 vector is 512+ bytes × millions of items).
# At 50M scale, the catalog has ~3M items; 500M+ could exceed available memory.
_EMBEDDINGS_EAGER_THRESHOLD: int = 2_000_000

SMOKE_EVENTS_ROWS: int = 10_000
SMOKE_EMBEDDINGS_ROWS: int = 1_000


class DataLoaderError(Exception):
    """Raised when the data directory, file layout, or schema is invalid."""


def _find_events_path(data_dir: Path) -> Path:
    candidates = [
        data_dir / "multi_event.parquet",
        data_dir / "multi_event",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise DataLoaderError(
        f"Cannot find multi_event.parquet or multi_event/ in {data_dir}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _validate_and_cast(df: pl.DataFrame) -> pl.DataFrame:
    """Assert required columns are present; cast to canonical dtypes.

    Raises DataLoaderError with a clear message — never KeyError or AttributeError.
    """
    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataLoaderError(
            f"events schema mismatch — missing columns: {missing}. "
            f"Present columns: {df.columns}"
        )

    exprs: list[pl.Expr] = []
    for col, dtype in _REQUIRED_COLUMNS.items():
        if df.schema[col] != dtype:
            try:
                exprs.append(pl.col(col).cast(dtype))
            except Exception as exc:
                raise DataLoaderError(
                    f"Cannot cast column '{col}' "
                    f"from {df.schema[col]} to {dtype}: {exc}"
                ) from exc

    return df.with_columns(exprs) if exprs else df


def _check_sort_order(events: pl.DataFrame) -> None:
    """Verify a 1% slice is sorted by (uid, timestamp).

    Takes the first max(100, 1% of rows) rows to preserve file order.
    Raises DataLoaderError if the slice is not in sorted order.
    """
    n_check = min(events.height, max(100, int(events.height * 0.01)))
    sample = events[:n_check].select(["uid", "timestamp"])
    if not sample.equals(sample.sort(["uid", "timestamp"])):
        raise DataLoaderError(
            f"events are NOT sorted by (uid, timestamp) in the first {n_check} rows. "
            "Ensure the parquet file is globally sorted before loading."
        )


def load_yambda(
    data_dir: str | Path,
    scale: Literal["50m", "500m", "5b"] = "50m",
    event_types: list[str] | None = None,
    smoke: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame | pl.LazyFrame, pl.DataFrame]:
    """Load Yambda multi_event, embeddings, and item_artist tables.

    Returns (events, embeddings, item_artist).

    Fails loudly if:
    - any expected file is missing
    - schema does not match expected columns and dtypes
    - events are not sorted by (uid, timestamp) — checks a 1% sample

    Embedding loading strategy:
    - If the embeddings table has more than _EMBEDDINGS_EAGER_THRESHOLD rows,
      returns a LazyFrame instead of DataFrame to avoid OOM on large catalogs.
      Callers must filter to their candidate set and call .collect() before use.
    - In smoke mode the first SMOKE_EMBEDDINGS_ROWS rows are loaded eagerly
      regardless of catalog size.

    smoke=True loads only the first SMOKE_EVENTS_ROWS / SMOKE_EMBEDDINGS_ROWS
    rows for fast iteration.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise DataLoaderError(f"data_dir does not exist: {data_dir}")

    # --- events ---
    events_path = _find_events_path(data_dir)

    if smoke:
        events_raw = pl.read_parquet(events_path, n_rows=SMOKE_EVENTS_ROWS)
    else:
        events_raw = pl.read_parquet(events_path)

    events = _validate_and_cast(events_raw)

    if event_types is not None:
        events = events.filter(pl.col("event_type").is_in(event_types))

    _check_sort_order(events)

    # --- embeddings ---
    emb_path = data_dir / "embeddings.parquet"
    if not emb_path.exists():
        raise DataLoaderError(f"embeddings.parquet not found in {data_dir}")

    embeddings: pl.DataFrame | pl.LazyFrame
    if smoke:
        embeddings = pl.read_parquet(emb_path, n_rows=SMOKE_EMBEDDINGS_ROWS)
    else:
        n_emb = pl.scan_parquet(emb_path).select(pl.len()).collect().item()
        if n_emb > _EMBEDDINGS_EAGER_THRESHOLD:
            logger.info(
                "Embeddings has %d rows (> %d threshold); returning LazyFrame. "
                "Call .filter(...).collect() before indexing.",
                n_emb,
                _EMBEDDINGS_EAGER_THRESHOLD,
            )
            embeddings = pl.scan_parquet(emb_path)
        else:
            embeddings = pl.read_parquet(emb_path)

    # --- item_artist (optional) ---
    item_artist_candidates = [
        data_dir / "item.parquet",
        data_dir / "item_artist.parquet",
        data_dir / "items.parquet",
    ]
    item_artist = pl.DataFrame()
    for p in item_artist_candidates:
        if p.exists():
            item_artist = pl.read_parquet(p)
            logger.info("Loaded item metadata from %s", p)
            break
    else:
        logger.warning(
            "No item metadata file found in %s (tried %s); item_artist is empty.",
            data_dir,
            [p.name for p in item_artist_candidates],
        )

    return events, embeddings, item_artist
