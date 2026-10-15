"""Tests for src/yc/data/loader.py using synthetic parquet files."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from yc.data.loader import (
    DataLoaderError,
    _REQUIRED_COLUMNS,
    load_yambda,
)
from yc.data.synthetic import generate_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_parquet(tmp_path: Path) -> Path:
    """Write synthetic data to tmp_path as parquet files matching the loader's schema."""
    events, embeddings, item_artist = generate_dataset(
        n_users=20, n_items=100, n_days=10, n_latent=3, embedding_dim=8, seed=7,
    )

    # Cast to the canonical schema expected by the loader.
    cast_exprs = [pl.col(col).cast(dtype) for col, dtype in _REQUIRED_COLUMNS.items()]
    events_cast = events.with_columns(cast_exprs)

    events_cast.write_parquet(tmp_path / "multi_event.parquet")
    embeddings.write_parquet(tmp_path / "embeddings.parquet")
    item_artist.write_parquet(tmp_path / "item_artist.parquet")
    return tmp_path


# ---------------------------------------------------------------------------
# test_loader_smoke
# ---------------------------------------------------------------------------

def test_loader_smoke(tmp_path: Path) -> None:
    """load_yambda in smoke mode against synthetic parquet files.

    Asserts:
    - returns without error
    - events schema matches canonical dtypes
    - sort check does not raise
    """
    _write_synthetic_parquet(tmp_path)

    events, embeddings, item_artist = load_yambda(tmp_path, smoke=True)

    # Schema check.
    for col, expected_dtype in _REQUIRED_COLUMNS.items():
        assert col in events.columns, f"Missing column '{col}' after loading"
        assert events.schema[col] == expected_dtype, (
            f"Column '{col}': expected {expected_dtype}, got {events.schema[col]}"
        )

    # Row count is bounded by SMOKE_EVENTS_ROWS.
    from yc.data.loader import SMOKE_EVENTS_ROWS
    assert events.height <= SMOKE_EVENTS_ROWS

    # Embeddings loaded (small catalog → DataFrame, not LazyFrame).
    assert isinstance(embeddings, pl.DataFrame)
    assert embeddings.height > 0

    # item_artist loaded (we wrote it).
    assert isinstance(item_artist, pl.DataFrame)


# ---------------------------------------------------------------------------
# test_loader_schema_mismatch
# ---------------------------------------------------------------------------

def test_loader_schema_mismatch(tmp_path: Path) -> None:
    """Loader must raise DataLoaderError with a clear message for missing columns."""
    _write_synthetic_parquet(tmp_path)

    # Overwrite multi_event.parquet with a file missing 'timestamp'.
    events, _, _ = generate_dataset(
        n_users=10, n_items=50, n_days=5, n_latent=2, embedding_dim=4, seed=9,
    )
    bad_events = events.drop("timestamp")
    bad_events.write_parquet(tmp_path / "multi_event.parquet")

    with pytest.raises(DataLoaderError, match="missing columns"):
        load_yambda(tmp_path, smoke=True)


# ---------------------------------------------------------------------------
# test_loader_unsorted_raises
# ---------------------------------------------------------------------------

def test_loader_unsorted_raises(tmp_path: Path) -> None:
    """Loader must raise DataLoaderError when events are not sorted by (uid, timestamp)."""
    _write_synthetic_parquet(tmp_path)

    # Overwrite multi_event.parquet with events sorted in reverse order.
    events, _, _ = generate_dataset(
        n_users=20, n_items=100, n_days=10, n_latent=3, embedding_dim=8, seed=7,
    )
    cast_exprs = [pl.col(col).cast(dtype) for col, dtype in _REQUIRED_COLUMNS.items()]
    reversed_events = (
        events.with_columns(cast_exprs)
        .sort(["uid", "timestamp"], descending=True)  # reverse order
    )
    reversed_events.write_parquet(tmp_path / "multi_event.parquet")

    with pytest.raises(DataLoaderError, match="NOT sorted"):
        load_yambda(tmp_path, smoke=True)
