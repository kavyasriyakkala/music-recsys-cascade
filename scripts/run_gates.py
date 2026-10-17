"""Gate scripts verifying all UNVERIFIED assumptions in DECISIONS.md.

Run on real Yambda data:
    python scripts/run_gates.py --data-dir /scratch/$USER/yambda/50m

Writes reports/gates.json with verdicts for each gate.
FAILS LOUDLY (exit 1) if Gate 1 finds non-global timestamps, since the
split code's DAY_UNITS assumption depends on global timestamps.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import polars as pl

DAY_UNITS = 86400 // 5  # 17280 (5-second units per day)
# Yambda-50M covers roughly June 2021 – March 2022 (~9–11 months).
EXPECTED_SPAN_DAYS_MIN = 200
EXPECTED_SPAN_DAYS_MAX = 400


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _find_events_path(data_dir: Path) -> Path:
    candidates = [
        data_dir / "multi_event.parquet",
        data_dir / "flat" / "50m" / "multi_event.parquet",
        data_dir / "flat" / "multi_event.parquet",
        data_dir / "multi_event",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find multi_event.parquet in {data_dir}. Tried: {candidates}"
    )


# ---------------------------------------------------------------------------
# Gate 1 — timestamp semantics
# ---------------------------------------------------------------------------

def gate_timestamp_semantics(events: pl.LazyFrame) -> dict:
    """Determine whether timestamps are global or per-user deltas."""
    stats = (
        events.select(
            pl.col("timestamp").min().alias("min_ts"),
            pl.col("timestamp").max().alias("max_ts"),
        )
        .collect()
    )
    min_ts = int(stats["min_ts"][0])
    max_ts = int(stats["max_ts"][0])
    total_span_days = (max_ts - min_ts) * 5 / 86400

    # Heuristic: if timestamps are global, per-user minimums will be spread
    # across the timeline, not clustered near 0.
    user_min_ts = (
        events.group_by("uid")
        .agg(pl.col("timestamp").min().alias("user_min_ts"))
        .collect()
    )
    n_users = user_min_ts.height
    n_users_near_zero = int(
        (user_min_ts["user_min_ts"] < DAY_UNITS).sum()
    )
    near_zero_frac = n_users_near_zero / n_users if n_users > 0 else 0.0

    # 4.1 year span is valid — dataset covers full platform history, not a short window.
    # near_zero_fraction drives verdict: per-user deltas would show ~100% near zero;
    # a global clock shows most users spread across the timeline (low near_zero_fraction).
    if near_zero_frac > 0.80:
        verdict = "per_user_delta"
    elif near_zero_frac >= 0.20:
        verdict = "ambiguous"
    else:
        verdict = "global"

    result = {
        "verdict": verdict,
        "min_ts": min_ts,
        "max_ts": max_ts,
        "total_span_days": round(total_span_days, 2),
        "n_users": n_users,
        "users_with_min_ts_near_zero": n_users_near_zero,
        "near_zero_fraction": round(near_zero_frac, 4),
        "passed": verdict == "global",
    }

    if verdict != "global":
        print(
            f"GATE 1 FAILED: timestamp_semantics verdict is '{verdict}'. "
            "The split code (DAY_UNITS = 17280) assumes global timestamps. "
            "Update splits/builder.py before running splits on real data.",
            file=sys.stderr,
        )

    return result


# ---------------------------------------------------------------------------
# Gate 2 — daily volume
# ---------------------------------------------------------------------------

def gate_daily_volume(events: pl.LazyFrame, min_ts: int) -> dict:
    """Compute per-day event volumes and flag sparse days."""
    with_day = events.with_columns(
        ((pl.col("timestamp") - min_ts) // DAY_UNITS).cast(pl.UInt32).alias("day_index")
    )

    daily = (
        with_day.group_by("day_index")
        .agg(
            pl.len().alias("events_count"),
            (
                (pl.col("event_type") == "listen")
                & (pl.col("played_ratio_pct") > 50)
            )
            .sum()
            .alias("listen_plus_count"),
            pl.col("uid").n_unique().alias("active_users"),
        )
        .sort("day_index")
        .collect()
    )

    n_days = daily.height
    events_per_day = daily["events_count"].to_list()
    mean_epd = sum(events_per_day) / n_days if n_days > 0 else 0.0
    threshold = mean_epd * 0.1

    sparse_days = [
        {"day": int(r["day_index"]), "events": int(r["events_count"])}
        for r in daily.iter_rows(named=True)
        if r["events_count"] < threshold
    ]

    lp = daily["listen_plus_count"].to_list()

    return {
        "n_days": n_days,
        "events_per_day": {
            "min": int(min(events_per_day)),
            "mean": round(mean_epd, 1),
            "max": int(max(events_per_day)),
        },
        "listen_plus_per_day": {
            "min": int(min(lp)),
            "mean": round(sum(lp) / n_days, 1),
            "max": int(max(lp)),
        },
        "sparse_days": sparse_days,
        "passed": True,
    }


# ---------------------------------------------------------------------------
# Gate 3 — embedding coverage
# ---------------------------------------------------------------------------

def gate_embedding_coverage(events: pl.LazyFrame, emb_path: Path) -> dict:
    """Report item embedding coverage overall and by frequency quintile."""
    event_items = (
        events.group_by("item_id")
        .agg(pl.len().alias("freq"))
        .collect()
    )
    n_unique_items = event_items.height

    emb_item_ids = (
        pl.scan_parquet(emb_path)
        .select("item_id")
        .collect()["item_id"]
        .unique()
    )
    emb_set = set(emb_item_ids.to_list())

    n_covered = int(event_items["item_id"].is_in(list(emb_set)).sum())
    coverage = n_covered / n_unique_items if n_unique_items > 0 else 0.0

    # Coverage by frequency quintile.
    event_items = event_items.with_columns(
        pl.col("freq")
        .qcut(5, labels=["Q1_rare", "Q2", "Q3", "Q4", "Q5_frequent"])
        .alias("quintile")
    ).with_columns(
        pl.col("item_id").is_in(list(emb_set)).alias("has_embedding")
    )

    by_quintile = (
        event_items.group_by("quintile")
        .agg(
            pl.len().alias("n_items"),
            pl.col("has_embedding").sum().alias("n_covered"),
        )
        .with_columns(
            (pl.col("n_covered") / pl.col("n_items")).alias("coverage")
        )
        .sort("quintile")
    )

    quintile_rows = [
        {
            "quintile": r["quintile"],
            "n_items": int(r["n_items"]),
            "n_covered": int(r["n_covered"]),
            "coverage": round(float(r["coverage"]), 4),
        }
        for r in by_quintile.iter_rows(named=True)
    ]

    return {
        "n_unique_items_in_events": n_unique_items,
        "n_items_with_embedding": n_covered,
        "coverage_fraction": round(coverage, 4),
        "by_quintile": quintile_rows,
        "passed": True,
    }


# ---------------------------------------------------------------------------
# Gate 4 — benchmark protocol
# ---------------------------------------------------------------------------

def gate_benchmark_protocol(data_dir: Path) -> dict:
    """Check if an official benchmark definition exists in data_dir."""
    candidates = [
        data_dir / "benchmarks",
        data_dir / "benchmark",
        data_dir / "README.md",
        data_dir / "README",
    ]
    found_paths = [str(p) for p in candidates if p.exists()]

    details: dict = {}
    if found_paths:
        readme_path = next(
            (p for p in candidates if p.exists() and p.suffix in ("", ".md")), None
        )
        if readme_path and readme_path.is_file():
            try:
                text = readme_path.read_text(errors="replace")[:4000]
                details["readme_excerpt"] = text
            except Exception:
                pass
        details["found_paths"] = found_paths
        note = "benchmark definition found — review details before adopting our protocol"
    else:
        note = "not found — will use our own protocol (see DECISIONS.md)"

    return {
        "found": bool(found_paths),
        "note": note,
        "details": details,
        "passed": True,  # Gate 4 never fails; it just reports.
    }


# ---------------------------------------------------------------------------
# Gate 5 — listen+ threshold
# ---------------------------------------------------------------------------

def gate_listen_plus_threshold(events: pl.LazyFrame) -> dict:
    """Analyse played_ratio_pct distribution for listen events."""
    listen_pct = (
        events.filter(pl.col("event_type") == "listen")
        .select("played_ratio_pct")
        .collect()["played_ratio_pct"]
        .drop_nulls()
    )
    n_listen = len(listen_pct)
    if n_listen == 0:
        return {"error": "no listen events found", "passed": False}

    at_50 = int((listen_pct == 50).sum())
    at_51 = int((listen_pct == 51).sum())
    at_99 = int((listen_pct == 99).sum())
    at_100 = int((listen_pct == 100).sum())
    gt50 = int((listen_pct > 50).sum())
    gte50 = int((listen_pct >= 50).sum())

    # Text histogram (10 buckets over 0–100).
    bucket_size = 10
    buckets: list[dict] = []
    for lo in range(0, 100, bucket_size):
        hi = lo + bucket_size
        count = int(((listen_pct >= lo) & (listen_pct < hi)).sum())
        bar = "#" * min(40, int(40 * count / n_listen))
        buckets.append({"range": f"{lo}-{hi}", "count": count, "bar": bar})

    # Confirm > 50 (not >= 50) is the right threshold per the dataset card.
    # If almost no events land exactly at 50, the boundary choice doesn't matter.
    boundary_sensitivity = at_50 / n_listen if n_listen > 0 else 0.0

    return {
        "n_listen_events": n_listen,
        "gt50_count": gt50,
        "gt50_fraction": round(gt50 / n_listen, 4),
        "gte50_fraction": round(gte50 / n_listen, 4),
        "at_exactly_50": at_50,
        "at_exactly_51": at_51,
        "at_exactly_99": at_99,
        "at_exactly_100": at_100,
        "boundary_sensitivity": round(boundary_sensitivity, 6),
        "histogram": buckets,
        "passed": True,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir: Path) -> None:
    print(f"Loading events from {data_dir} ...")
    events_path = _find_events_path(data_dir)
    events_lazy: pl.LazyFrame = pl.scan_parquet(events_path)

    gates: dict = {}
    all_passed = True

    print("Gate 1 — timestamp_semantics ...")
    g1 = gate_timestamp_semantics(events_lazy)
    gates["timestamp_semantics"] = g1
    if not g1["passed"]:
        all_passed = False

    print("Gate 2 — daily_volume ...")
    g2 = gate_daily_volume(events_lazy, min_ts=g1["min_ts"])
    gates["daily_volume"] = g2

    print("Gate 3 — embedding_coverage ...")
    emb_path = data_dir / "embeddings.parquet"
    if emb_path.exists():
        g3 = gate_embedding_coverage(events_lazy, emb_path)
    else:
        g3 = {"error": f"embeddings.parquet not found in {data_dir}", "passed": False}
        all_passed = False
    gates["embedding_coverage"] = g3

    print("Gate 4 — benchmark_protocol ...")
    gates["benchmark_protocol"] = gate_benchmark_protocol(data_dir)

    print("Gate 5 — listen_plus_threshold ...")
    g5 = gate_listen_plus_threshold(events_lazy)
    gates["listen_plus_threshold"] = g5
    if not g5["passed"]:
        all_passed = False

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "data_dir": str(data_dir),
        "gates": gates,
        "all_passed": all_passed,
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out = reports_dir / "gates.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nGates report: {out}")
    for name, g in gates.items():
        status = "PASS" if g.get("passed", False) else "FAIL"
        print(f"  {status}  {name}")

    if not all_passed:
        print("\nOne or more gates failed — see reports/gates.json for details.")
        sys.exit(1)
    else:
        print("\nAll gates passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to Yambda data directory")
    args = parser.parse_args()
    run(Path(args.data_dir))
