"""Smoke pipeline: synthetic data -> splits -> baselines -> metrics -> reports/smoke.json.

Run with:
    python scripts/run_smoke.py --smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Ensure src/ is on sys.path when run directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yc.data.synthetic import generate_dataset
from yc.eval.metrics import evaluate
from yc.models.baselines import (
    PopularityBaseline,
    UserMostListenedBaseline,
    UserMostRecentBaseline,
)
from yc.splits.builder import SplitConfig, build_all_splits, check_leakage


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(config_path: Path, smoke: bool) -> None:
    cfg = _load_config(config_path)

    ds_cfg = cfg["dataset"]
    sp_cfg = cfg["splits"]
    ev_cfg = cfg["eval"]
    bl_cfg = cfg["baselines"]

    if smoke:
        # Reduce scale for fast smoke run.
        ds_cfg = {**ds_cfg, "n_users": 50, "n_items": 500, "n_days": 25}
        sp_cfg = {**sp_cfg, "n_days": 25, "min_prior_events": 5}

    t0 = time.time()

    print("Generating synthetic dataset...")
    events, embeddings, item_artist = generate_dataset(
        n_users=ds_cfg["n_users"],
        n_items=ds_cfg["n_items"],
        n_days=ds_cfg["n_days"],
        n_latent=ds_cfg.get("n_latent", 5),
        embedding_dim=ds_cfg.get("embedding_dim", 64),
        embedding_coverage=ds_cfg.get("embedding_coverage", 0.8),
        listen_plus_threshold=sp_cfg.get("listen_plus_threshold", 0.5),
        seed=ds_cfg["seed"],
    )
    print(f"  events: {events.height:,}  users: {events['uid'].n_unique()}  items: {events['item_id'].n_unique()}")

    split_config = SplitConfig(
        n_days=sp_cfg["n_days"],
        min_prior_events=sp_cfg["min_prior_events"],
        label_hours=sp_cfg.get("label_hours", 24),
        listen_plus_threshold=sp_cfg.get("listen_plus_threshold", 0.5),
    )

    print("Building splits...")
    splits = build_all_splits(events, split_config)
    for name, snaps in splits.items():
        print(f"  {name}: {len(snaps)} snapshots")

    print("Checking leakage...")
    for name, snaps in splits.items():
        check_leakage(snaps)
    print("  No leakage detected.")

    # Use validation split for evaluation.
    val_snaps = splits["validation"]
    if not val_snaps:
        raise RuntimeError("No validation snapshots — synthetic data may be too small.")

    ks = ev_cfg.get("ks", [10, 50, 100])
    eval_uids = [s.uid for s in val_snaps]
    from yc.splits.builder import DAY_UNITS
    val_cutoff = val_snaps[0].cutoff

    print("Running baselines...")
    baseline_results: dict[str, dict] = {}

    pop = PopularityBaseline()
    pop.fit(events, val_cutoff, window_days=bl_cfg.get("popularity_window_days", 7))
    pop_recs = pop.predict(eval_uids, k=max(ks))
    baseline_results["popularity"] = _summarise(evaluate(val_snaps, pop_recs, ks))

    uml = UserMostListenedBaseline()
    uml.fit(events, val_cutoff)
    uml_recs = uml.predict(eval_uids, k=max(ks))
    baseline_results["user_most_listened"] = _summarise(evaluate(val_snaps, uml_recs, ks))

    umr = UserMostRecentBaseline()
    umr.fit(events, val_cutoff)
    umr_recs = umr.predict(eval_uids, k=max(ks))
    baseline_results["user_most_recent"] = _summarise(evaluate(val_snaps, umr_recs, ks))

    elapsed = time.time() - t0

    report = {
        "git_commit": _git_commit(),
        "config": {"dataset": ds_cfg, "splits": sp_cfg, "eval": ev_cfg, "baselines": bl_cfg},
        "seed": ds_cfg["seed"],
        "dataset_scale": {
            "users": events["uid"].n_unique(),
            "items": events["item_id"].n_unique(),
            "events": events.height,
            "val_snapshots": len(val_snaps),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "results": baseline_results,
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "smoke.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSmoke pipeline complete in {elapsed:.1f}s. Report: {out_path}")
    for bl_name, res in baseline_results.items():
        r10 = res.get("recall@10_mean", "n/a")
        print(f"  {bl_name}: recall@10={r10:.4f}" if isinstance(r10, float) else f"  {bl_name}: {r10}")


def _summarise(metrics: dict) -> dict:
    """Flatten per_user arrays to just mean values for JSON serialisation."""
    out: dict = {}
    for key, val in metrics.items():
        if isinstance(val, dict):
            out[f"{key}_mean"] = val["mean"]
        else:
            out[key] = val
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run on tiny synthetic data")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.yaml"))
    args = parser.parse_args()
    run(Path(args.config), smoke=args.smoke)
