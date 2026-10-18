"""Run the long-history ablation study.

Usage:
    python scripts/run_history_study.py \\
        --config configs/ranker.yaml \\
        --retriever-checkpoint /scratch/ky2751/outputs/retriever/retriever_best.pt \\
        --candidates-dir /scratch/ky2751/outputs/candidates \\
        --splits-cache /scratch/ky2751/outputs/retriever/splits.pkl \\
        --data-dir /scratch/ky2751/yambda/50m \\
        --output-dir /scratch/ky2751/outputs/history_study \\
        --device cuda \\
        --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_config(config_path: str) -> dict:
    import yaml  # type: ignore
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--retriever-checkpoint", required=True)
    parser.add_argument("--candidates-dir", required=True)
    parser.add_argument("--splits-cache", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()

    config = _load_config(args.config)

    from yc.data.real_loader import load_real_events, load_artist_mapping
    from yc.eval.long_history_study import run_history_study

    print("Loading events...", flush=True)
    events = load_real_events(args.data_dir)

    print("Loading artist mapping...", flush=True)
    item_artist = load_artist_mapping(args.data_dir)

    results = run_history_study(
        base_config=config,
        retriever_checkpoint=args.retriever_checkpoint,
        candidates_dir=args.candidates_dir,
        splits_cache=args.splits_cache,
        events=events,
        item_artist=item_artist,
        output_dir=args.output_dir,
        device=args.device,
        seeds=args.seeds,
    )

    print("\n=== Top-3 configs ===", flush=True)
    for name, v in results.get("top3_full", {}).items():
        print(f"  {name}: NDCG@10 = {v['mean']:.4f} ± {v['std']:.4f}", flush=True)


if __name__ == "__main__":
    main()
