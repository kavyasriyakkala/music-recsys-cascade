"""CLI wrapper for train_retriever.

Usage:
    python scripts/train_retriever.py \
        --config configs/retriever.yaml \
        --data-dir /scratch/ky2751/yambda/50m \
        --output-dir /scratch/ky2751/outputs/retriever

    python scripts/train_retriever.py \
        --config configs/retriever.yaml \
        --data-dir none \
        --output-dir /tmp/retriever_smoke \
        --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yc.train.retriever_trainer import train_retriever

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    train_retriever(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        smoke=args.smoke,
    )
