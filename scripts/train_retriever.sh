#!/bin/bash
#SBATCH --job-name=yambda-retriever
#SBATCH --output=logs/retriever_%j.out
#SBATCH --error=logs/retriever_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -euo pipefail

DATA_DIR=${DATA_DIR:-/scratch/ky2751/yambda/50m}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/ky2751/outputs/retriever}

mkdir -p logs "$OUTPUT_DIR"

python scripts/train_retriever.py \
    --config configs/retriever.yaml \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR"
