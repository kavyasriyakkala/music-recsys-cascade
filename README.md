# Yambda Cascade

Three-stage recommender (retrieval → L1 → L2) on Yambda-50M. Predicts which tracks a user will Listen+ (played_ratio_pct > 50) during the day after a cutoff. Offline-evaluation portfolio project.

## Setup

```bash
pip install -e ".[dev]"
```

## Running on Torch (SLURM)

```bash
# Download data
python scripts/download_yambda.py

# Verify assumptions on real data
make gates DATA_DIR=/scratch/ky2751/yambda/50m

# Train retriever — submit via sbatch
sbatch scripts/train_retriever.sh
```

### sbatch template

```bash
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
```

Override data and output dirs with env vars:

```bash
DATA_DIR=/scratch/ky2751/yambda/50m \
OUTPUT_DIR=/scratch/ky2751/outputs/retriever \
sbatch scripts/train_retriever.sh
```

## Local smoke tests

```bash
make test            # all unit tests
make smoke           # synthetic end-to-end smoke
make smoke-retriever # retriever smoke (<2 min on CPU)
```

## Repo layout

```
configs/          hyperparameter YAML files
scripts/          CLI entry points and SLURM scripts
src/yc/
  data/           loaders, synthetic generator, dataset classes
  splits/         snapshot builder (leakage-safe)
  models/         two-tower model
  train/          loss, trainer
  eval/           retriever metrics + ANN comparison
tests/            pytest suite
reports/          gates.json and training logs (git-ignored)
DECISIONS.md      verified and unverified dataset assumptions
AI_LOG.md         what AI wrote and how it was verified
```
