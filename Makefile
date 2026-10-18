.PHONY: setup test lint smoke check-leakage gates train-retriever smoke-retriever \
        generate-candidates train-rankers smoke-rankers run-history-study

setup:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

smoke:
	python scripts/run_smoke.py --smoke

gates:
	python scripts/run_gates.py --data-dir $(DATA_DIR)

train-retriever:
	python scripts/train_retriever.py \
		--config configs/retriever.yaml \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR)

smoke-retriever:
	python scripts/train_retriever.py \
		--config configs/retriever.yaml \
		--data-dir none \
		--output-dir /tmp/retriever_smoke \
		--smoke

generate-candidates:
	python scripts/generate_candidates.py \
		--retriever-checkpoint $(RETRIEVER_CKPT) \
		--data-dir $(DATA_DIR) \
		--splits-cache $(SPLITS_CACHE) \
		--output-dir $(CANDIDATES_DIR) \
		--top-k 1000 \
		--device cuda

train-rankers:
	python scripts/train_rankers.py \
		--config configs/ranker.yaml \
		--retriever-checkpoint $(RETRIEVER_CKPT) \
		--candidates-dir $(CANDIDATES_DIR) \
		--splits-cache $(SPLITS_CACHE) \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR)

smoke-rankers:
	python scripts/train_rankers.py \
		--config configs/ranker.yaml \
		--retriever-checkpoint none \
		--candidates-dir none \
		--splits-cache none \
		--data-dir none \
		--output-dir /tmp/ranker_smoke \
		--smoke

run-history-study:
	python scripts/run_history_study.py \
		--config configs/ranker.yaml \
		--retriever-checkpoint $(RETRIEVER_CKPT) \
		--candidates-dir $(CANDIDATES_DIR) \
		--splits-cache $(SPLITS_CACHE) \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR)/history_study \
		--device cuda

check-leakage:
	python -c "\
from yc.data.synthetic import generate_dataset; \
from yc.splits.builder import build_all_splits, check_leakage, SplitConfig; \
events, _, _ = generate_dataset(); \
splits = build_all_splits(events, SplitConfig(n_days=30)); \
all_snaps = [s for v in splits.values() for s in v]; \
check_leakage(all_snaps); \
print('check-leakage passed: {} snapshots'.format(len(all_snaps)))"
