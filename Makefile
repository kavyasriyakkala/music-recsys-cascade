.PHONY: setup test lint smoke check-leakage gates

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

check-leakage:
	python -c "\
from yc.data.synthetic import generate_dataset; \
from yc.splits.builder import build_all_splits, check_leakage, SplitConfig; \
events, _, _ = generate_dataset(); \
splits = build_all_splits(events, SplitConfig(n_days=30)); \
all_snaps = [s for v in splits.values() for s in v]; \
check_leakage(all_snaps); \
print('check-leakage passed: {} snapshots'.format(len(all_snaps)))"
