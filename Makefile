.PHONY: setup test lint smoke

setup:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

smoke:
	python scripts/run_smoke.py --smoke
