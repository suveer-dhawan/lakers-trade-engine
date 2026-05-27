.PHONY: setup test notebook dashboard lint format clean

setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev,dashboard]"

test:
	.venv/bin/pytest tests/ -v

notebook:
	.venv/bin/jupyter notebook notebooks/

dashboard:
	.venv/bin/streamlit run dashboard/app.py

lint:
	.venv/bin/ruff check src/ tests/

format:
	.venv/bin/black src/ tests/ notebooks/
	.venv/bin/ruff check --fix src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
