.PHONY: help install test lint format type-check train backtest paper-trade serve-api serve-mcp docker-build docker-up clean

help:
	@echo "FX ML Trading System - Available commands:"
	@echo "  install        - Install dependencies"
	@echo "  test           - Run tests"
	@echo "  lint           - Run ruff linter"
	@echo "  format         - Format code with black"
	@echo "  type-check     - Run mypy type checker"
	@echo "  download-data  - Download 15 years of OANDA data"
	@echo "  train          - Train models"
	@echo "  backtest       - Run backtest"
	@echo "  paper-trade    - Start paper trading on demo account"
	@echo "  serve-api      - Start API server"
	@echo "  serve-mcp      - Start MCP server"
	@echo "  docker-build   - Build Docker images"
	@echo "  docker-up      - Start Docker services"
	@echo "  clean          - Clean build artifacts"

install:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/

type-check:
	mypy src/

download-data:
	python scripts/download_data.py --full

train:
	python scripts/train_models.py

backtest:
	python scripts/run_backtest.py --model latest

paper-trade:
	python scripts/start_paper_trading.py

serve-api:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

serve-mcp:
	python -m src.serving.mcp_server models/ensemble_latest data/processed

docker-build:
	docker build -f docker/Dockerfile.train -t fx-ml-train .
	docker build -f docker/Dockerfile.serve -t fx-ml-serve .

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete