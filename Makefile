# Makefile for FiberWatch development

.PHONY: help install install-dev clean test test-cov lint format type-check docs docs-serve build upload dev web

help:  ## Show this help message
	@echo "FiberWatch Development Commands"
	@echo "==============================="
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install package
	uv pip install -e .

install-dev:  ## Install development dependencies
	uv pip install -e ".[dev,test,docs]"

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=fiberwatch --cov-report=html --cov-report=term-missing

lint:  ## Run linting
	flake8 fiberwatch/ tests/
	bandit -r fiberwatch/

format:  ## Format code
	black fiberwatch/ tests/
	isort fiberwatch/ tests/

type-check:  ## Run type checking
	mypy fiberwatch/

docs:  ## Build documentation
	cd docs && make html

docs-serve:  ## Serve documentation locally
	cd docs && python -m http.server 8000 --directory _build/html

build:  ## Build package
	python -m build

upload:  ## Upload to PyPI (requires credentials)
	python -m twine upload dist/*

dev: install-dev  ## Setup development environment
	pre-commit install

web:  ## Launch web interface
	fiberwatch web

analyze:  ## Example analysis command
	fiberwatch analyze data/UPC_dataset/normal_dirty.txt --baseline data/UPC_dataset/normal_clean.txt

visualize:  ## Example visualization command
	fiberwatch visualize data/UPC_dataset/normal_dirty.txt --baseline data/UPC_dataset/normal_clean.txt

setup-example:  ## Setup with example data
	@echo "Setting up FiberWatch with example data..."
	@$(MAKE) install-dev
	@echo "Running example analysis..."
	@$(MAKE) analyze
	@echo "Setup complete! Run 'make web' to launch the web interface."
