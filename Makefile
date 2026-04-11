# Makefile for Bat Face Recognition Project
# Provides convenient commands for development workflow

.PHONY: help install install-dev install-exact clean test lint format type-check security docs serve-docs build docker-build docker-run run train permutation-test cleanup-checkpoints

# Default target
help:
	@echo "Bat Face Recognition Project - Available Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  install-exact    Install exact version dependencies"
	@echo "  setup-hooks      Install pre-commit hooks"
	@echo ""
	@echo "Development Commands:"
	@echo "  clean           Clean up build artifacts and cache files"
	@echo "  test            Run all tests"
	@echo "  test-fast       Run fast tests only"
	@echo "  lint            Run all linting checks"
	@echo "  format          Format code with black and isort"
	@echo "  type-check      Run type checking with mypy"
	@echo "  security        Run security checks"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs            Build documentation"
	@echo "  serve-docs      Serve documentation locally"
	@echo ""
	@echo "Build Commands:"
	@echo "  build           Build package for distribution"
	@echo "  docker-build    Build Docker image"
	@echo "  docker-run      Run Docker container"
	@echo ""
	@echo "Experiment Commands:"
	@echo "  run             Interactive experiment launcher"
	@echo "  train           Train Siamese model (BAT_TYPE=r DATA_SOURCE=video BACKGROUND=random)"
	@echo "  permutation-test  Run permutation test (BAT_TYPE=r DATA_SOURCE=video)"
	@echo "  cleanup-checkpoints  Preview checkpoint cleanup (dry-run)"
	@echo ""
	@echo "Data Commands:"
	@echo "  augment-data    Run data augmentation script"
	@echo "  train-model     Train Siamese network model (legacy)"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

install-exact:
	pip install -r requirements-exact.txt

setup-hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/

# Testing
test:
	pytest -v --cov=app --cov-report=html --cov-report=term

test-fast:
	pytest -v -m "not slow" --cov=app

# Code quality
lint: format type-check security
	@echo "All linting checks completed!"

format:
	black app/ scripts/ --line-length=88
	isort app/ scripts/ --profile=black

type-check:
	mypy app/ --ignore-missing-imports

security:
	bandit -r app/ -x /tests/
	safety check

# Documentation
docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8000

# Building
build: clean
	python -m build

# Docker
docker-build:
	docker build -t bat-face-recognition:latest .

docker-run:
	docker run -p 5000:5000 -v $(PWD):/workspace bat-face-recognition:latest

# Experiment commands
run:
	python run.py

train:
	python -m app.siamese_training.train_siamese --bat-type $(BAT_TYPE) --data-source $(DATA_SOURCE) --background $(BACKGROUND)

permutation-test:
	python -m app.statistical_tests.run_permutation_test --bat-type $(BAT_TYPE) --data-source $(DATA_SOURCE)

cleanup-checkpoints:
	python scripts/cleanup_checkpoints.py --dry-run

# Legacy project-specific commands
augment-data:
	cd app && python ../legacy/face_annotation_eyes_nose/create_augmented_dataset.py

train-model:
	python -m app.siamese_training.train_siamese --bat-type r --data-source video

predict:
	python -m app.generate_predictions --help

# Environment setup
setup-env:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "source venv/bin/activate  # On Linux/macOS"
	@echo "venv\\Scripts\\activate     # On Windows"

# Quick start
quick-start: install-dev setup-hooks
	@echo "Development environment set up!"
	@echo "Run 'make augment-data' to start data augmentation"

# CI/CD simulation
ci: lint test
	@echo "CI pipeline completed successfully!"

# Development workflow
dev-check: format lint test-fast
	@echo "Development checks passed!"

# Project info
info:
	@echo "Project: Bat Face Recognition"
	@echo "Python: $(shell python --version)"
	@echo "Pip: $(shell pip --version)"
	@echo "Location: $(PWD)"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repository')"