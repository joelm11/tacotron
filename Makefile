.PHONY: help install data train serve format clean

# Use uv for all Python operations
UV := uv
PYTHON := $(UV) run python

SRC_DIR := src

help: ## Show this help
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and the project in the uv venv
	$(UV) sync
	$(UV) pip install -e .

data: ## Download and prepare the dataset
	$(PYTHON) $(SRC_DIR)/data.py

preproc: ## Preprocess input text
	$(PYTHON) $(SRC_DIR)/preproc.py

train: ## Train the model
	$(PYTHON) $(SRC_DIR)/train.py

serve: ## Serve the trained model
	$(PYTHON) $(SRC_DIR)/serve.py

format: ## Format and lint code with ruff
	$(UV) run ruff check --fix src/
	$(UV) run ruff format src/

clean: ## Remove cached files and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache build dist
