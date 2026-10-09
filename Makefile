.PHONY: help install data train train-cpu train-gpu serve format clean

# Use uv for all Python operations
UV := uv
PYTHON := $(UV) run python

SRC_DIR := src/taco

# Ensure Keras uses the TensorFlow backend (required for GPU support)
export KERAS_BACKEND := tensorflow

# Ensure TensorFlow can find the CUDA/cuDNN libraries installed via pip
NVIDIA_LIB_DIRS := $(shell test -d .venv && find $(PWD)/.venv/lib/python*/site-packages/nvidia -type d -name lib 2>/dev/null | tr '\n' ':' || true)
export LD_LIBRARY_PATH := $(NVIDIA_LIB_DIRS)$(LD_LIBRARY_PATH)

help: ## Show this help
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and the project in the uv venv
	$(UV) sync
	$(UV) pip install -e .

data: ## Download and prepare the dataset
	$(PYTHON) $(SRC_DIR)/data.py

dataset: ## Download and prepare the dataset
	$(PYTHON) $(SRC_DIR)/dataset.py

preproc: ## Preprocess input text
	$(PYTHON) $(SRC_DIR)/pre_proc_data.py

train: ## Train the model (uses GPU if available, otherwise CPU)
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
