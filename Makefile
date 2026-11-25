# Mini-GPT task runner.
#
# Works with GNU Make on Linux/macOS, and on Windows from Git Bash.
# Windows does not ship `make`; install it with `choco install make` (or use
# `python tasks.py <target>`, which mirrors these targets without make).
#
# Recipes use relative paths only: this project's path may contain spaces or `&`.
#
# Examples:
#   make setup                                   # CUDA 12.6 build of PyTorch
#   make setup TORCH_INDEX=https://download.pytorch.org/whl/cpu   # CPU-only
#   make test ARGS="-k causal -v"
#   make train ARGS="-o training.max_iters=500"
#   make train ARGS="--resume"                   # continue checkpoints/last.pt
#   make train ARGS="--force"                    # start over, replacing checkpoints/

SHELL := bash
.DEFAULT_GOAL := help

PYTHON_VERSION ?= 3.12
VENV           ?= .venv
CONFIG         ?= configs/default.yaml
# PyTorch wheel index for `setup`. PyPI's Windows wheels are CPU-only, so the CUDA build
# comes from the PyTorch index. cu126 is the CUDA 12.x index that still gets new releases.
# On macOS use TORCH_INDEX=https://pypi.org/simple.
TORCH_INDEX    ?= https://download.pytorch.org/whl/cu126
ARGS           ?=

ifeq ($(OS),Windows_NT)
  PY := $(VENV)/Scripts/python.exe
else
  PY := $(VENV)/bin/python
endif

.PHONY: help setup test train lint clean

help: ## Show available targets
	@grep -E '^[a-z]+:.*## ' $(firstword $(MAKEFILE_LIST)) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  make %-7s %s\n", $$1, $$2}'

setup: ## Create .venv with uv, install PyTorch (TORCH_INDEX) and the package + dev tools
	uv venv $(VENV) --python $(PYTHON_VERSION) --allow-existing
	uv pip install --python $(PY) torch --index-url $(TORCH_INDEX)
	uv pip install --python $(PY) -e ".[dev]"
	$(PY) -c "import torch; print('torch', torch.__version__, '| CUDA build', torch.version.cuda, '| GPU available', torch.cuda.is_available())"

test: ## Run the pytest suite (ARGS passed to pytest)
	$(PY) -m pytest $(ARGS)

train: ## Train with CONFIG (ARGS passed to the trainer, e.g. -o key=value, --resume)
	$(PY) -m mini_gpt.train --config $(CONFIG) $(ARGS)

lint: ## Run ruff
	$(PY) -m ruff check src tests

clean: ## Remove caches and build artifacts (keeps .venv, data/ and checkpoints/)
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist src/*.egg-info
	find . -path ./$(VENV) -prune -o -name __pycache__ -type d -prune -exec rm -rf {} +
