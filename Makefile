.PHONY: help install test ui up

DEVICE ?= cuda

help:
	@echo "install  install dependencies (DEVICE=cpu|cuda, default cuda)"
	@echo "test     run tests"
	@echo "ui       start the UI without indexing"
	@echo "up       start the full application"

install:
	uv sync --extra $(DEVICE)
	uv pip install "numpy>=2.0" librosa soundfile transformers tqdm einops h5py torchlibrosa
	uv pip install braceexpand ftfy progressbar webdataset wget
	uv pip install laion-clap --no-deps

test:
	uv run python -m pytest tests -s

ui:
	uv run python main.py --ui-only

up:
	uv run python main.py
