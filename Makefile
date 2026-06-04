.PHONY: install test run

install:
	uv sync
	uv pip install torch --index-url https://download.pytorch.org/whl/cu124
	uv pip install "numpy>=2.0" librosa soundfile transformers tqdm einops h5py torchlibrosa
	uv pip install braceexpand ftfy progressbar webdataset wget
	uv pip install laion-clap --no-deps

test:
	uv run pytest tests

run:
	uv run python main.py
