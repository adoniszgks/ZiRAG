FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.13 python3.13-venv python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cu124
RUN uv pip install "numpy>=2.0" librosa soundfile transformers tqdm einops h5py
RUN uv pip install laion-clap --no-deps

COPY src/ src/
COPY main.py app.py ./

ENV PYTHONPATH=src
ENV GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 7860

VOLUME ["/app/cache", "/app/data"]

CMD ["uv", "run", "python", "main.py"]
