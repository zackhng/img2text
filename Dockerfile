# GPU image for img2text: data generation, Granite baseline runs, and training/eval work.
# Targets NVIDIA Blackwell (RTX 5060 Ti, sm_120), which needs PyTorch built for CUDA 12.8+;
# pyproject.toml pins torch to the cu128 index.
#
# Build:  docker build -t img2text .
# Run:    docker run --gpus all -v "$PWD/data_cache:/app/data_cache" img2text \
#             python scripts/run_baseline.py --model granite --task tables --batch 8
# data_cache/ (datasets, models, generated charts) is mounted, not baked in.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HOME=/app/data_cache/hf_home \
    HF_HUB_DISABLE_XET=1 \
    PATH="/app/.venv/bin:$PATH"

# Fonts for Chinese chart labels (matplotlib/plotly fall back to these on Linux), plus the
# shared libraries that headless plotly/kaleido export needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-noto-cjk \
        libnss3 libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libxkbcommon0 \
        libxcomposite1 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2t64 \
        libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

WORKDIR /app

# Dependencies first, so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev --python 3.12 \
        --extra synth --extra eval --extra export

COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
RUN uv sync --frozen --no-dev --python 3.12 --extra synth --extra eval --extra export

CMD ["python", "scripts/run_baseline.py", "--help"]
