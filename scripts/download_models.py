"""Download the baseline model checkpoints into data_cache/models/ (resumable).

Usage:
    uv run python scripts/download_models.py                 # all baselines
    uv run python scripts/download_models.py --only granite-docling-258M
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "data_cache" / "models"

MODELS = {
    "granite-docling-258M": "ibm-granite/granite-docling-258M",
    "SmolVLM-256M-Instruct": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "Florence-2-base": "microsoft/Florence-2-base",
    "deplot": "google/deplot",
    "PaddleOCR-VL": "PaddlePaddle/PaddleOCR-VL",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    names = args.only or list(MODELS)
    for name in names:
        print(f"==> {name} ({MODELS[name]})", flush=True)
        snapshot_download(repo_id=MODELS[name], local_dir=ROOT / name, max_workers=4)
        print(f"done -> {ROOT / name}", flush=True)


if __name__ == "__main__":
    main()
