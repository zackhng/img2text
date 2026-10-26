"""Download the P0/P1 datasets listed in docs/datasets.md into data_cache/raw/.

Resumable: snapshot_download skips files that are already complete.
Usage:
    uv run python scripts/download_datasets.py            # all entries
    uv run python scripts/download_datasets.py --only ChartQA DocLayNet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Windows cannot create the symlinks huggingface_hub uses by default; copy files instead.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "data_cache" / "raw"

# local name -> (hub repo, role, approx size in GB)
DATASETS: dict[str, tuple[str, str, float]] = {
    "FinTabNet_OTSL": ("ds4sd/FinTabNet_OTSL", "train+eval tables (S&P 500 annual reports)", 3.05),
    "PubTabNet_OTSL": ("ds4sd/PubTabNet_OTSL", "train tables (scientific)", 8.86),
    "DocLayNet": ("ds4sd/DocLayNet", "router / layout crops", 0.1),
    "ChartQA": ("HuggingFaceM4/ChartQA", "train+eval charts (real)", 0.96),
    "image-to-mermaid-v3": ("DangIT02/image-to-mermaid-v3", "train diagrams -> Mermaid", 1.21),
    "FinChart-Bench": ("Tizzzzy/FinChart-Bench", "eval financial charts", 0.25),
    "ECDBench": ("ChartFoundation/ECDBench", "eval charts", 0.66),
    "pixmo-cap": ("allenai/pixmo-cap", "other-figure descriptions", 1.10),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="local names to download")
    args = parser.parse_args()

    names = args.only or list(DATASETS)
    unknown = set(names) - set(DATASETS)
    if unknown:
        raise SystemExit(f"unknown dataset(s): {sorted(unknown)}")

    for name in names:
        repo, role, size = DATASETS[name]
        target = ROOT / name
        print(f"==> {name} ({repo}, ~{size} GB, {role})", flush=True)
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=target,
            max_workers=4,
        )
        print(f"    done -> {target}", flush=True)


if __name__ == "__main__":
    main()
