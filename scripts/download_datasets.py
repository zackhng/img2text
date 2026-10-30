"""Download datasets listed in docs/datasets.md into data_cache/raw/<language>/<name>/.

Each dataset has a primary language:
  en     English-only sources (the default; always downloaded)
  zh     Chinese-only sources (opt-in: pass --lang zh)
  multi  sources whose images mix English and Chinese (opt-in: pass --lang multi)

Storage is per language folder. A sample from a mixed source is still tagged with every
language it contains (record meta `languages`), so filtering by language works across folders.

Resumable: snapshot_download skips files that are already complete.
Usage:
    uv run python scripts/download_datasets.py                  # English datasets
    uv run python scripts/download_datasets.py --lang zh        # add Chinese datasets
    uv run python scripts/download_datasets.py --lang zh multi --only PolyChartQA
    uv run python scripts/download_datasets.py --only ChartQA DocLayNet
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

# Windows cannot create the symlinks huggingface_hub uses by default; copy files instead.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "data_cache" / "raw"
OPT_IN_LANGS = ("zh", "multi")


@dataclass(frozen=True)
class Dataset:
    repo: str
    role: str
    size_gb: float
    lang: str  # en | zh | multi
    license: str = "verify"


DATASETS: dict[str, Dataset] = {
    "FinTabNet_OTSL": Dataset("ds4sd/FinTabNet_OTSL", "train+eval tables (S&P 500 annual reports)", 3.05, "en", "CDLA-Permissive-1.0"),
    "PubTabNet_OTSL": Dataset("ds4sd/PubTabNet_OTSL", "train tables (scientific)", 8.86, "en", "CDLA-Permissive-1.0"),
    "DocLayNet": Dataset("ds4sd/DocLayNet", "router / layout crops", 0.1, "en", "CDLA-Permissive-1.0"),
    "ChartQA": Dataset("HuggingFaceM4/ChartQA", "train+eval charts (real)", 0.96, "en", "GPL-3.0"),
    "image-to-mermaid-v3": Dataset("DangIT02/image-to-mermaid-v3", "train diagrams -> Mermaid", 1.21, "en"),
    "FinChart-Bench": Dataset("Tizzzzy/FinChart-Bench", "eval financial charts", 0.25, "en"),
    "ECDBench": Dataset("ChartFoundation/ECDBench", "eval charts", 0.66, "en"),
    "pixmo-cap": Dataset("allenai/pixmo-cap", "other-figure descriptions", 1.10, "en", "ODC-BY (verify)"),
    # Opt-in. Mike33/PolyChartQA is a third-party mirror of PolyChartQA (10 languages incl. en + zh,
    # arXiv 2507.11939). Confirm the official release and its license before using it for training.
    "PolyChartQA": Dataset("Mike33/PolyChartQA", "multilingual chart QA incl. Chinese (mirror)", 0.0, "multi", "other (verify)"),
}


def target_dir(name: str) -> Path:
    return ROOT / DATASETS[name].lang / name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="dataset names to download (default: all in the chosen languages)")
    parser.add_argument("--lang", nargs="*", choices=OPT_IN_LANGS, default=[],
                        help="opt-in languages to include on top of English")
    args = parser.parse_args()

    allowed = {"en", *args.lang}
    if args.only:
        unknown = set(args.only) - set(DATASETS)
        if unknown:
            raise SystemExit(f"unknown dataset(s): {sorted(unknown)}")
        names = args.only
    else:
        names = [n for n, d in DATASETS.items() if d.lang in allowed]

    skipped = [n for n in names if DATASETS[n].lang not in allowed]
    if skipped:
        raise SystemExit(f"{skipped} are opt-in; add --lang {' '.join(sorted({DATASETS[n].lang for n in skipped}))}")

    for name in names:
        ds = DATASETS[name]
        target = target_dir(name)
        print(f"==> [{ds.lang}] {name} ({ds.repo}, ~{ds.size_gb} GB, license: {ds.license}, {ds.role})", flush=True)
        snapshot_download(
            repo_id=ds.repo,
            repo_type="dataset",
            local_dir=target,
            max_workers=4,
        )
        print(f"    done -> {target}", flush=True)


if __name__ == "__main__":
    main()
