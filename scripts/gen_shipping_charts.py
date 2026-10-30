"""Generate oil-shipping charts in English, Chinese, and mixed English + Chinese.

For each (language, backend, chart type) combination, draws --per-cell charts. Crude-price
values are real EIA Brent annual figures; tanker earnings, cargo volumes and cost drivers are
synthetic within realistic ranges. Each chart is rendered by matplotlib, seaborn, and plotly.

Outputs, one folder per language (mixed English + Chinese goes to `multi/`):
  data_cache/synth/<lang>/shipping/<backend>/*.png   rendered images (and augmented copies)
  data_cache/synth/<lang>/shipping.jsonl             one Record per image

  <lang> is en | zh | multi. Train and eval use different seed ranges and separate jsonl files
  (shipping_train.jsonl, shipping_eval.jsonl). Eval charts are never augmented.

Usage:
    uv run python scripts/gen_shipping_charts.py --per-cell 20 --aug-copies 1
    uv run python scripts/gen_shipping_charts.py --split eval --per-cell 5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

from img2text.data.schema import Record, write_jsonl
from img2text.synth.augment import augment
from img2text.synth.chart_gen import BACKENDS, LANGUAGES, random_config, render
from img2text.synth.shipping import generate_shipping_spec

ROOT = Path(__file__).resolve().parents[1] / "data_cache" / "synth"
SHIP_TYPES = ("line", "grouped_bar", "stacked_bar", "combo", "donut", "waterfall")
FOLDER = {"en": "en", "zh": "zh", "mixed": "multi"}  # mixed English + Chinese -> multi/
EVAL_SEED_BASE = 9_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--per-cell", type=int, default=20,
                        help="charts per (language, backend, chart type)")
    parser.add_argument("--aug-copies", type=int, default=1,
                        help="augmented copies per rendered train chart (eval is never augmented)")
    parser.add_argument("--langs", nargs="*", choices=sorted(LANGUAGES), default=["en", "zh", "mixed"])
    parser.add_argument("--seed", type=int, default=1, help="block of seeds: seed * 1,000,000")
    args = parser.parse_args()

    base = (EVAL_SEED_BASE if args.split == "eval" else args.seed * 1_000_000)
    aug_copies = 0 if args.split == "eval" else args.aug_copies

    for lang in args.langs:
        folder = FOLDER[lang]
        records: list[Record] = []
        counter = 0
        for backend in BACKENDS:
            for ctype in SHIP_TYPES:
                for _ in range(args.per_cell):
                    seed = base + counter
                    cfg = random_config(ctype, random.Random(seed * 31 + 7))
                    spec = generate_shipping_spec(ctype, seed, lang=lang,
                                                  n_categories=cfg["n_categories"],
                                                  n_series=cfg["n_series"], style=cfg["style"])
                    rel = f"{folder}/shipping/{backend}/{ctype}_{seed:07d}.png"
                    render(spec, backend, ROOT / rel)
                    label = spec.to_chartcsv()
                    records.append(Record(
                        id=f"ship-chart/{lang}/{backend}/{seed:07d}", image=rel, task="chart",
                        target=label, target_format="chartcsv", source=f"synth-shipping-{backend}",
                        split=args.split, meta={**spec.meta(), "augmented": False, "aug": {}},
                    ))
                    for k in range(aug_copies):
                        with Image.open(ROOT / rel) as im:
                            result = augment(im, random.Random(seed * 97 + k))
                        aug_rel = f"{folder}/shipping/{backend}/{ctype}_{seed:07d}_aug{k}.png"
                        result.image.save(ROOT / aug_rel)
                        records.append(Record(
                            id=f"ship-chart/{lang}/{backend}/{seed:07d}-aug{k}", image=aug_rel,
                            task="chart", target=label, target_format="chartcsv",
                            source=f"synth-shipping-{backend}", split=args.split,
                            meta={**spec.meta(), "augmented": True, "aug": result.params},
                        ))
                    counter += 1
        out = ROOT / folder / f"shipping_{args.split}.jsonl"
        n = write_jsonl(records, out)
        print(f"[{lang} -> {folder}/] wrote {n} records -> {out}", flush=True)


if __name__ == "__main__":
    main()
