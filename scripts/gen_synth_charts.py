"""Generate synthetic financial charts with random variation in every style axis.

For each (library, chart type) pair, draws --per-cell charts with random bar/series counts,
palette, legend, orientation, value labels, grid, figure shape, font size, title and axis
visibility, background, and render resolution (dpi). For each chart, --aug-copies augmented
copies are also written (blur, JPEG, noise, rotation, crop, resampling, colour jitter).
Augmentation never changes the label.

Outputs:
  data_cache/synth/charts/<library>/*.png    rendered and augmented images
  data_cache/synth/charts.jsonl              one Record per image, with style + augmentation meta

Usage:
    uv run python scripts/gen_synth_charts.py --per-cell 50 --aug-copies 1 --seed 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

from img2text.data.schema import Record, write_jsonl
from img2text.synth.augment import augment
from img2text.synth.chart_gen import BACKENDS, CHART_TYPES, generate_spec, random_config, render

ROOT = Path(__file__).resolve().parents[1] / "data_cache" / "synth"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-cell", type=int, default=50,
                        help="charts per (library, chart type) pair")
    parser.add_argument("--aug-copies", type=int, default=1,
                        help="augmented copies written per rendered chart")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records: list[Record] = []
    counter = 0
    for backend in BACKENDS:
        for ctype in CHART_TYPES:
            for _ in range(args.per_cell):
                seed = args.seed * 1_000_000 + counter
                cfg = random_config(ctype, random.Random(seed * 31 + 7))
                spec = generate_spec(ctype, seed, n_categories=cfg["n_categories"],
                                     n_series=cfg["n_series"], style=cfg["style"])
                rel = f"charts/{backend}/{ctype}_{seed:07d}.png"
                render(spec, backend, ROOT / rel)
                label = spec.to_chartcsv()
                records.append(Record(
                    id=f"synth-chart/{backend}/{seed:07d}", image=rel, task="chart",
                    target=label, target_format="chartcsv", source=f"synth-{backend}",
                    split="train", meta={**spec.meta(), "augmented": False, "aug": {}},
                ))
                for k in range(args.aug_copies):
                    aug_rng = random.Random(seed * 97 + k)
                    with Image.open(ROOT / rel) as im:
                        result = augment(im, aug_rng)
                    aug_rel = f"charts/{backend}/{ctype}_{seed:07d}_aug{k}.png"
                    result.image.save(ROOT / aug_rel)
                    records.append(Record(
                        id=f"synth-chart/{backend}/{seed:07d}-aug{k}", image=aug_rel,
                        task="chart", target=label, target_format="chartcsv",
                        source=f"synth-{backend}", split="train",
                        meta={**spec.meta(), "augmented": True, "aug": result.params},
                    ))
                counter += 1
            print(f"rendered {args.per_cell} x {ctype} with {backend}", flush=True)

    n = write_jsonl(records, ROOT / "charts.jsonl")
    print(f"wrote {n} records -> {ROOT / 'charts.jsonl'}")


if __name__ == "__main__":
    main()
