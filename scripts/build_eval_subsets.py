"""Build small evaluation subsets for the baseline runs.

Writes data_cache/eval/<task>/images/*.png and data_cache/eval/<task>.jsonl with fields
{id, image, target} (target = HTML for tables, chartcsv for charts, Mermaid for diagrams).

- tables:   FinTabNet test split (real annual-report tables)
- diagrams: image-to-mermaid-v3 test split
- charts:   synthetic charts from seeds that are NOT used for training

Usage:
    uv run python scripts/build_eval_subsets.py --n 60
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from img2text.synth.chart_gen import BACKENDS, CHART_TYPES, generate_spec, random_config, render

ROOT = Path(__file__).resolve().parents[1] / "data_cache"
EVAL = ROOT / "eval"
TRAIN_SEED_LIMIT = 9_000_000  # training charts use seeds < 1,000,000; eval uses 9,000,000+


def _write(task: str, items: list[dict]) -> None:
    out_dir = EVAL / task / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (EVAL / f"{task}.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{task}: {len(items)} items")


def build_tables(n: int) -> None:
    shards = sorted((ROOT / "raw" / "en" / "FinTabNet_OTSL" / "data").glob("test-*.parquet"))
    if not shards:
        shards = sorted((ROOT / "raw" / "en" / "FinTabNet_OTSL").rglob("test*.parquet"))
    items: list[dict] = []
    out_dir = EVAL / "tables" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        table = pq.read_table(shard, columns=["imgid", "html", "image"])
        for row in table.to_pylist():
            if len(items) >= n:
                break
            img_bytes = row["image"]["bytes"]
            rel = f"tables/images/{row['imgid']}.png"
            Image.open(io.BytesIO(img_bytes)).convert("RGB").save(EVAL / rel)
            html = "".join(row["html"])
            items.append({"id": f"fintabnet/{row['imgid']}", "image": rel, "target": html})
        if len(items) >= n:
            break
    _write("tables", items)


def build_diagrams(n: int) -> None:
    base = ROOT / "raw" / "en" / "image-to-mermaid-v3"
    records = json.loads((base / "test.json").read_text(encoding="utf-8"))
    rng = random.Random(0)
    rng.shuffle(records)
    items: list[dict] = []
    img_re = re.compile(r"<img>(.*?)</img>")
    out_dir = EVAL / "diagrams" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        if len(items) >= n:
            break
        user, assistant = rec["conversations"][0]["value"], rec["conversations"][1]["value"]
        m = img_re.search(user)
        if not m:
            continue
        src = base / m.group(1)
        if not src.exists():
            continue
        rel = f"diagrams/images/{rec['id']}.png"
        Image.open(src).convert("RGB").save(EVAL / rel)
        items.append({"id": rec["id"], "image": rel, "target": assistant.strip()})
    _write("diagrams", items)


def build_charts(n: int) -> None:
    items: list[dict] = []
    for i in range(n):
        seed = TRAIN_SEED_LIMIT + i
        ctype = CHART_TYPES[i % len(CHART_TYPES)]
        backend = BACKENDS[i % len(BACKENDS)]
        cfg = random_config(ctype, random.Random(seed * 31 + 7))
        spec = generate_spec(ctype, seed, n_categories=cfg["n_categories"],
                             n_series=cfg["n_series"], style=cfg["style"])
        rel = f"charts/images/{ctype}_{backend}_{seed:08d}.png"
        render(spec, backend, EVAL / rel)
        items.append({"id": f"synth-eval/{backend}/{seed:08d}", "image": rel,
                      "target": spec.to_chartcsv()})
    _write("charts", items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    args = parser.parse_args()
    build_tables(args.n)
    build_diagrams(args.n)
    build_charts(args.n)


if __name__ == "__main__":
    main()
