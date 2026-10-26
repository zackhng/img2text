# img2text

Small, fast models (~250M params, CPU-first, target <1s per crop) that turn images in
professional documents — tables, charts, diagrams, other figures — into Markdown.

Approach: fine-tune a pretrained small document VLM (Granite-Docling-258M; fallbacks
SmolVLM-256M, Florence-2-base) on financial-domain + public data with compact output
formats (OTSL tables, CSV chart data, Mermaid diagrams).

- Architecture research: [docs/research.md](docs/research.md)
- Dataset catalog: [docs/datasets.md](docs/datasets.md)

## Setup
```bash
uv sync --extra dev          # installs PyTorch built for CUDA 12.8 (needed for RTX 50xx)
uv run pytest
```

## Layout
```
docs/            research + dataset catalog
configs/         train / eval YAMLs
src/img2text/
  data/          downloaders + converters -> unified JSONL (schema.py)
  synth/         synthetic chart / table / diagram generators
  formats/       OTSL<->Markdown, chart CSV<->Markdown, Mermaid validation
  model/         base VLM loading, task prompts
  train/         fine-tuning
  eval/          TEDS, chart RMS, graph F1, latency
  export/        ONNX + int8
scripts/
tests/
```

## Roadmap
1. Research docs + scaffold  ← done
2. Baselines + benchmark harness (zero-shot Granite-Docling, SmolVLM, Florence-2, DePlot, PaddleOCR-VL)
3. Data pipeline (downloaders, synthetic generators, EDGAR + teacher pseudo-labels)
4. Fine-tuning on the 16GB GPU
5. CPU export (ONNX int8) + structure-only table path
