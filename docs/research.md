# Research: Small, Fast Document-Image → Text Models

Goal: specialized models that convert images found in professional documents (financial
reports, earnings decks, corporate/strategy reports) into text — **tables, charts, diagrams,
and other figures**. Target: **under 1 second per image crop, CPU-first**, falling back to a
low-tier CUDA GPU only if a larger model is clearly better. Strategy: **start from a pretrained
model and fine-tune**, not train from scratch.

Hardware: 1× RTX 5060 Ti 16GB (Blackwell, sm_120) for training/eval.

---

## 1. Why not just use a general OCR VLM?

The strong 2025–2026 open document VLMs all parse pages to Markdown/HTML well, but they are
built for *quality on full pages*, not per-crop CPU latency:

- **PaddleOCR-VL** (~0.9B): NaViT dynamic-resolution encoder + ERNIE-0.3B LM. ~96% on
  OmniDocBench v1.6, 100+ languages, layout-then-recognition. ~2GB VRAM at FP16. Great quality
  ceiling, too heavy for sub-second CPU.
- **DeepSeek-OCR** (3B-MoE): "optical compression" — a 16× conv token compressor turns a
  1024×1024 page into ~256 vision tokens. On OmniDocBench it beats GOT-OCR2.0 using 100 vision
  tokens and MinerU2.0 using <800. Key lesson for us, not a deployable size.
- **dots.ocr / MinerU2.5** (~3B): decoupled layout → element recognition, high accuracy.
- **Granite-Docling-258M** (IBM): the exception — a genuinely small model, see below.

**The core latency fact.** On CPU, a ~135M decoder in int8 runs roughly 60–150 tokens/s. A
20-row financial table serialized as HTML can exceed 400 output tokens → several seconds. So
the dominant cost for us is **autoregressive decoding of the output**, not the vision encoder.
Everything in the design flows from minimizing output tokens, then vision tokens, then
decoder size.

---

## 2. Architectures to learn from

| Model | Size | Encoder / Decoder | What we take |
|---|---|---|---|
| **Granite-Docling-258M** (succeeds SmolDocling-256M) | 258M | SigLIP2-base encoder + pixel-shuffle connector + ~165M LM (based on SmolLM2) | **Primary base checkpoint.** Emits DocTags — a compact markup that losslessly converts to Markdown/HTML/JSON. Strongest small model on financial tables. |
| **SmolVLM-256M / 500M** | 256M / 500M | SigLIP encoder + pixel-shuffle (9–16× token reduction) + SmolLM2; image tiling for high-res | Proves a ~250M VLM works; recipe for cutting vision tokens. Apache-2.0. |
| **Florence-2-base** | 232M | DaViT vision encoder + compact transformer **encoder-decoder** | Encoder-decoder decodes very cheaply; strong CPU baseline. Task controlled by prompt tokens — matches our multi-task plan. MIT. |
| **Pix2Struct** | 282M / 1.3B | ViT with variable-resolution/aspect-ratio patches → text decoder | Variable-resolution patching; pretrained to parse screenshots to structured text. |
| **MatCha / DePlot** | 282M (Pix2Struct base) | same | *Chart de-rendering* pretraining = image → data table or plotting code. **DePlot = chart→linearized table.** Defines the chart task and the RMS/RNSS metric. |
| **UniChart** | ~200M | Donut-style (Swin encoder + BART decoder) | Chart-specialized small model; large chart→table pretraining corpus. |
| **PaddleOCR-VL** | 0.9B | NaViT + ERNIE-0.3B | **Teacher / quality ceiling**, GPU only. |
| **DeepSeek-OCR** | 3B-MoE | DeepEncoder (window+global attn, 16× conv compressor) | Vision-token count drives latency; conv compression trick. |
| **FastVLM (Apple)** | – | FastViTHD hybrid conv-transformer encoder | Few tokens at high res, 3.2× faster time-to-first-token. Fallback if the encoder becomes the bottleneck. |
| **TableFormer / FastTab** | small, non-generative | CNN + structure decoder predicting cell grid + boxes | **Structure-only** table path: predict table structure + cell boxes, copy text from the PDF layer. Avoids generating every cell → big tables go from seconds to ~100ms. |
| **Twin-T-1B** (CVPR 2026) | 1B | structure/detail-separating VLM | Upper bound for "small but good" on chart+table; beats much larger models. The target if 250M underperforms. |

### Speed levers, in priority order
1. **Small decoder** (~100–165M), int8 via ONNX Runtime / OpenVINO.
2. **Few output tokens** — compact target formats:
   - Tables: **OTSL** (the DocTags table vocabulary) uses ~half the tokens of HTML.
   - Charts: **CSV / linearized table** for the data, plus a short header (title, axis labels, legend).
   - Diagrams: terse **Mermaid**.
   - Convert these to final Markdown in Python *after* decoding — free, deterministic.
3. **Few vision tokens** — cap per-crop resolution (e.g. 512px long side → ~64 tokens after
   pixel shuffle); tile only for dense tables.
4. **PDF text layer** for born-digital PDFs — structure-only table head + copied cell text.
5. Later: speculative / multi-token decoding, FastViT-style encoder.

---

## 3. Recommended architecture

```
PDF/page ──► layout detector (reuse Docling/DocLayNet RT-DETR — NOT trained by us)
          └► crops ──► type router (tiny CNN ~5M, or the VLM's own first token)
                       ├─ table   ─► [born-digital] structure-only head + PDF text  ┐
                       │           └ [scanned]     VLM → OTSL → Markdown            │
                       ├─ chart   ─► VLM → title/axes/legend + CSV data → MD table   ├─► Markdown
                       ├─ diagram ─► VLM → Mermaid (+ fallback text summary)        │
                       └─ other   ─► VLM → 1–2 sentence description                 ┘
```

- **One shared multi-task VLM (~250M)**, task selected by a prompt token — not separate models
  per type. One dataset schema, one export path, easier to fit in 16GB, and tasks reinforce
  each other (chart↔table structure overlap).
- Fine-tune from **Granite-Docling-258M**; fallbacks **SmolVLM-256M** and **Florence-2-base**.
  Phase 2 benchmarks pick the final base. Full fine-tune fits 16GB with bf16 + gradient
  checkpointing + batch 4–8 and gradient accumulation.
- **Distillation**: a GPU teacher (PaddleOCR-VL / Qwen2.5-VL class) pseudo-labels real
  financial figures; the student learns from those plus synthetic ground truth.
- **Decision gate after Phase 2**: if the ~0.9B class is >10 points better on our gold set and
  the 250M model can't close the gap after fine-tuning, ship on a low-tier GPU instead of CPU.

---

## 4. Target output formats (what the model emits, before Markdown conversion)

- **Table** → OTSL token sequence (or HTML for the baseline), then `formats/otsl.py` →
  Markdown/HTML. Financial specifics: parenthesized negatives, unit/currency headers,
  multi-level headers, indentation, totals rows.
- **Chart** → small header block (`type`, `title`, `x_title`, `y_title`, `series`) + CSV data
  rows, then `formats/chartcsv.py` → a Markdown table + caption.
- **Diagram** → Mermaid source, validated by `formats/mermaid.py` (parse-check); on parse
  failure, fall back to a text summary.
- **Other figure** → 1–2 sentence description.

---

## 5. Open questions resolved in later phases
- Exact base checkpoint — Phase 2 zero-shot benchmark.
- Whether CPU hits sub-second for mid-size tables or they must take the PDF-text path — Phase 5.
- Router as separate CNN vs. first-token-of-VLM — Phase 2/3, decided on accuracy vs. latency.

---

## 6. Sources
- PaddleOCR-VL — arXiv 2510.14528; github.com/PaddlePaddle/PaddleOCR
- DeepSeek-OCR: Contexts Optical Compression — arXiv 2510.18234
- SmolDocling / Granite-Docling & DocTags — Docling project (docling-project.github.io/docling); IBM Granite
- SmolVLM: Redefining small and efficient multimodal models — arXiv 2504.05299
- Florence-2 — Microsoft; arXiv 2311.06242
- Pix2Struct — arXiv 2210.03347
- MatCha — arXiv 2212.09662 (ACL 2023); DePlot — arXiv 2212.10505
- UniChart — arXiv 2305.14761
- FastVLM: Efficient Vision Encoding — arXiv 2412.13303 (CVPR 2025)
- TableFormer — arXiv 2203.01017; FastTab — arXiv 2605.22422
- Twin-T & TwintVQA — CVPR 2026
- PubTables-1M — arXiv 2110.00061 (CVPR 2022)
