# Dataset Catalog

All datasets are converted to one JSONL schema (see `src/img2text/data/schema.py`):

```json
{"id": "fintabnet/000123", "image": "path/to/crop.png", "task": "table",
 "target": "<otsl ...>", "target_format": "otsl", "source": "fintabnet", "split": "train"}
```

`task` ∈ `table | chart | diagram | other`. `target_format` ∈ `otsl | html | chartcsv | mermaid | text`.

Research use is allowed for this project, so non-commercial licenses are included.
**"verify"** = license or exact Hub repo id not confirmed yet; check the dataset card before
downloading. Priority: **P0** = needed for the first training run, **P1** = add next, **P2** = optional / eval only.

## Tables

| Dataset | Size | Content | Target | License | Hub / source | Pri |
|---|---|---|---|---|---|---|
| **FinTabNet** | ~113k tables | S&P 500 annual reports — core financial domain | HTML / OTSL | CDLA-Permissive-1.0 | `ds4sd/FinTabNet_OTSL` (verify id) | P0 |
| **PubTabNet** | ~568k tables | PubMed Central article tables | HTML / OTSL | CDLA-Permissive-1.0 | `ds4sd/PubTabNet_OTSL` (verify id) | P0 |
| **SynthTabNet** | ~600k tables | Synthetic, 4 styles incl. finance-like | HTML / OTSL | CDLA-Permissive (verify) | `ds4sd/SynthTabNet_OTSL` (verify id) | P0 |
| **PubTables-1M** | ~948k tables / 460k pages | Scientific tables, structure + cell boxes | structure + boxes | CDLA-Permissive-2.0 | `bsmock/pubtables-1m` (verify) | P1 |
| **PubTables-v2** (Kensho, Dec 2025) | ~548k tables; full-page + multi-page | Tables in page context | structure + boxes | verify | arXiv 2512.10888 | P1 |
| TableVQA-Bench (FinTabNetQA) | small | QA over FinTabNet tables | eval | verify | arXiv 2404.19205 | P2 |

## Charts

| Dataset | Size | Content | Target | License | Hub / source | Pri |
|---|---|---|---|---|---|---|
| **PlotQA** | ~224k charts | Synthetic bar/line/dot plots with underlying data | data table | verify | github.com/NiteshMethani/PlotQA | P0 |
| **ChartQA** | ~21k charts | Real charts (Statista, Pew, OECD…) with CSVs for many | data table | GPL-3.0 | `HuggingFaceM4/ChartQA` / github vis-nlp/ChartQA | P0 (also eval) |
| **UniChart pretrain corpus** | ~600k charts | Charts paired with data tables | data table | verify | github vis-nlp/UniChart | P0 |
| **ChartGalaxy** (ICLR 2026) | 1.7M synthetic + 62k real | Infographic charts, 75 types | data / spec | verify | ChartGalaxy paper | P1 |
| **ECD** | 10k+ charts, 300k QA | 25 topics, 250+ chart-type combos | QA + code | verify | ECD / `ChartFoundation/ECDBench` (eval) | P1 |
| **Chart2Code-160k** | 160k pairs | Chart → matplotlib code, 27 types (GPT-4o synthesized) | code → data | verify | ChartCoder, arXiv 2501.06598 | P1 |
| DVQA | ~300k bar charts | Synthetic bar charts | QA / data | CC BY-NC 4.0 (verify) | kushalkafle.com | P2 |
| FigureQA | ~100k figures | Synthetic charts | QA | verify | Microsoft Research | P2 |
| **FinChart-Bench** | 1.2k charts, ~7k QA | Financial charts 2015–2024 | eval | verify | `Tizzzzy/FinChart-Bench` | P2 (eval) |
| **ChartArena** (2026) | – | Chart-parsing benchmark across formats | eval | verify | arXiv 2606.01348 | P2 (eval) |
| ChartMimic | – | Chart → code benchmark | eval | verify | ChartMimic paper | P2 (eval) |

## Diagrams

| Dataset | Size | Content | Target | License | Hub / source | Pri |
|---|---|---|---|---|---|---|
| **FlowLearn** | 10k simulated + 3.8k scientific | Flowcharts; simulated set ships Mermaid source | Mermaid | verify | arXiv 2407.05183 | P0 |
| **image-to-mermaid-v3** | ~15k (12k/1.5k/1.6k) | 26 Mermaid diagram types, 3 difficulty tiers | Mermaid | verify | `DangIT02/image-to-mermaid-v3` | P0 |
| **CoSyn-400K** (AllenAI) | ~400k | Code-rendered charts, tables, diagrams + source code | code / text | ODC-BY (verify) | `allenai/CoSyn-400K` | P1 |
| AI2D | ~5k | Science diagrams with annotations | text | verify | AllenAI | P2 |
| FlowchartQA generator | unlimited | Synthetic flowchart + Mermaid + QA | Mermaid | verify | github hacheyz/FlowchartQA | P1 |

## Mixed / other figures / replay

| Dataset | Size | Use | License | Hub / source | Pri |
|---|---|---|---|---|---|
| **DocLayNet** | ~80k pages | Picture/Table crops for router training; layout model | CDLA-Permissive-1.0 | `ds4sd/DocLayNet` | P0 |
| olmOCR-mix-0225 | large | General page-OCR replay (keeps text-reading skill) | ODC-BY (verify) | `allenai/olmOCR-mix-0225` | P1 |
| PixMo-Docs | large | Text-rich synthetic docs | ODC-BY (verify) | `allenai/pixmo-docs` | P1 |
| PixMo-Cap | large | Dense image captions → "other figure" descriptions | ODC-BY (verify) | `allenai/pixmo-cap` | P1 |
| SciCap | ~400k | Scientific figure captions | verify | SciCap | P2 |

## Own data (built in Phase 3 — biggest lever for the financial domain)

| Source | How | Ground truth | Pri |
|---|---|---|---|
| Synthetic financial charts | `synth/chart_gen.py`: matplotlib/plotly with realistic finance data (quarterly revenue, segment mix, margins, waterfalls, stacked bars, bar+line dual axis), corporate styling | exact (we generate the data) | P0 |
| Synthetic financial tables | `synth/table_gen.py`: HTML/CSS → image; parenthesized negatives, units/currency headers, multi-level headers, indentation, totals | exact | P0 |
| Synthetic business diagrams | `synth/diagram_gen.py`: Mermaid/Graphviz org charts, process flows, timelines, funnels, 2×2, SWOT | exact | P0 |
| Real financial figures | SEC EDGAR 10-K / annual reports + earnings decks → layout model crops → teacher pseudo-labels → self-consistency filter (charts: re-render and compare) | pseudo-labels | P1 |
| **Gold eval set** | ~300 hand-verified real crops: 100 tables, 120 charts, 50 diagrams, 30 other | human-verified | P0 (eval) |

## Starting training mix (Phase 4, Stage A)
~35% tables, 35% charts, 15% diagrams, 10% other, 5% general OCR replay. Stage B then
fine-tunes on synthetic financial data + filtered pseudo-labels.
