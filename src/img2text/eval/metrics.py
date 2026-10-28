"""Evaluation metrics for the four output types: tables, charts, diagrams.

- teds(): Tree-Edit-Distance Similarity between two HTML tables (PubTabNet-style TEDS).
  structure_only=True ignores cell text (TEDS-S).
- score_chart(): parses the chartcsv target format and scores chart type and data cells.
  A cell is correct when its value is within rel_tol of the ground truth.
- score_mermaid(): checks the diagram keyword and compares node and edge sets (approximate,
  regex-based; not a full Mermaid parser).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from apted import APTED, Config
from apted.helpers import Tree
from lxml import html as lxml_html

# ---------------------------------------------------------------- TEDS (tables)

_BRACE_RE = re.compile(r"[{}]")


def _clean_label(text: str) -> str:
    return _BRACE_RE.sub("", " ".join(text.split()))


def _html_to_bracket(table_html: str, structure_only: bool) -> str:
    """Convert an HTML table to APTED bracket notation: {table{tr{td:text}}...}."""
    root = lxml_html.fromstring(table_html)
    tables = root.xpath("//table") or [root]
    table = tables[0]

    def walk(node) -> str:
        tag = node.tag if isinstance(node.tag, str) else None
        if tag is None:
            return ""
        if tag in ("td", "th"):
            label = tag
            if not structure_only:
                text = _clean_label("".join(node.itertext()))
                label = f"{tag}:{text}"
            # cells keep their span attributes so structure differences are penalised
            span = f"|r{node.get('rowspan', '1')}c{node.get('colspan', '1')}"
            return "{" + label + span + "}"
        children = "".join(walk(c) for c in node)
        return "{" + tag + children + "}"

    return walk(table)


def _count_nodes(bracket: str) -> int:
    return bracket.count("{")


def teds(pred_html: str, gt_html: str, structure_only: bool = False) -> float:
    """Return similarity in [0, 1]; 1.0 means identical trees."""
    pred = _html_to_bracket(pred_html, structure_only)
    gold = _html_to_bracket(gt_html, structure_only)
    dist = APTED(Tree.from_text(pred), Tree.from_text(gold), Config()).compute_edit_distance()
    denom = max(_count_nodes(pred), _count_nodes(gold))
    return max(0.0, 1.0 - dist / denom) if denom else 1.0


# ---------------------------------------------------------------- charts (chartcsv)

@dataclass
class ParsedChart:
    chart_type: str
    columns: list[str]  # series names, in order
    rows: dict[str, dict[str, float]]  # category -> series -> value


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def parse_chartcsv(text: str) -> ParsedChart | None:
    """Parse the header block and data table. Returns None if the text is not chartcsv."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if "data:" not in lines:
        return None
    chart_type = ""
    if lines[0].lower().startswith("type:"):
        chart_type = _norm(lines[0].split(":", 1)[1])
    idx = lines.index("data:")
    body = lines[idx + 1:]
    if not body:
        return None
    header = [_norm(c) for c in body[0].split(",")]
    series = header[1:]
    rows: dict[str, dict[str, float]] = {}
    for line in body[1:]:
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(header):
            continue
        cat = _norm(cells[0])
        vals: dict[str, float] = {}
        for name, raw in zip(series, cells[1:]):
            try:
                vals[name] = float(raw.replace("%", "").replace("$", "").replace(",", ""))
            except ValueError:
                vals[name] = math.nan
        rows[cat] = vals
    return ParsedChart(chart_type=chart_type, columns=series, rows=rows)


def score_chart(pred_text: str, gt_text: str, rel_tol: float = 0.05) -> dict:
    """Score a predicted chartcsv against ground truth.

    Returns type_ok (0/1), cell_acc (fraction of ground-truth cells matched), and the counts.
    """
    gt = parse_chartcsv(gt_text)
    if gt is None:
        raise ValueError("ground truth is not chartcsv")
    pred = parse_chartcsv(pred_text)
    if pred is None:
        return {"type_ok": 0, "cell_acc": 0.0, "cells_correct": 0, "cells_total": _cell_count(gt)}

    type_ok = int(pred.chart_type == gt.chart_type)
    correct = 0
    total = 0
    for cat, gt_vals in gt.rows.items():
        pred_vals = pred.rows.get(cat, {})
        for series, g in gt_vals.items():
            total += 1
            p = pred_vals.get(series, math.nan)
            if math.isnan(p) or math.isnan(g):
                continue
            if abs(p - g) <= rel_tol * max(abs(g), 1e-9):
                correct += 1
    return {
        "type_ok": type_ok,
        "cell_acc": correct / total if total else 1.0,
        "cells_correct": correct,
        "cells_total": total,
    }


def _cell_count(gt: ParsedChart) -> int:
    return sum(len(v) for v in gt.rows.values())


# ---------------------------------------------------------------- diagrams (Mermaid)

_KEYWORDS = ("graph", "flowchart", "sequencediagram", "classdiagram", "statediagram",
             "erdiagram", "journey", "gantt", "pie", "mindmap", "timeline")
_ARROW_RE = re.compile(r"\s*(-\.->|==>|-->|---|-\.-|==)\s*(\|[^|]*\|)?\s*")
_NODE_ID_RE = re.compile(r"^\s*([A-Za-z0-9_]+)")


def _mermaid_nodes_edges(code: str) -> tuple[set[str], set[tuple[str, str]]]:
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    for line in code.splitlines()[1:]:
        if not line.strip() or line.strip().startswith("%%"):
            continue
        parts = _ARROW_RE.split(line)
        ids = []
        for part in parts[::3]:  # alternate: text, arrow, text, ...
            m = _NODE_ID_RE.match(part or "")
            if m:
                ids.append(m.group(1))
                nodes.add(m.group(1))
        for a, b in zip(ids, ids[1:]):
            edges.add((a, b))
    return nodes, edges


def _f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def score_mermaid(pred_code: str, gt_code: str) -> dict:
    """Return parse_ok (diagram keyword present), node_f1 and edge_f1 against ground truth."""
    first = next((ln.strip() for ln in pred_code.splitlines() if ln.strip()), "")
    parse_ok = int(first.lower().replace(" ", "").startswith(_KEYWORDS))
    p_nodes, p_edges = _mermaid_nodes_edges(pred_code)
    g_nodes, g_edges = _mermaid_nodes_edges(gt_code)
    return {
        "parse_ok": parse_ok,
        "node_f1": _f1(p_nodes, g_nodes),
        "edge_f1": _f1(p_edges, g_edges),
    }
