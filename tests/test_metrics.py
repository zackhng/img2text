import pytest

from img2text.eval.latency import measure_latency
from img2text.eval.metrics import score_chart, score_mermaid, teds

T1 = "<table><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr></table>"
T_STRUCT_SAME_TEXT_DIFF = "<table><tr><td>X</td><td>9</td></tr><tr><td>Y</td><td>8</td></tr></table>"
T_MORE_ROWS = "<table><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr><tr><td>C</td><td>3</td></tr></table>"
T_SPAN = '<table><tr><td colspan="2">A</td></tr><tr><td>B</td><td>2</td></tr></table>'


def test_teds_identical_is_one():
    assert teds(T1, T1) == pytest.approx(1.0)


def test_teds_structure_only_ignores_text():
    assert teds(T1, T_STRUCT_SAME_TEXT_DIFF, structure_only=True) == pytest.approx(1.0)
    assert teds(T1, T_STRUCT_SAME_TEXT_DIFF) < 1.0


def test_teds_penalises_missing_row_and_span():
    assert 0.0 < teds(T1, T_MORE_ROWS) < 1.0
    assert teds(T1, T_SPAN, structure_only=True) < 1.0


GT = """type: grouped_bar
title: Revenue
x_axis: Period
y_axis: Revenue ($m)
unit: $m
data:
Period,Cloud,Services
Q1,100,50
Q2,200,60"""


def test_chart_perfect_match():
    s = score_chart(GT, GT)
    assert s["type_ok"] == 1 and s["cell_acc"] == 1.0


def test_chart_tolerance_and_errors():
    pred = GT.replace("200", "203").replace("50", "90")  # 203 within 5%, 90 is not
    s = score_chart(pred, GT)
    assert s["cells_correct"] == 3 and s["cells_total"] == 4


def test_chart_unparseable_prediction_scores_zero():
    s = score_chart("I cannot read this chart.", GT)
    assert s["cell_acc"] == 0.0 and s["type_ok"] == 0


MM_GT = "flowchart TD\n  A[Start] --> B[Check]\n  B --> C[End]"


def test_mermaid_perfect():
    s = score_mermaid(MM_GT, MM_GT)
    assert s["parse_ok"] == 1 and s["node_f1"] == 1.0 and s["edge_f1"] == 1.0


def test_mermaid_rejects_non_diagram():
    s = score_mermaid("Here is a flowchart of the process.", MM_GT)
    assert s["parse_ok"] == 0


def test_measure_latency_returns_percentiles():
    r = measure_latency(lambda x: sum(range(x)), [1000] * 10, warmup=1)
    assert r["n"] == 10 and r["p50_ms"] >= 0 and r["p95_ms"] >= r["p50_ms"]
