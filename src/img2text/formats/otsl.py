"""Convert Granite-Docling OTSL table output to HTML.

OTSL tokens: fcel = cell with content, ecel = empty cell, lcel = merged with the cell on the left,
ucel = merged with the cell above, xcel = merged in both directions, nl = end of row.
<loc_N> location tags carry no table content and are dropped.
"""

from __future__ import annotations

import html
import re

_TOKEN_RE = re.compile(r"(<(?:fcel|ecel|lcel|ucel|xcel|nl)>)")
_LOC_RE = re.compile(r"<loc_\d+>")
_CELL_TOKENS = {"<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>"}


def otsl_to_html(otsl: str) -> str:
    body = otsl.replace("<otsl>", "").replace("</otsl>", "")
    body = _LOC_RE.sub("", body).replace("<|end_of_text|>", "")
    parts = [p for p in _TOKEN_RE.split(body) if p != ""]

    # Build a grid of (kind, text) entries, one list per row.
    rows: list[list[dict]] = [[]]
    for part in parts:
        if part == "<nl>":
            rows.append([])
        elif part in _CELL_TOKENS:
            rows[-1].append({"kind": part, "text": ""})
        else:
            if rows[-1]:
                rows[-1][-1]["text"] += part
    if not rows[-1]:
        rows.pop()

    # Resolve spans: lcel extends the cell to its left, ucel the cell above.
    spanned = [[False] * len(r) for r in rows]
    colspan = [[1] * len(r) for r in rows]
    rowspan = [[1] * len(r) for r in rows]
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if cell["kind"] == "<lcel>" and j > 0:
                spanned[i][j] = True
                colspan[i][j - 1] += 1
            elif cell["kind"] == "<ucel>" and i > 0 and j < len(rows[i - 1]):
                spanned[i][j] = True
                rowspan[i - 1][j] += 1
            elif cell["kind"] == "<xcel>":
                spanned[i][j] = True

    out = ["<table>"]
    for i, row in enumerate(rows):
        out.append("<tr>")
        for j, cell in enumerate(row):
            if cell["kind"] in ("<ecel>",) or spanned[i][j]:
                if cell["kind"] == "<ecel>":
                    out.append("<td></td>")
                continue
            attrs = ""
            if colspan[i][j] > 1:
                attrs += f' colspan="{colspan[i][j]}"'
            if rowspan[i][j] > 1:
                attrs += f' rowspan="{rowspan[i][j]}"'
            out.append(f"<td{attrs}>{html.escape(cell['text'].strip())}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)
