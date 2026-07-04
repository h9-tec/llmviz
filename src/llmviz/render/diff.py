"""Side-by-side architecture diff in the Raschka style: two towers + comparison table."""

from __future__ import annotations

from llmviz.render.block import draw_tower
from llmviz.render.factsheet import attribute_rows, stat_tiles
from llmviz.render.svg import SVG
from llmviz.render.tokens import accent_for, scope_id, stylesheet
from llmviz.spec import ArchSpec

COL_W = 1010


def render_diff(a: ArchSpec, b: ArchSpec, theme: str = "auto") -> str:
    accent = accent_for(a.model_type)
    g = SVG()
    width = 2 * COL_W

    bottoms = [draw_tower(g, a, accent)[0], draw_tower(g, b, accent, x_off=COL_W)[0]]
    y = max(bottoms) + 40
    g.line(60, y, width - 60, y, width=2)
    y += 56

    g.text(70, y, "Comparison", size=27, weight=700)
    g.text(width / 2 - 40, y, a.name, size=22, weight=700, anchor="end")
    g.text(width / 2 + 340, y, b.name, size=22, weight=700, anchor="end")
    y += 20

    rows_b = {label: v for label, v, _ in stat_tiles(b) + attribute_rows(b)}
    table = [
        (label, va, rows_b.get(label, "—")) for label, va, _ in stat_tiles(a) + attribute_rows(a)
    ]

    for label, va, vb in table:
        differs = va != vb
        g.line(70, y, width - 70, y, cls="leader")
        g.text(84, y + 30, label.title() if label.isupper() else label, size=20)
        g.text(width / 2 - 40, y + 30, va, size=20, anchor="end", weight=700 if differs else None)
        g.text(width / 2 + 340, y + 30, vb, size=20, anchor="end", weight=700 if differs else None)
        if differs:
            g.text(width - 90, y + 30, "≠", size=22, cls="accent", weight=800)
        y += 44

    return g.document(width, int(y + 40), stylesheet(accent), svg_id=scope_id(accent))
