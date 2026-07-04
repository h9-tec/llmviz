"""Evolution strip: N generations of a family side by side, changes flagged under each."""

from __future__ import annotations

from llmviz.render.block import draw_tower
from llmviz.render.factsheet import attribute_rows, stat_tiles
from llmviz.render.svg import SVG
from llmviz.render.tokens import accent_for, scope_id, stylesheet
from llmviz.spec import ArchSpec

COL_W = 1010


def render_lineage(specs: list[ArchSpec]) -> str:
    accent = accent_for(specs[0].model_type)
    g = SVG()
    width = COL_W * len(specs)

    bottoms = [draw_tower(g, s, accent, x_off=i * COL_W)[0] for i, s in enumerate(specs)]
    y0 = max(bottoms) + 46

    # under each generation after the first: what changed vs its predecessor
    max_rows = 0
    for i in range(1, len(specs)):
        prev = {k: v for k, v, _ in stat_tiles(specs[i - 1]) + attribute_rows(specs[i - 1])}
        cur = stat_tiles(specs[i]) + attribute_rows(specs[i])
        x = i * COL_W + 90
        g.text(x, y0, "What changed", size=24, weight=700)
        y = y0 + 40
        rows = 0
        for label, value, _ in cur:
            if prev.get(label) != value:
                nice = label.title() if label.isupper() else label
                g.rich_text(
                    x,
                    y,
                    [
                        (f"{nice}:  ", ""),
                        (str(prev.get(label, "—")), ""),
                        ("  →  ", ""),
                        (value, "accent"),
                    ],
                    size=20,
                    weight=600,
                )
                y += 33
                rows += 1
        if rows == 0:
            g.text(x, y, "architecturally identical", size=20)
            rows = 1
        max_rows = max(max_rows, rows)

    height = int(y0 + 40 + max_rows * 33 + 40) if len(specs) > 1 else int(y0)
    return g.document(width, height, stylesheet(accent), svg_id=scope_id(accent))
