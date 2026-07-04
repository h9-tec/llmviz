"""Poster: a print-ready grid of architecture towers, one sheet."""

from __future__ import annotations

from llmviz.render.block import draw_tower
from llmviz.render.svg import SVG
from llmviz.render.tokens import accent_for, scope_id, stylesheet

CELL_W, SCALE = 560, 0.52


def render_poster(specs: list, title: str = "LLM Architecture Gallery", cols: int = 4) -> str:
    g = SVG()
    cols = min(cols, len(specs))
    header = 150

    # render each tower unscaled to learn its height, then place scaled
    cells = []
    for spec in specs:
        t = SVG()
        accent = accent_for(spec.model_type)
        bottom, _ = draw_tower(t, spec, accent)
        cells.append((t, bottom))
    row_h = max(b for _, b in cells) * SCALE + 30

    for i, ((t, _), spec) in enumerate(zip(cells, specs, strict=True)):
        x = (i % cols) * CELL_W + 20
        y = header + (i // cols) * row_h
        # carry the per-accent scope id so each tower keeps its own palette
        g.raw(
            f'<g id="{scope_id(accent_for(spec.model_type))}" '
            f'transform="translate({x},{y}) scale({SCALE})">'
        )
        g.parts.extend(t.parts)
        g.raw("</g>")

    width = cols * CELL_W + 40
    rows = (len(specs) + cols - 1) // cols
    height = int(header + rows * row_h + 20)
    g.text(width / 2, 78, title, size=54, cls="accent", anchor="middle", weight=800)
    g.text(
        width / 2,
        116,
        f"{len(specs)} architectures · rendered by llmviz from config.json",
        size=19,
        anchor="middle",
    )
    # one accent for the chrome; each tower carries its own scoped palette
    chrome = accent_for(specs[0].model_type)
    # chrome first so per-cell scoped rules (later in the cascade) win inside their cells
    css = stylesheet(chrome) + "".join(
        stylesheet(accent_for(s.model_type)) for s in {s.model_type: s for s in specs}.values()
    )
    return g.document(width, height, css, svg_id=scope_id(chrome))
