"""1200x630 social card: headline stats left, the decoder block scaled on the right."""

from __future__ import annotations

from llmviz.render.block import _display_name, draw_tower
from llmviz.render.svg import SVG, fmt_count, fmt_params
from llmviz.render.tokens import accent_for, scope_id, stylesheet
from llmviz.spec import ArchSpec

W, H = 1200, 630


def render_card(s: ArchSpec, accent: str | None = None) -> str:
    accent = accent or accent_for(s.model_type)
    g = SVG()

    # right half: the full tower, scaled to fit the card height
    tower = SVG()
    tower_bottom, _ = draw_tower(tower, s, accent, title=False)
    scale = (H - 40) / tower_bottom
    g.raw(f'<g transform="translate(560, 20) scale({scale:.3f})">')
    g.parts.extend(tower.parts)
    g.raw("</g>")

    # left half: name + headline numbers
    g.text(56, 118, _display_name(s.name), size=52, cls="accent", weight=800)
    a = s.attention
    ffn = (
        f"MoE · {s.moe.num_experts} experts · top-{s.moe.experts_per_tok}"
        if s.moe
        else f"dense FFN · d_ff {s.intermediate_size:,}"
    )
    stats = [
        ("Parameters", fmt_params(s.total_params) if s.total_params else "—"),
        (
            "Active / token",
            fmt_params(s.active_params) if s.is_moe and s.active_params else "= total",
        ),
        ("Layers", str(s.num_layers)),
        ("Attention", f"{a.kind.value} · {a.num_heads} heads / {a.num_kv_heads} KV"),
        ("Feed-forward", ffn),
        ("Context", fmt_count(s.context_length) if s.context_length else "—"),
    ]
    y = 186
    for label, value in stats:
        g.text(56, y, label.upper(), size=17)
        g.text(56, y + 34, value, size=29, weight=700)
        y += 70
    g.text(56, H - 18, "rendered by llmviz from config.json", size=15)

    return g.document(W, H, stylesheet(accent), svg_id=scope_id(accent))
