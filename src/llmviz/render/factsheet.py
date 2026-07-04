"""Fact-sheet panel: stat tiles + attribute rows, drawn to the right of the diagram."""

from __future__ import annotations

from llmviz.render.svg import SVG, fmt_count, fmt_params, fmt_theta
from llmviz.spec import ArchSpec


def _kv_human(n: int) -> str:
    return f"{n / 1024:.1f} KB" if n >= 1024 else f"{n} B"


def stat_tiles(s: ArchSpec) -> list[tuple[str, str, str | None]]:
    return [
        ("TOTAL PARAMS", fmt_params(s.total_params), None),
        ("ACTIVE / TOKEN", fmt_params(s.active_params) if s.is_moe else "= total", None),
        ("LAYERS", str(s.num_layers), None),
        ("CONTEXT", fmt_count(s.context_length), None),
        ("VOCAB", f"{s.vocab_size:,}" if s.vocab_size else "—", None),
        ("KV CACHE / TOKEN", _kv_human(s.kv_cache_per_token_bytes), None),
    ]


def attribute_rows(s: ArchSpec) -> list[tuple[str, str, str | None]]:
    a = s.attention
    return [
        ("Attention", f"{a.kind.value} · {a.num_heads} heads / {a.num_kv_heads} KV", "l-attn"),
        (
            "Feed-forward",
            (
                f"MoE · {s.moe.num_experts} experts, top-{s.moe.experts_per_tok}"
                if s.moe
                else f"dense · d_ff {fmt_count(s.intermediate_size)}"
            ),
            "l-moe" if s.moe else "l-ffn",
        ),
        (
            "Positional",
            {"rope": "RoPE", "learned": "learned", "nope": "NoPE"}[s.positional]
            + (f" θ={fmt_theta(s.rope_theta)}" if s.rope_theta else ""),
            None,
        ),
        (
            "Normalization",
            ("RMSNorm" if s.norm_type == "rmsnorm" else "LayerNorm")
            + (" + QK-norm" if a.qk_norm else ""),
            None,
        ),
        ("Activation", s.activation, None),
        ("Hidden size", f"{s.hidden_size:,}", None),
        (
            "Sliding window",
            f"{fmt_count(a.sliding_window)} · {a.sliding_pattern}" if a.sliding_window else "—",
            None,
        ),
        ("Embeddings", "tied" if s.tied_embeddings else "untied", "l-embed"),
    ]


def draw_factsheet(g: SVG, s: ArchSpec, x: float, y: float, w: float) -> float:
    """Returns the bottom y coordinate."""
    tile_w, tile_h, gap = (w - 2 * 10) / 3, 66, 10

    for i, (label, value, _) in enumerate(stat_tiles(s)):
        tx = x + (i % 3) * (tile_w + gap)
        ty = y + (i // 3) * (tile_h + gap)
        g.rect(tx, ty, tile_w, tile_h, r=8, cls="tile")
        g.text(tx + 12, ty + 22, label, size=9.5, cls="muted")
        g.text(tx + 12, ty + 48, value, size=19, cls="ink value")

    ry = y + 2 * tile_h + gap + 28
    g.text(x, ry, "ARCHITECTURE", size=9.5, cls="muted")
    ry += 10
    for label, value, chip in attribute_rows(s):
        g.line(x, ry, x + w, ry)
        if chip:
            g.circle(x + 5, ry + 15, 4, cls=chip)
        g.text(x + 16, ry + 19, label, size=11.5, cls="ink2")
        g.text(x + w, ry + 19, value, size=11.5, cls="ink value", anchor="end")
        ry += 29
    return ry
