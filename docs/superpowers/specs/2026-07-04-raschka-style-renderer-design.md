# Raschka-identical renderer for llmviz

Approved by Hesham 2026-07-04. Replaces the colorful semantic-palette diagram style entirely.

## Goal

Make `llmviz render` output visually identical to Sebastian Raschka's LLM Architecture Gallery figures. Ground truth: his Llama 3 (8B), GPT-OSS (20B), and DeepSeek V3/R1 (671B) figures (downloaded to scratchpad and pixel-sampled).

## His grammar (fixed) + per-model accent (variable)

- Colors sampled: outer container `#d5d5d5`, mid gray (dense-model block) `#929292`, attention charcoal `#5e5e5e` (white text), pills white with black ~2.5px borders, canvas white.
- Accent per model: title, repeat count, callout numbers, block-container tint (accent mixed ~33% toward white), MoE inset border, Router outline, last expert tag. Sampled accents: blue `#5079d9`, coral `#ff664f`. Curated palette (blue, coral, green, violet, amber, teal) assigned deterministically by model family; `--accent` overrides.
- Flow bottom-up: `Sample input text` (monospace) → `Tokenized text` → outer container → `Token embedding layer` → accent block container [`RMSNorm 1` → dark attention box (distinguishing word in accent) ← `RoPE` pill w/ arrow → ⊕ → `RMSNorm 2` → `MoE`/`Feed forward` pill → ⊕] → `Final RMSNorm` → `Linear output layer` → arrow out.
- Residual: right-side rails into ⊕ circles (arrowhead into ⊕).
- Repeat: `N ×` in accent + big `{` brace glyph at block lower-left.
- Callouts: bold ~19px black text, numbers as accent tspans, dotted black leader lines. Slots: vocabulary (top right), context length (left), embedding dimension (bottom right), heads (lower right), dense-first-layers note (bottom left, MoE only), weight tying (GPT-2), sliding window (when present), QK-norm (when present).
- Right column insets: `FeedForward (SwiGLU|GELU) module` — gray fill, black dotted border, Linear/SiLU/⊗/Linear; below it the intermediate-dim callout. MoE models add accent-dotted `MoE layer` inset (Router → `Feed forward` cards tagged `1`…`N` with `···` → ⊕) and an accent `Resource savings:` bullet list (total params, active experts incl. shared, active params — computed values).
- MLA: NO separate inset (faithful to his DeepSeek figure): dark box reads "Multi-head Latent Attention" with "Latent" in accent; heads callout as usual.
- Typography: `Arial, "Helvetica Neue", "Liberation Sans", "DejaVu Sans", sans-serif` (Arial→Liberation Sans on Linux keeps cairo PNG export identical-ish); monospace stack for the input sample; no watermark; light theme only (his figures are white-canvas; dark mode dropped in this style).

## Changes

- `render/tokens.py`: replace palette with grays + `ACCENTS` dict + `accent_for(model_type)`.
- `render/block.py`: full rewrite of `draw_diagram`/`render_model` to the grammar above; fact sheet removed from render (stays in `inspect` and diff table).
- `render/svg.py`: add rich text (tspan runs), upward arrows, dotted leader helper.
- `render/factsheet.py`: keep `stat_tiles`/`attribute_rows` for inspect + diff only.
- `render/diff.py`: two towers side by side + comparison table restyled (black text, accent ≠ flags).
- `gallery.py`: white page, accent-blue chrome.
- Tests: update label assertions (`Linear output layer`, `Token embedding layer`, `Resource savings`).

## Verification

Render all 8 fixtures → PNG; visually compare side-by-side against his Llama-3/GPT-OSS/DeepSeek figures; pytest; ruff/black; rebuild gallery + headless-Chrome screenshot.
