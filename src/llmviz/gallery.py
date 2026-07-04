"""Static HTML gallery: inline SVGs, client-side search/sort, zero external JS."""

from __future__ import annotations

from pathlib import Path

import yaml

from llmviz.fetch import load_spec
from llmviz.render.block import render_model
from llmviz.render.svg import fmt_params
from llmviz.spec import ArchSpec

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Architecture Gallery</title>
<style>
:root {{
  --page: #ffffff; --surface: #ffffff; --ink: #000000; --ink2: #444444;
  --muted: #8a8a8a; --hairline: #d5d5d5; --accent: #4d76ea;
}}
* {{ box-sizing: border-box; margin: 0; }}
body {{ background: var(--page); color: var(--ink);
  font-family: Arial, "Helvetica Neue", "Liberation Sans", sans-serif; padding: 48px clamp(16px, 4vw, 64px); }}
h1 {{ color: var(--accent); font-family: Arial, "Helvetica Neue", sans-serif; font-weight: 800;
  font-size: clamp(2rem, 5vw, 3.4rem); letter-spacing: -0.02em; }}
.sub {{ color: var(--muted); margin: 8px 0 28px; font-size: 0.95rem; }}
.bar {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 28px; }}
input, select {{ background: var(--surface); color: var(--ink); border: 1px solid var(--hairline);
  border-radius: 8px; padding: 9px 14px; font: inherit; }}
input {{ flex: 1; min-width: 220px; }}
input:focus, select:focus {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(480px, 100%), 1fr)); gap: 20px; }}
.card {{ background: var(--surface); border: 1px solid var(--hairline); border-radius: 14px;
  padding: 10px 14px 4px; overflow: hidden; }}
.card svg {{ width: 100%; height: auto; display: block; }}
.count {{ color: var(--muted); align-self: center; font-size: 0.85rem; white-space: nowrap; }}
</style>
</head>
<body>
<h1>LLM Architecture Gallery</h1>
<p class="sub">{n} open-weight architectures · rendered by <strong>llmviz</strong> from config.json alone</p>
<div class="bar">
  <input id="q" type="search" placeholder="Filter models… (name, GQA, MLA, MoE)" aria-label="Filter models">
  <select id="sort" aria-label="Sort">
    <option value="name">Sort: name</option>
    <option value="params">Sort: total params</option>
    <option value="active">Sort: active params</option>
    <option value="layers">Sort: layers</option>
  </select>
  <span class="count" id="count"></span>
</div>
<div class="grid" id="grid">
{cards}
</div>
<script>
const q = document.getElementById('q'), sort = document.getElementById('sort'),
      grid = document.getElementById('grid'), count = document.getElementById('count');
const cards = [...grid.children];
function apply() {{
  const needle = q.value.toLowerCase();
  let shown = 0;
  for (const c of cards) {{
    const hit = c.dataset.tags.includes(needle);
    c.style.display = hit ? '' : 'none';
    shown += hit;
  }}
  count.textContent = shown + ' shown';
  const key = sort.value;
  [...cards].sort((a, b) => key === 'name'
      ? a.dataset.name.localeCompare(b.dataset.name)
      : (+b.dataset[key]) - (+a.dataset[key]))
    .forEach(c => grid.appendChild(c));
}}
q.addEventListener('input', apply); sort.addEventListener('change', apply); apply();
</script>
</body>
</html>
"""


def _card(s: ArchSpec, svg: str) -> str:
    tags = " ".join(
        [
            s.name.lower(),
            s.model_type,
            s.attention.kind.value.lower(),
            "moe" if s.moe else "dense",
            fmt_params(s.total_params).lower(),
        ]
    )
    return (
        f'<article class="card" data-name="{s.name.lower()}" data-tags="{tags}" '
        f'data-params="{s.total_params}" data-active="{s.active_params}" data-layers="{s.num_layers}">'
        f"{svg}</article>"
    )


def build_gallery(
    models_file: Path,
    out_dir: Path,
    token: str | None = None,
    limit: int | None = None,
    skip_errors: bool = False,
) -> Path:
    models: list[str] = yaml.safe_load(models_file.read_text())
    if not isinstance(models, list):
        raise SystemExit("models file must be a YAML list of HF model ids")
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for m in models:
        try:
            spec = load_spec(str(m), token=token)
            svg = render_model(spec)
        except (SystemExit, Exception) as e:  # noqa: BLE001 — watch mode skips non-decoder repos
            if skip_errors:
                print(f"  skipped {m}: {e}")
                continue
            raise
        (out_dir / f"{spec.name}.svg").write_text(svg)
        cards.append(_card(spec, svg))
        print(f"  rendered {spec.name}")
        if limit and len(cards) >= limit:
            break
    index = out_dir / "index.html"
    index.write_text(_PAGE.format(n=len(cards), cards="\n".join(cards)))
    return index


def deploy_space(out_dir: Path, space: str, token: str | None = None) -> str:
    """Push the gallery folder to a static HF Space — free public hosting."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=space, repo_type="space")
    return f"deployed: https://huggingface.co/spaces/{space}"
