"""llmviz CLI: render / diff / gallery / inspect."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="LLM architecture diagrams from config.json — no weights needed.",
)

Token = typer.Option(None, envvar="HF_TOKEN", help="HF token for gated repos")


def _write(svg: str, out: Path) -> None:
    if out.suffix == ".png":
        try:
            import cairosvg
        except ImportError as e:
            raise SystemExit("PNG export needs the 'png' extra: pip install llmviz[png]") from e
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out), scale=2)
    else:
        out.write_text(svg)
    typer.echo(f"wrote {out}")


@app.command()
def render(
    model: str = typer.Argument(help="HF model id (org/name) or path to a config.json"),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Output .svg or .png (default <name>.svg)"
    ),
    animate: bool = typer.Option(False, "--animate", help="Staggered build-up (SVG, browsers)"),
    token: str | None = Token,
):
    """Render one model's Raschka-style architecture figure."""
    from llmviz.fetch import load_spec
    from llmviz.render.block import render_model

    spec = load_spec(model, token=token)
    _write(render_model(spec, animate=animate), out or Path(f"{spec.name}.svg"))


@app.command()
def diff(
    model_a: str,
    model_b: str,
    out: Path | None = typer.Option(None, "--out", "-o"),
    token: str | None = Token,
):
    """Render two architectures side by side with differences flagged."""
    from llmviz.fetch import load_spec
    from llmviz.render.diff import render_diff

    a, b = load_spec(model_a, token=token), load_spec(model_b, token=token)
    _write(render_diff(a, b), out or Path(f"{a.name}-vs-{b.name}.svg"))


@app.command()
def card(
    model: str,
    out: Path | None = typer.Option(None, "--out", "-o", help="Output .svg or .png (1200x630)"),
    token: str | None = Token,
):
    """Render a 1200x630 social card for the model."""
    from llmviz.fetch import load_spec
    from llmviz.render.card import render_card

    spec = load_spec(model, token=token)
    _write(render_card(spec), out or Path(f"{spec.name}-card.png"))


@app.command()
def lineage(
    models: list[str] = typer.Argument(help="2+ HF model ids, oldest first"),
    out: Path | None = typer.Option(None, "--out", "-o"),
    token: str | None = Token,
):
    """Render a family evolution strip with per-generation changes flagged."""
    from llmviz.fetch import load_spec
    from llmviz.render.lineage import render_lineage

    specs = [load_spec(m, token=token) for m in models]
    _write(render_lineage(specs), out or Path("lineage.svg"))


@app.command()
def watch(
    n: int = typer.Option(12, "--n", help="How many trending models to include"),
    out_dir: Path = typer.Option(Path("gallery"), "--out", "-o"),
    space: str | None = typer.Option(
        None, "--space", help="Also deploy to this HF Space (user/name)"
    ),
    token: str | None = Token,
):
    """Build a gallery of the Hub's trending text-generation models right now."""
    from huggingface_hub import list_models

    from llmviz.gallery import build_gallery, deploy_space

    ids = []
    for m in list_models(pipeline_tag="text-generation", sort="trendingScore", limit=n * 4):
        ids.append(m.id)
        if len(ids) >= n * 4:
            break
    import tempfile

    import yaml as _yaml

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        _yaml.safe_dump(ids, f)
        listing = Path(f.name)
    index = build_gallery(listing, out_dir, token=token, limit=n, skip_errors=True)
    typer.echo(f"wrote {index}")
    if space:
        typer.echo(deploy_space(out_dir, space, token=token))


@app.command()
def gallery(
    models_file: Path = typer.Argument(help="YAML file: a list of HF model ids"),
    out_dir: Path = typer.Option(Path("gallery"), "--out", "-o"),
    space: str | None = typer.Option(
        None, "--space", help="Also deploy to this HF Space (user/name)"
    ),
    token: str | None = Token,
):
    """Batch-render a static HTML gallery from a YAML list of model ids."""
    from llmviz.gallery import build_gallery, deploy_space

    index = build_gallery(models_file, out_dir, token=token)
    typer.echo(f"wrote {index}")
    if space:
        typer.echo(deploy_space(out_dir, space, token=token))


@app.command()
def mcp():
    """Run llmviz as an MCP server (stdio) — inspect/fit/render/diff as agent tools."""
    from llmviz.mcp_server import serve

    serve()


@app.command()
def fit(
    model: str,
    context: int | None = typer.Option(None, "--context", "-c", help="Context length for KV cache (default min(model max, 32k))"),
    token: str | None = Token,
):
    """Can I run it? Quantization-aware memory needs and which GPUs fit."""
    from rich.console import Console
    from rich.table import Table

    from llmviz.fetch import load_spec
    from llmviz.fit import fit_report, local_vram_gb

    spec = load_spec(model, token=token)
    rows = fit_report(spec, context)
    t = Table(title=f"{spec.name} — memory to run ({rows[0]['context']:,} ctx)")
    for col in ("Quant", "Weights", "KV cache", "Total", "Fits on"):
        t.add_column(col)
    gpu = local_vram_gb()
    for r in rows:
        fits = ", ".join(g.split()[0] + " " + g.split()[-1] for g in r["fits"][:3]) or "multi-GPU only"
        t.add_row(r["quant"], f"{r['weights_gb']:.1f} GB", f"{r['kv_gb']:.1f} GB",
                  f"{r['total_gb']:.1f} GB", fits)
    Console().print(t)
    if gpu:
        name, vram = gpu
        ok = [r["quant"] for r in rows if r["total_gb"] <= vram]
        verdict = (
            f"fits at {', '.join(ok)}"
            if ok
            else "does not fit fully in VRAM — Ollama/llama.cpp will offload layers to CPU (slower)"
        )
        Console().print(f"Your GPU ({name.strip()}, {vram:.0f} GB): {verdict}")
    if spec.is_moe:
        Console().print("MoE: all experts must be resident — totals use all "
                        f"{spec.moe.num_experts} experts, not the active subset.")


@app.command()
def explain(
    model: str,
    llm: str | None = typer.Option(
        None,
        "--llm",
        help="LiteLLM provider string (default ollama/llama3.2; llama.cpp: openai/local "
        "with --api-base; also groq/…, gemini/…, anthropic/…)",
    ),
    api_base: str | None = typer.Option(
        None,
        "--api-base",
        help="OpenAI-compatible endpoint, e.g. llama.cpp at http://localhost:8080/v1",
    ),
    token: str | None = Token,
):
    """LLM-written notes on what's architecturally interesting (local-first via Ollama/llama.cpp)."""
    from llmviz.explain import explain_spec
    from llmviz.fetch import load_spec

    spec = load_spec(model, token=token)
    typer.echo(explain_spec(spec, llm=llm, api_base=api_base))


@app.command()
def inspect(
    model: str,
    token: str | None = Token,
):
    """Print the parsed architecture fact sheet as a table."""
    from rich.console import Console
    from rich.table import Table

    from llmviz.fetch import load_spec
    from llmviz.render.factsheet import attribute_rows, stat_tiles

    spec = load_spec(model, token=token)
    t = Table(title=spec.name, show_header=False, title_justify="left")
    for label, value, _ in stat_tiles(spec) + attribute_rows(spec):
        t.add_row(label.title() if label.isupper() else label, value)
    Console().print(t)
