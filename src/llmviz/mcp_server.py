"""MCP server: architecture facts and figures as agent tools (stdio transport).

Register in Claude Code / any MCP client:
    {"mcpServers": {"llmviz": {"command": "llmviz", "args": ["mcp"]}}}
"""

from __future__ import annotations

from pathlib import Path


def serve() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise SystemExit("mcp server needs the 'mcp' extra: pip install llmviz[mcp]") from e

    from llmviz.fetch import load_spec
    from llmviz.fit import fit_report
    from llmviz.render.block import render_model
    from llmviz.render.diff import render_diff

    app = FastMCP("llmviz")

    @app.tool()
    def inspect_architecture(model: str) -> dict:
        """Normalized architecture facts for an LLM (HF id, ollama:<name>, or .gguf path):
        attention kind, MoE config, computed total/active parameters, KV cache per token,
        norm placement, hybrid token-mixer summary. Computed from config metadata, not
        scraped — reliable for models newer than your training data."""
        return load_spec(model).model_dump()

    @app.tool()
    def memory_to_run(model: str, context_tokens: int = 32768) -> list[dict]:
        """VRAM needed to run an LLM at fp16/q8_0/q4_K_M quantization including KV cache
        at the given context length, plus which common GPUs each fits on."""
        return fit_report(load_spec(model), context_tokens)

    @app.tool()
    def render_architecture_figure(model: str, out_path: str) -> str:
        """Render a Raschka-style architecture figure (SVG) for the model and save it
        to out_path. Returns the absolute path written."""
        p = Path(out_path).expanduser().resolve()
        p.write_text(render_model(load_spec(model)))
        return str(p)

    @app.tool()
    def diff_architectures(model_a: str, model_b: str, out_path: str) -> str:
        """Render a side-by-side architecture comparison of two models (SVG) with every
        differing attribute flagged. Returns the absolute path written."""
        p = Path(out_path).expanduser().resolve()
        p.write_text(render_diff(load_spec(model_a), load_spec(model_b)))
        return str(p)

    app.run()
