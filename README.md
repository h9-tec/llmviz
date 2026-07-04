# llmviz

**Publication-quality LLM architecture figures from `config.json` alone — no weights, no GPU, no `transformers` install.**

[![CI](https://github.com/h9-tec/llmviz/actions/workflows/ci.yml/badge.svg)](https://github.com/h9-tec/llmviz/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llmviz)](https://pypi.org/project/llmviz/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/llmviz/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Point it at any model — a Hugging Face id, a local GGUF file, or a model installed in Ollama — and get a hand-drawn-quality architecture figure in the visual language of Sebastian Raschka's [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/): the decoder tower with dotted-leader callouts, the MoE router inset, the SwiGLU module, and parameter counts **computed from the config**, not scraped from a model card.

![DeepSeek-V3 architecture](docs/deepseek-v3.svg)

## Why

Hand-drawn galleries are wonderful and update episodically. llmviz generates the same figure in milliseconds, for any model, the day its config lands on the Hub — including architectures that didn't exist when this tool was written. The parser is generic-first (field-name synonyms, capability detection, graceful degradation), so hybrid Mamba mixers, linear attention, MLA, sandwich norms, and whatever ships next all render correctly or degrade honestly.

The numbers are the point: total and active parameters, KV-cache bytes per token, and VRAM footprints are reconstructed from per-layer math. The test suite pins them against published figures — Llama-3-8B at 8.03B, DeepSeek-V3 at 671B/37.5B, Qwen3-235B-A22B at 235B/22B — within 0.5–3%. When Kimi-Linear-48B-A3B (a model the code was never tuned for) parses to 48.9B total / 3.3B active, you know the math is doing the work.

## Install

```bash
pip install llmviz                # SVG figures
pip install "llmviz[png]"         # + PNG export (cairosvg)
pip install "llmviz[explain]"     # + LLM-written notes (LiteLLM: Ollama, llama.cpp, any provider)
pip install "llmviz[mcp]"         # + MCP server for agents
```

## Sixty seconds

```bash
llmviz render deepseek-ai/DeepSeek-V3            # the figure above → DeepSeek-V3.svg
llmviz render ollama:deepseek-r1                 # a model installed in YOUR Ollama
llmviz inspect ./model-q4.gguf                   # any local or remote .gguf (header-only read)
llmviz fit Qwen/Qwen3-235B-A22B -c 131072        # can I run it? fp16/q8/q4 + your GPU verdict
llmviz diff deepseek-ai/DeepSeek-V3 NousResearch/Meta-Llama-3-8B
```

## Commands

| Command | What it produces |
|---|---|
| `render <model>` | The architecture figure (SVG/PNG). `--animate` adds a staggered build-up. |
| `diff <a> <b>` | Two towers side by side + a comparison table with every difference flagged |
| `lineage <m1> <m2> …` | Family evolution strip with per-generation "what changed" deltas |
| `card <model>` | 1200×630 social card — headline stats + the tower, share-ready |
| `poster models.yaml` | Print-ready grid of towers on one sheet (`--cols`, `--title`) |
| `gallery models.yaml` | Self-contained static HTML gallery with search/sort. `--space user/name` deploys it to a free HF Space |
| `watch` | Gallery of the Hub's **trending** models right now — pair with `--space` on a cron for a self-updating public gallery |
| `inspect <model>` | The normalized fact sheet as a terminal table |
| `fit <model>` | Quantization-aware memory needs (weights + KV cache) and which GPUs fit — detects your local GPU via `nvidia-smi` |
| `explain <model>` | Five LLM-written notes on what's architecturally notable — local-first via Ollama |
| `mcp` | MCP server (stdio): `inspect_architecture`, `memory_to_run`, `render_architecture_figure`, `diff_architectures` as agent tools |

Every command accepts a Hugging Face id (`org/name`), a local `config.json` path, a `.gguf` file or URL, or `ollama:<name>`. Gated repos (meta-llama, google) need `--token` or `hf auth login`.

![DeepSeek-V3 vs Llama-3-8B](docs/deepseek-v3-vs-llama-3-8b.svg)

## What it reads

| Signal | Source fields |
|---|---|
| MHA / GQA / MQA | `num_key_value_heads` vs `num_attention_heads`, `multi_query` |
| MLA (DeepSeek-style latent KV) | `kv_lora_rank`, `q_lora_rank`, decoupled-RoPE head dims |
| MoE | `num_experts` / `n_routed_experts` / `num_local_experts`, five spellings of top-k, shared experts, leading dense layers |
| Hybrid token mixers | `layer_types`, `linear_attn_config`, `full_attn_idxs`, `attn_type_list`, `mamba_*` — summarized as e.g. "20 linear-attention (KDA) : 7 full attention layers" |
| Norm placement | pre (default), post (OLMo-2), sandwich (Gemma) — drawn structurally |
| The rest | sliding windows and local:global ratios, QK-norm, RoPE θ / ALiBi / learned, tied embeddings, activation |

GGUF sources are read **header-only** (a few MB, never the weights), including vocab size recovered from the tokenizer array length — so `llmviz render ollama:qwen2.5-coder:14b` diagrams a 9 GB model in under a second. For remote GGUF URLs only the metadata bytes are fetched via ranged HTTP.

Counting convention: "active" means every parameter touched in a forward pass, including embeddings and the LM head — some vendors report actives excluding the unembedding, so their number may read slightly lower.

## `explain` providers

`explain` is local-first through LiteLLM:

```bash
llmviz explain zai-org/GLM-4.5-Air                                    # local Ollama (auto-picks an installed model)
llmviz explain <m> --llm openai/local --api-base http://localhost:8080/v1   # llama.cpp server
llmviz explain <m> --llm groq/llama-3.3-70b-versatile                 # any hosted LiteLLM provider
export LLMVIZ_LLM="ollama/deepseek-r1:latest"                         # set your default
```

Reasoning models (DeepSeek-R1, Qwen3) are handled — thinking is stripped, the answer is kept.

## MCP

Give any agent architecture facts computed from configs instead of recalled from training data:

```json
{"mcpServers": {"llmviz": {"command": "llmviz", "args": ["mcp"]}}}
```

## Python API

```python
from llmviz.fetch import load_spec
from llmviz.render.block import render_model

spec = load_spec("Qwen/Qwen3-235B-A22B")      # or "ollama:deepseek-r1", "./model.gguf"
spec.total_params, spec.active_params, spec.attention.kind, spec.hybrid_note
svg = render_model(spec)
```

`ArchSpec` is a Pydantic model — `spec.model_dump_json()` gives you the normalized architecture for your own tooling.

## Development

```bash
git clone https://github.com/h9-tec/llmviz && cd llmviz
python -m venv .venv && .venv/bin/pip install -e ".[dev,png]"
.venv/bin/pytest                              # offline; fixtures are real Hub configs
.venv/bin/ruff check src tests
```

Tests treat published parameter counts as ground truth — if the per-layer math doesn't reproduce a model's documented size, the parser is wrong. CI runs on 3.11/3.12; a nightly workflow rebuilds the trending gallery and deploys it to a Hugging Face Space; tagging `v*` publishes to PyPI via trusted publishing.

## Acknowledgements

The visual language is a faithful implementation of **Sebastian Raschka's** LLM Architecture Gallery figures ([sebastianraschka.com/llm-architecture-gallery](https://sebastianraschka.com/llm-architecture-gallery/)) — colors were sampled from his published figures with admiration. If you want the hand-crafted originals with his commentary, go read [Ahead of AI](https://magazine.sebastianraschka.com/).

## License

Apache-2.0
