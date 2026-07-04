"""LLM-generated architecture notes via LiteLLM — Ollama/llama.cpp local-first,
any LiteLLM provider string works (optional feature)."""

from __future__ import annotations

import os

from llmviz.spec import ArchSpec

# LiteLLM provider string. Local-first default; override via env or --llm.
#   ollama/llama3.2                     -> local Ollama (`ollama serve`)
#   openai/local + --api-base http://localhost:8080/v1 -> llama.cpp server
#   groq/..., gemini/..., anthropic/..., openrouter/... -> hosted providers
DEFAULT_LLM = os.environ.get("LLMVIZ_LLM", "ollama/llama3.2")


def _pick_local_ollama(default: str) -> str:
    """If the default Ollama model isn't pulled, use whatever IS installed locally."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
    except OSError:
        return default  # server down — let the normal error path explain
    want = default.removeprefix("ollama/")
    resolved = want if ":" in want else want + ":latest"
    if not names or resolved in names or want in names:
        return default
    # prefer a differently-tagged pull of the same family, else the first installed model
    family = [n for n in names if n.split(":")[0] == want.split(":")[0]]
    pick = (family or names)[0]
    print(f"note: {want} not pulled — using local model {pick}")
    return f"ollama/{pick}"

_PROMPT = """You are an expert on LLM architectures, writing in the style of Sebastian Raschka's
'The Big LLM Architecture Comparison'. Below is a normalized architecture spec parsed from a
model's config.json. Write exactly 5 concise bullets (one line each, plain text, no markdown
headers) on what is architecturally NOTABLE about this model — design trade-offs, what the
choices optimize for, and how it compares to mainstream decoder designs. Be specific with
numbers from the spec, and use the terminology legend exactly — do not invent expansions.

Terminology legend:
- MHA = multi-head attention; GQA = grouped-query attention (query heads share KV heads);
  MQA = multi-query attention (single KV head)
- MLA = Multi-head Latent Attention (DeepSeek-style: K/V compressed into a low-rank latent
  that is what gets cached; kv_lora_rank/q_lora_rank are the latent ranks)
- MoE = Mixture of Experts; experts_per_tok = how many experts the router activates PER TOKEN;
  dense_layers = the first N transformer blocks use a dense FFN instead of MoE
- total_params / active_params are in raw parameter counts (e.g. 671000000000 = 671B)
- kv_cache_per_token_bytes = fp16 KV-cache bytes per token across all layers

Spec:
{spec}"""


def explain_spec(spec: ArchSpec, llm: str | None = None, api_base: str | None = None) -> str:
    try:
        import litellm
    except ImportError as e:
        raise SystemExit("explain needs the 'explain' extra: pip install llmviz[explain]") from e

    model = llm or DEFAULT_LLM
    if llm is None and "LLMVIZ_LLM" not in os.environ and model.startswith("ollama/"):
        model = _pick_local_ollama(model)
    litellm.suppress_debug_info = True
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "user", "content": _PROMPT.format(spec=spec.model_dump_json(indent=1))}
            ],
            api_base=api_base,
            max_tokens=3000,  # reasoning models spend budget on thinking first
        )
    except Exception as e:
        raise SystemExit(
            f"LLM call failed for '{model}': {e}\n\n"
            "Hints:\n"
            "  Ollama:     ollama serve && ollama pull llama3.2   (default: ollama/llama3.2)\n"
            "  llama.cpp:  llama-server -m model.gguf, then: --llm openai/local "
            "--api-base http://localhost:8080/v1\n"
            "  hosted:     any LiteLLM string works, e.g. --llm groq/llama-3.3-70b-versatile,\n"
            "              gemini/gemini-2.0-flash, anthropic/claude-opus-4-8 "
            "(set the provider's API key env var)\n"
            "  default:    set LLMVIZ_LLM to change the default provider string"
        ) from e

    msg = response.choices[0].message
    content = msg.content or ""
    if not content.strip():
        # reasoning models (deepseek-r1, qwen3) may put everything in reasoning_content
        content = getattr(msg, "reasoning_content", None) or ""
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1]
    if not content.strip():
        raise SystemExit(f"'{model}' returned an empty response.")
    return content.strip()
