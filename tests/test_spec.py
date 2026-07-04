"""Ground truth: published parameter counts. If the math reproduces them, the parser is right."""

import json
from pathlib import Path

import pytest

from llmviz.spec import ArchSpec, AttentionKind, parse_config

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> ArchSpec:
    return parse_config(json.loads((FIXTURES / f"{name}.json").read_text()), name=name)


def approx(actual: int, expected: int, tol: float) -> bool:
    return abs(actual - expected) / expected <= tol


# --- detection ---


@pytest.mark.parametrize(
    "fixture,kind",
    [
        ("gpt2", AttentionKind.MHA),
        ("llama3_8b", AttentionKind.GQA),
        ("mistral_7b", AttentionKind.GQA),
        ("qwen3_06b", AttentionKind.GQA),
        ("qwen3_moe", AttentionKind.GQA),
        ("deepseek_v3", AttentionKind.MLA),
        ("gemma3_4b", AttentionKind.GQA),
        ("gpt_oss_20b", AttentionKind.GQA),
    ],
)
def test_attention_kind(fixture, kind):
    assert load(fixture).attention.kind == kind


def test_moe_detection():
    assert load("llama3_8b").moe is None
    ds = load("deepseek_v3")
    assert ds.moe and ds.moe.num_experts == 256 and ds.moe.experts_per_tok == 8
    assert ds.moe.shared_experts == 1 and ds.moe.dense_layers == 3
    qw = load("qwen3_moe")
    assert qw.moe and qw.moe.num_experts == 128 and qw.moe.experts_per_tok == 8
    oss = load("gpt_oss_20b")
    assert oss.moe and oss.moe.num_experts == 32 and oss.moe.experts_per_tok == 4


def test_structure_flags():
    assert load("gpt2").positional == "learned"
    assert load("llama3_8b").positional == "rope"
    assert load("mistral_7b").attention.sliding_window == 4096
    assert load("qwen3_06b").attention.sliding_window is None
    assert load("qwen3_06b").attention.qk_norm  # qwen3 model_type quirk
    assert load("qwen3_06b").tied_embeddings
    assert not load("llama3_8b").tied_embeddings
    assert load("gemma3_4b").attention.sliding_window == 1024  # nested text_config unwrapped
    assert load("gpt_oss_20b").attention.sliding_window == 128
    assert load("gpt_oss_20b").attention.attention_bias
    assert load("gpt2").norm_type == "layernorm"
    assert load("llama3_8b").norm_type == "rmsnorm"


# --- parameter estimation vs published counts ---


@pytest.mark.parametrize(
    "fixture,total,tol",
    [
        ("gpt2", 124_439_808, 0.005),
        ("llama3_8b", 8_030_261_248, 0.005),
        ("mistral_7b", 7_241_732_096, 0.005),
        ("qwen3_06b", 596_049_920, 0.02),
        ("qwen3_moe", 235_000_000_000, 0.02),
        ("deepseek_v3", 671_000_000_000, 0.03),
        ("gpt_oss_20b", 20_900_000_000, 0.03),
        ("gemma3_4b", 3_880_000_000, 0.05),  # text tower only (vision excluded)
    ],
)
def test_total_params(fixture, total, tol):
    spec = load(fixture)
    assert approx(spec.total_params, total, tol), f"{spec.total_params:,} vs {total:,}"


@pytest.mark.parametrize(
    "fixture,active,tol",
    [
        ("qwen3_moe", 22_000_000_000, 0.05),
        ("deepseek_v3", 37_000_000_000, 0.05),
        # OpenAI reports 3.61B "active" excluding the untied unembedding matrix (579M);
        # llmviz counts every parameter touched in a forward pass, hence 4.19B.
        ("gpt_oss_20b", 4_186_000_000, 0.01),
    ],
)
def test_active_params(fixture, active, tol):
    spec = load(fixture)
    assert approx(spec.active_params, active, tol), f"{spec.active_params:,} vs {active:,}"


def test_dense_active_equals_total():
    spec = load("llama3_8b")
    assert spec.active_params == spec.total_params


def test_kv_cache_mla_much_smaller_than_gqa():
    # MLA's raison d'être: DeepSeek V3 (61 layers) caches less per token than Llama 3 8B (32 layers)
    ds, llama = load("deepseek_v3"), load("llama3_8b")
    assert ds.kv_cache_per_token_bytes < llama.kv_cache_per_token_bytes


# --- generic parsing of exotic/upcoming architectures ---


def test_hybrid_mixer_notes():
    assert "KDA" in load("kimi_linear").hybrid_note
    assert "mamba" in load("granite4").hybrid_note
    assert "conv-mixer" in load("lfm2").hybrid_note
    assert "lightning" in load("minimax_m1").hybrid_note
    assert load("llama3_8b").hybrid_note is None


def test_falcon_flags():
    f = load("falcon7b")
    assert f.attention.kind == AttentionKind.MQA  # via multi_query flag
    assert f.positional == "rope" and f.parallel_block


def test_olmo2_post_norm():
    assert load("olmo2").norm_mode == "post"
    assert load("gemma3_4b").norm_mode == "sandwich"
    assert load("llama3_8b").norm_mode == "pre"


def test_moe_synonyms():
    k = load("kimi_linear")  # num_experts_per_token + num_shared_experts spelling
    assert k.moe.experts_per_tok == 8 and k.moe.shared_experts == 1
    q = load("qwen3_next")  # shared expert via shared_expert_intermediate_size
    assert q.moe.shared_experts >= 1 and q.moe.num_experts == 512


def test_kimi_linear_params_match_name():
    # Kimi-Linear-48B-A3B: totals reconstructed from config alone
    k = load("kimi_linear")
    assert approx(k.total_params, 48_000_000_000, 0.05)
    assert approx(k.active_params, 3_300_000_000, 0.15)


def test_gguf_ollama_roundtrip():
    """Synthetic GGUF: header + the metadata keys our reader maps."""
    import struct

    from llmviz.gguf import gguf_to_config, parse_gguf_metadata

    def kv_str(key, val):
        k, v = key.encode(), val.encode()
        return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v

    def kv_u32(key, val):
        k = key.encode()
        return struct.pack("<Q", len(k)) + k + struct.pack("<I", 4) + struct.pack("<I", val)

    def kv_f32(key, val):
        k = key.encode()
        return struct.pack("<Q", len(k)) + k + struct.pack("<I", 6) + struct.pack("<f", val)

    kvs = [
        kv_str("general.architecture", "llama"),
        kv_str("general.name", "Tiny Test"),
        kv_u32("llama.block_count", 4),
        kv_u32("llama.embedding_length", 64),
        kv_u32("llama.attention.head_count", 8),
        kv_u32("llama.attention.head_count_kv", 2),
        kv_u32("llama.feed_forward_length", 256),
        kv_u32("llama.context_length", 2048),
        kv_f32("llama.rope.freq_base", 10000.0),
        kv_u32("llama.vocab_size", 1000),
    ]
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    blob += b"".join(kvs)

    cfg = gguf_to_config(parse_gguf_metadata(blob))
    spec = parse_config(cfg, name="tiny")
    assert spec.num_layers == 4 and spec.attention.kind == AttentionKind.GQA
    assert spec.attention.num_kv_heads == 2 and spec.vocab_size == 1000
    assert spec.positional == "rope" and spec.activation == "silu"
