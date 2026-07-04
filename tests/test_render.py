"""Smoke tests: every fixture renders to valid SVG in the Raschka grammar."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from llmviz.render.block import render_model
from llmviz.render.diff import render_diff
from llmviz.spec import parse_config

FIXTURES = Path(__file__).parent / "fixtures"
ALL = sorted(p.stem for p in FIXTURES.glob("*.json"))


def load(name):
    return parse_config(json.loads((FIXTURES / f"{name}.json").read_text()), name=name)


@pytest.mark.parametrize("fixture", ALL)
def test_renders_valid_svg(fixture):
    svg = render_model(load(fixture))
    ET.fromstring(svg)  # well-formed XML
    for label in ("Token embedding layer", "Linear output layer", "Tokenized text"):
        assert label in svg


def test_moe_vs_dense():
    moe = render_model(load("deepseek_v3"))
    assert "MoE layer" in moe and "Router" in moe and "Resource savings:" in moe
    dense = render_model(load("llama3_8b"))
    assert "Router" not in dense and "Feed forward" in dense
    assert "FeedForward (SwiGLU) module" in dense


def test_attention_labels():
    assert "Latent" in render_model(load("deepseek_v3"))
    assert "grouped-query" in render_model(load("qwen3_06b"))
    assert "Masked multi-head" in render_model(load("gpt2"))


def test_diff_flags_differences():
    svg = render_diff(load("deepseek_v3"), load("llama3_8b"))
    ET.fromstring(svg)
    assert "≠" in svg
    assert "Deepseek V3" in svg and "Llama3 (8B)" in svg


def test_gpt2_learned_positions_and_tying():
    svg = render_model(load("gpt2"))
    assert "Positional embedding layer" in svg
    assert "token embedding weights" in svg  # tied-weights callout
    assert "Positional embedding layer" not in render_model(load("llama3_8b"))


def test_dense_first_layers_note():
    assert "instead of MoE" in render_model(load("deepseek_v3"))
    assert "instead of MoE" not in render_model(load("qwen3_moe"))
