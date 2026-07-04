"""'Can I run it?' — quantization-aware memory math from the same spec the figures use."""

from __future__ import annotations

import shutil
import subprocess

from llmviz.spec import ArchSpec

# bytes per parameter, including quantization overhead (llama.cpp-style)
QUANTS = [("fp16 / bf16", 2.0), ("q8_0", 1.0625), ("q4_K_M", 0.5625)]

GPUS = [
    ("RTX 3060 12GB", 12),
    ("RTX 4080 16GB", 16),
    ("RTX 4090 / 3090 24GB", 24),
    ("RTX 6000 Ada 48GB", 48),
    ("A100 / H100 80GB", 80),
]

OVERHEAD_GB = 1.5  # CUDA context + activations headroom


def local_vram_gb() -> tuple[str, float] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        name, mem = out.rsplit(",", 1)
        return name.strip(), float(mem) / 1024
    except Exception:
        return None


def fit_report(spec: ArchSpec, context: int | None = None) -> list[dict]:
    """One row per quantization: weight GB, KV GB at context, total GB, fitting GPUs."""
    ctx = context or min(spec.context_length or 8192, 32768)
    kv_gb = spec.kv_cache_per_token_bytes * ctx / 2**30  # fp16 KV
    rows = []
    for name, bpp in QUANTS:
        weights_gb = spec.total_params * bpp / 2**30
        kv = kv_gb if bpp == 2.0 else kv_gb / 2  # assume q8 KV alongside quantized weights
        total = weights_gb + kv + OVERHEAD_GB
        rows.append(
            {
                "quant": name,
                "weights_gb": weights_gb,
                "kv_gb": kv,
                "context": ctx,
                "total_gb": total,
                "fits": [g for g, vram in GPUS if total <= vram],
            }
        )
    return rows
