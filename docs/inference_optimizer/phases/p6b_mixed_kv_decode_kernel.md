## P6b：decode-only mixed-KV attention kernel，可选 block_attn_mass 输出

### Objective

实现 decode-only fused mixed-KV attention kernel，直接读取 FULL/QUANT visible entries，并可选输出 block attention mass。P6b 默认基于 B3 QUANT-only；EVICT skip 支持存在，但 B5 headline 不默认依赖 EVICT。

### Dependencies / Parallelism

```text
Dependencies:
  P4a mixed-KV fallback reference
  P6a gather/dequant lessons
  P5 only if testing EVICT optional path, not required for B5 QUANT-only

Must finish before:
  P7 final benchmark report
```

### Files to Add / Modify

```text
Modify:
  nanovllm/layers/attention.py
  nanovllm/engine/model_runner.py
  nanovllm/engine/kv_policy.py
  nanovllm/config.py

Add:
  nanovllm/kernels/mixed_kv_decode_attention.py
  tests/kernels/test_mixed_kv_decode_attention.py
  tests/kernels/test_attention_mass_output.py
  benchmarks/microbench_mixed_kv_decode.py
```

### Public Interfaces / Function Signatures

```python
def mixed_kv_decode_attention_supported(
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    block_size: int,
    device: torch.device,
) -> bool:
    """Return whether fused mixed-KV decode kernel supports this config."""


def mixed_kv_decode_attention(
    q: torch.Tensor,
    full_k_cache: torch.Tensor,
    full_v_cache: torch.Tensor,
    quant_k_cache: torch.Tensor,
    quant_v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    visible_entries: torch.Tensor,
    seq_lens: torch.Tensor,
    head_map: torch.Tensor,
    output: torch.Tensor,
    block_attn_mass: torch.Tensor | None = None,
) -> None:
    """Decode-only mixed-KV paged attention kernel.
    Reads FULL and QUANT entries; skips EVICT entries if present.
    Optionally writes per-block attention mass.
    Raises:
        KernelNotSupportedError: unsupported shape.
        KernelRuntimeError: kernel execution failure.
    """


def update_attention_mass_ema(
    refs: list[SequenceKVRef],
    block_attn_mass: torch.Tensor,
    alpha: float,
) -> None:
    """Update per-ref attention mass EMA after decode step.
    Raises:
        PolicyStateError: mass output shape does not match visible entries.
    """
```

### Key Implementation Steps

1. Implement torch reference for fused semantics.
2. Implement Triton decode-only kernel.
3. Support FULL direct read and QUANT dequant inside kernel.
4. Skip EVICT entries if present.
5. Add optional `block_attn_mass` output.
6. Add runtime fallback to P4/P6a backend.
7. Run B5.

### Feature Flags

```text
enable_mixed_kv_decode_kernel=False
enable_attention_mass_output=False
```

### Validation Commands

```bash
python -m pytest tests/kernels/test_mixed_kv_decode_attention.py -v
python -m pytest tests/kernels/test_attention_mass_output.py -v
python benchmarks/microbench_mixed_kv_decode.py --head-dim 128 --block-size <calibrated_block_size> --kv-mix-ratio 0.5
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b5_mixed_kv_decode_kernel.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-mixed-kv-decode-kernel --reclaim-policy arkv_q8
```

### Definition of Done

- Fused decode kernel matches fallback reference within tolerance。
- Unsupported shapes fallback automatically。
- B5 runs without EVICT by default。
- Optional attention mass output updates EMA only when enabled。
- Turning off kernel flag returns to P4/P6a path。

### Failure Modes and Rollback

| Failure mode | Rollback operation | Impact on later phases |
|---|---|---|
| Kernel parity failure | Disable `enable_mixed_kv_decode_kernel` | B5 no headline; B3/B4 remain valid |
| attention_mass output corrupts policy | Disable `enable_attention_mass_output` | Core B5 still valid |
| Kernel slower than fallback | Keep kernel experimental/off | Release can proceed with fallback |
| EVICT skip bug | Test only QUANT-only B5; disable EVICT kernel path | Optional EVICT result blocked |

### Estimated Days

```text
7-12 days
```

### Codex Implementation Prompt

```text
Implement a decode-only mixed-KV attention kernel that consumes VisibleBlockEntry
metadata and directly reads FULL and QUANT KV pages, skipping EVICT entries if
present. Add a torch reference, numerical parity tests, performance microbench,
runtime fallback, and optional block_attn_mass output for policy EMA. Keep the
kernel default-off. B5 must be evaluated on B3 QUANT-only by default, not on
EVICT.
```

---
