## P6a：Triton gather/dequant kernel

### Objective

实现 Triton gather+dequant kernel，降低 mixed-KV fallback 的 QUANT materialization 成本。P6a 不改变 policy，不引入 EVICT headline；默认基于 B3 QUANT-only。

### Dependencies / Parallelism

```text
Dependencies:
  P4a mixed-KV fallback reference

Can run in parallel with:
  P4b after P4a reference is stable
  P5 quality gate, but metrics must remain separated

Must finish before:
  P6b fused decode kernel comparison
```

### Files to Add / Modify

```text
Modify:
  nanovllm/layers/mixed_kv_fallback.py
  nanovllm/config.py

Add:
  nanovllm/kernels/triton_gather_dequant.py
  tests/kernels/test_gather_dequant.py
  benchmarks/microbench_gather_dequant.py
```

### Public Interfaces / Function Signatures

```python
def triton_gather_dequant_supported(
    head_dim: int,
    dtype: torch.dtype,
    block_size: int,
    device: torch.device,
) -> bool:
    """Return whether Triton gather/dequant supports this shape."""


def gather_dequant_triton(
    quant_k: torch.Tensor,
    quant_v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    visible_entries: torch.Tensor,
    output_k: torch.Tensor,
    output_v: torch.Tensor,
    *,
    block_size: int,
    head_dim: int,
) -> None:
    """Materialize quantized visible blocks into output scratch using Triton.
    Raises:
        KernelNotSupportedError: unsupported shape or dtype.
        KernelRuntimeError: Triton execution failed.
    """


def gather_dequant_reference(
    quant_cache: QuantKVCache,
    entries: list[VisibleBlockEntry],
    output: torch.Tensor,
) -> torch.Tensor:
    """Torch reference path for numerical parity."""
```

### Key Implementation Steps

1. Implement Triton kernel for supported shapes.
2. Add runtime dispatch with shape checks.
3. Compare against torch reference.
4. Add microbench GB/s and latency.
5. Fallback automatically on compile/runtime/parity failure.

### Feature Flags

```text
enable_triton_gather_dequant=False
```

### Validation Commands

```bash
python -m pytest tests/kernels/test_gather_dequant.py -v
python benchmarks/microbench_gather_dequant.py --head-dim 128 --block-size <calibrated_block_size> --quant-blocks 1024
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b4_triton_gather_dequant.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-triton-gather-dequant --reclaim-policy arkv_q8
```

### Definition of Done

- Triton gather/dequant matches torch reference within tolerance。
- Unsupported shapes automatically fallback。
- B4 runs without EVICT by default。
- Microbench reports latency and bandwidth。
- Runtime failure disables kernel without crashing serving。

### Failure Modes and Rollback

| Failure mode | Rollback operation | Impact on later phases |
|---|---|---|
| Triton compile failure | Auto fallback to torch reference | B4 no headline speedup, P6b can continue |
| Numerical mismatch | Disable Triton dispatch for shape | Kernel merge blocked for that shape |
| Performance worse than reference | Keep flag off by default | Correctness phases unaffected |
| Scratch layout incompatible | Add adapter in fallback layer, not metadata redesign | P6b may need same layout fix |

### Estimated Days

```text
4-7 days
```

### Codex Implementation Prompt

```text
Implement a Triton gather+dequant kernel for quantized KV pages using the existing
mixed-KV fallback materialization as the functional reference. Add shape support
checks, numerical parity tests, runtime fallback, and microbenchmarks. Keep the
feature default-off. B4 must be based on B3 QUANT-only and must not depend on
EVICT.
```

---
