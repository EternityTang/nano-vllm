## P6c：end-to-end performance profiling and stabilization

### Objective

解释并降低 P6a/P6b kernel microbench speedup 与 B5 serving goodput 之间的落差。P6c 只允许增加 profiling、解释热点、做低风险 hot-path 修复；不新增 optimizer feature，不改变 EVICT 默认关闭行为，不进入 P7 release。

### Dependencies / Parallelism

```text
Dependencies:
  P6a Triton gather/dequant
  P6b decode-only mixed-KV attention kernel
  B3/B4/B5 QUANT-only benchmark lanes

Must finish before:
  P7 final ablation / release documentation
```

### Files Added / Modified

```text
Add:
  nanovllm/engine/profiler.py
  results/b3_profile.json
  results/b4_profile.json
  results/b5_profile.json

Modify:
  benchmarks/benchmark_serving.py
  benchmarks/report.py
  nanovllm/engine/llm_engine.py
  nanovllm/engine/model_runner.py
  nanovllm/engine/scheduler.py
  nanovllm/engine/arkv_kv_manager.py
  nanovllm/layers/mixed_kv_fallback.py
  nanovllm/layers/attention.py
  nanovllm/kernels/mixed_kv_decode_attention.py
  nanovllm/utils/context.py
```

### Profiling Metrics

P6c report summary must include:

```text
scheduler_ms
admission_ms
reclaim_planning_ms
quantize_from_full_ms
visible_table_build_ms
visible_table_tensor_pack_ms
workspace_planning_ms
gather_dequant_ms
mixed_kv_decode_kernel_ms
model_forward_non_attention_ms
cuda_sync_ms
fused_kernel_calls
fused_kernel_fallbacks
fallback_count
parity_check_calls
avg_step_ms
```

### Low-Risk Stabilization Applied

- Packed P6b visible metadata once in `ModelRunner.prepare_decode()` instead of rebuilding it inside every layer's attention call.
- Added a signature cache for packed visible metadata so unchanged decode batches can reuse the existing tensors.
- Changed visible metadata tensor packing to CPU batch construction plus one transfer, avoiding per-field CUDA scalar writes.
- Counted fused kernel calls, fused fallbacks, materialized fallback calls, and parity checks.
- Kept parity validation disabled in benchmark mode unless explicit validation environment variables are set.
- Confirmed B4/B5 profile runs remain QUANT-only with `evicted_blocks=0`.

### Validation Commands

```bash
python -m pytest tests/kernels/test_mixed_kv_decode_attention.py tests/kernels/test_attention_mass_output.py -q
python -m pytest tests/integration/test_decode_mixed_kv_fallback.py -q
python -m pytest tests -q
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b3_profile.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b4_profile.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-triton-gather-dequant --reclaim-policy arkv_q8
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b5_profile.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-mixed-kv-decode-kernel --reclaim-policy arkv_q8
```

### Real Results

| Metric | B3 profile | B4 profile | B5 profile |
|---|---:|---:|---:|
| `slo_goodput_tokens_per_s` | 23.04 | 25.16 | 38.64 |
| `avg_step_ms` | 199.99 | 183.14 | 119.23 |
| `fused_kernel_calls` | 0 | 0 | 15512 |
| `fused_kernel_fallbacks` | 0 | 0 | 0 |
| `visible_table_tensor_pack_ms` | 0.0 | 0.0 | 55.64 |
| `mixed_kv_decode_kernel_ms` | 0.0 | 0.0 | 4074.25 |
| `fallback_count` | 15512 | 15512 | 0 |
| `active_quant_blocks` | 16 | 16 | 16 |
| `evicted_blocks` | 0 | 0 | 0 |

Additional validation:

- Kernel/profile targeted tests: `5 passed`.
- Decode mixed fallback regression: `2 passed`.
- Full suite: `78 passed`.
- B5 dispatch is real: `fused_kernel_calls=15512`.
- Unsupported shapes did not dominate: `fused_kernel_fallbacks=0`.
- Benchmark parity validation remained disabled: `parity_check_calls=0`.
- Hot-path CUDA sync was not introduced: `cuda_sync_ms=0.0`; remaining `torch.cuda.synchronize()` sites are `ModelRunner.exit()` and CUDA graph capture, not the B3/B4/B5 eager decode hot path.

### Interpretation

P6c explains the previous B5 serving gap: P6b was dispatching the fused kernel, but visible metadata packing happened in the layer attention path and could be rebuilt repeatedly across layers/steps. Moving packing to decode preparation and caching packed tensors raised B5 from the old `results/b5_mixed_kv_decode_kernel.json` goodput of `10.02 tok/s` to `38.64 tok/s` in `results/b5_profile.json`.

Remaining gap is not caused by EVICT, unsupported kernel shapes, or parity validation. It is dominated by eager model-forward overhead, many per-layer small fused decode kernel launches, and further kernel tuning that should be considered after P6c.

### Definition of Done

- B3/B4/B5 emit all P6c profile fields.
- B5 has `fused_kernel_calls > 0` and `fused_kernel_fallbacks == 0` for the profiled workload.
- Benchmark path has `parity_check_calls == 0` unless validation is explicitly requested.
- B4/B5 remain QUANT-only with `evicted_blocks == 0`.
- Full test suite remains green.

### Failure Modes and Rollback

| Failure mode | Rollback operation | Impact on later phases |
|---|---|---|
| Profiling fields missing | Fix report schema and rerun B3/B4/B5 profile | P7 blocked |
| Packed metadata cache stale | Disable cache and keep per-step packing | B5 slower but correct |
| Fused kernel fallback dominates | Report unsupported shape and keep P4/P6a fallback | B5 experimental |
| B4/B5 accidentally include EVICT | Rerun QUANT-only flags | Release blocked until corrected |

### Next Safe Phase

```text
P7 release ablation and documentation. Do not add new optimizer behavior in P7; only harden, document, and run reproducible ablations.
```
