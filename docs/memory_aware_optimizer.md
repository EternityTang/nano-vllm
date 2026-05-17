# Memory-Aware Optimizer Release Notes

## 目标

本优化器在 Nano-vLLM 的 full-only 推理路径之外，提供默认关闭的 memory-aware scheduler、admission controller、ARKV-inspired KV metadata、Q8 KV tiering、mixed-KV fallback，以及可选 Triton kernel 加速。发布目标是提高长上下文压力下的可用 KV 容量、稳定并发和 SLO-goodput，同时保留 full-only fallback。

## 架构边界

```text
Scheduler / Admission
  -> FULL block allocation
  -> PhysicalBlockMeta + SequenceKVRef
  -> Reclaim policy selects protected-safe FULL/QUANT candidates
  -> FULL -> QUANT two-phase commit
  -> VisibleBlockTable read view
  -> mixed-KV fallback or fused decode kernel
```

关键不变量：

- 所有 optimizer/kernel flags 默认关闭。
- `slot_mapping` 只写 FULL block。
- `VisibleBlockTable` 是 attention 的唯一 mixed-KV 读视图。
- P5 之前不生成 EVICT；P5 之后 EVICT 仍默认关闭且必须质量门控。
- B4/B5 默认基于 B3 QUANT-only，不使用 B2c EVICT。

## Feature Flags

| Flag | 默认值 | 作用 |
|---|---:|---|
| `enable_memory_aware_scheduler` | `False` | 启用 decode-first、chunked prefill、lane 和 starvation guard |
| `enable_admission_controller` | `False` | 启用 KV reserve-aware admission |
| `enable_arkv_metadata` | `False` | 启用 PhysicalBlockMeta / SequenceKVRef / VisibleBlockTable |
| `enable_kv_q8_shadow` | `False` | Q8 shadow 统计，不释放 FULL |
| `enable_kv_q8_runtime` | `False` | 启用 FULL->QUANT runtime commit |
| `enable_mixed_kv_fallback` | `False` | 启用 FULL/QUANT mixed read fallback |
| `enable_prefill_mixed_kv_fallback` | `False` | 启用 prefill-prefix mixed read |
| `enable_kv_evict` | `False` | 启用 P5 EVICT optional path |
| `enable_quality_gate` | `False` | 启用 EVICT quality gate |
| `enable_triton_gather_dequant` | `False` | 启用 P6a Triton gather/dequant |
| `enable_mixed_kv_decode_kernel` | `False` | 启用 P6b fused decode-only mixed-KV attention |
| `enable_attention_mass_output` | `False` | 可选输出 block attention mass |

## 运行命令

B3 QUANT-only：

```bash
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b3_profile.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8
```

B4 QUANT-only + Triton gather/dequant：

```bash
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b4_profile.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-triton-gather-dequant --reclaim-policy arkv_q8
```

B5 QUANT-only + fused decode kernel：

```bash
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b5_profile.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-mixed-kv-decode-kernel --reclaim-policy arkv_q8
```

生成 release 消融报告：

```bash
python benchmarks/report.py --results-dir results --output-markdown docs/benchmark_ablation.md --output-csv results/ablation_summary.csv
```

## 回滚策略

- 任何 scheduler/admission 异常：关闭 `enable_memory_aware_scheduler` 和 `enable_admission_controller`。
- 任何 metadata/commit 异常：关闭 `enable_arkv_metadata` 和 `enable_kv_q8_runtime`，退回 full-only。
- 任何 mixed-KV fallback 异常：关闭 `enable_mixed_kv_fallback`，不释放 FULL 给 runtime 复用。
- 任何 Triton gather/dequant 异常：关闭 `enable_triton_gather_dequant`，回退 P4 torch fallback。
- 任何 fused decode kernel 异常：关闭 `enable_mixed_kv_decode_kernel`，回退 P4/P6a path。
- 任何 EVICT 质量风险：关闭 `enable_kv_evict` 和 `enable_direct_full_evict`，B2c 不进入 headline。

## 当前效果

基于 P7 final ablation smoke sweep artifacts：

- B3 profile：`slo_goodput_tokens_per_s=24.41`，`active_quant_blocks=16`，`evicted_blocks=0`。
- B4 profile：`slo_goodput_tokens_per_s=24.22`，`gather_dequant_ms=8991.81`，`evicted_blocks=0`。
- B5 profile：`slo_goodput_tokens_per_s=40.44`，`fused_kernel_calls=15512`，`fused_kernel_fallbacks=0`，`evicted_blocks=0`。
- B2c optional quality gate：`quality_passkey` 通过，`quality_gate_passed=true`，`evicted_blocks=31`。

P6c 解释了 B5 端到端差距：metadata packing 已从 per-layer attention hot path 移到 decode preparation 并缓存；剩余瓶颈主要是 eager model-forward overhead 和大量小 fused kernel launch。

## 已知限制

- B4/B5 release headline 只覆盖 QUANT-only；EVICT 只能作为 B2c optional。
- mixed-KV fallback 仍要求 eager，直到图安全路径被独立证明。
- P6b 是 decode-only kernel，不承诺 true mixed decode+prefill batch。
- kernel unsupported shape、compile/runtime failure 或 parity mismatch 必须回退。
