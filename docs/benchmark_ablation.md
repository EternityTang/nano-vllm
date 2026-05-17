# Memory-Aware Optimizer 消融报告

本报告由 benchmark JSON artifacts 生成。B2c 是 optional 且必须通过 quality gate；B4/B5 默认基于 B3 QUANT-only，必须保持 `evicted_blocks=0`。

## 汇总

| 组别 | 名称 | 状态 | Workload | Throughput tok/s | SLO-goodput tok/s | OOM | Active QUANT | EVICT | Fused calls | Fused fallbacks | Fallback count | Avg step ms |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | Baseline | ok | scheduler_stress | 243.95 | 243.95 | 0 | 0 | 0 | 0 | 0 | 0 | 38.52 |
| B1 | Scheduler only | ok | scheduler_stress | 241.47 | 241.47 | 0 | 0 | 0 | 0 | 0 | 0 | 38.92 |
| B2a | Naive Q8 | ok | long_context_pressure | 27.68 | 27.68 | 0 | 16 | 0 | 0 | 0 | 6328 | 402.14 |
| B2b | ARKV Q8 | ok | long_context_pressure | 28.14 | 28.14 | 0 | 16 | 0 | 0 | 0 | 6328 | 395.50 |
| B2c | Optional EVICT | ok | quality_passkey | 67.13 | 67.13 | 0 | 0 | 31 | 0 | 0 | 868 | 231.12 |
| B3 | Scheduler + ARKV Q8 | ok | long_context_pressure | 24.41 | 24.41 | 0 | 16 | 0 | 0 | 0 | 15512 | 188.76 |
| B4 | B3 + Triton gather/dequant | ok | long_context_pressure | 24.22 | 24.22 | 0 | 16 | 0 | 0 | 0 | 15512 | 190.22 |
| B5 | B3 + fused mixed-KV decode | ok | long_context_pressure | 40.44 | 40.44 | 0 | 16 | 0 | 15512 | 0 | 0 | 113.92 |

## Release Gates

1. All optimizer feature flags default off.
2. Full-only fallback path always available.
3. Logical / physical / visible table invariants preserved.
4. FULL->QUANT uses rollback-safe two-phase commit.
5. KV budget accounting uses fixed `total_kv_budget_bytes` split.
6. EVICT is locked to P5 optional quality gate; B4/B5 default to QUANT-only.
7. Kernel paths require torch reference parity and automatic fallback.

## 当前限制

- 当前 ARKV policy 是 block-level / rule-driven，还不是训练出的动态策略。
- `attention_mass_ema` 和 `layer_sensitivity` 仍是后续工作，不属于当前 release headline。
- P6c 后 mixed-KV path 明显优于 fallback，但仍未超过原始 full-cache fast path；当前收益重点是容量、reclaim 激活和 QUANT-only kernel path 的恢复，而不是全面替代 full-cache 快路径。

## P6c Profile 解释

B5 profile 中 `fused_kernel_calls > 0` 且 `fused_kernel_fallbacks == 0` 证明 fused kernel 已真实 dispatch。剩余 serving gap 应结合 `model_forward_non_attention_ms`、`avg_step_ms` 和小 kernel launch overhead 解释，而不是归因于 unsupported shape 或 parity check。
