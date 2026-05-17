# Memory-Aware Optimizer Risk Gates

本文件是 release 面向的风险门控摘要。详细版本见 `docs/inference_optimizer/risk_gates.md`。

## Gate 1：所有 optimizer flags 默认关闭

阻断条件：

- 任一 optimizer flag 默认 `True`。
- 关闭 flags 后仍走 memory-aware scheduler、metadata mutation 或 mixed-KV backend。

验证：

```bash
python -m pytest tests/integration/test_all_flags_off_baseline.py -v
```

## Gate 2：full-only fallback 始终可用

阻断条件：

- kernel unsupported shape 导致 serving crash。
- mixed-KV fallback 或 fused kernel 失败后不能退回 reference/full-only path。

验证：

```bash
python -m pytest tests/integration/test_fallback_paths.py -v
```

## Gate 3：metadata 三表不变量

阻断条件：

- 合并或绕过 `PhysicalBlockMeta` / `SequenceKVRef` / `VisibleBlockTable`。
- `slot_mapping` 指向 QUANT/EVICT。
- `VisibleBlockTable` 被用于写路径。

验证：

```bash
python -m pytest tests/engine/test_kv_meta.py tests/engine/test_visible_tables.py tests/integration/test_decode_mixed_kv_fallback.py -v
```

## Gate 4：FULL->QUANT 两阶段提交

阻断条件：

- commit 失败后 FULL 被释放。
- metadata 部分更新但没有 rollback。
- P4a read path 不可用时释放 FULL。

验证：

```bash
python -m pytest tests/engine/test_quant_commit.py tests/integration/test_full_reuse_after_quant.py -v
```

## Gate 5：固定 KV budget accounting

阻断条件：

- quant pool 在 full pool 之外额外追加。
- scratch、scale、metadata 未计入 `total_kv_budget_bytes`。
- headline 只报告 raw VRAM，不报告 effective KV/free full/OOM/stable concurrency。

验证：

```bash
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b3_profile.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8
```

## Gate 6：EVICT 只属于 P5 optional path

阻断条件：

- B3/B4/B5 默认启用 EVICT。
- `enable_direct_full_evict` 默认打开。
- B2c 未通过 quality gate 却进入 headline。

验证：

```bash
python -m pytest tests/engine/test_evict_policy_guards.py tests/quality/test_evict_quality_gate.py -v
```

## Gate 7：kernel parity and fallback

阻断条件：

- Triton path 无 torch reference。
- unsupported shape、compile failure、runtime failure 或 parity mismatch 不能 fallback。
- benchmark 中 parity validation 默认开启并污染 hot path。

验证：

```bash
python -m pytest tests/kernels/test_gather_dequant.py tests/kernels/test_mixed_kv_decode_attention.py tests/kernels/test_attention_mass_output.py -v
```

## 当前 P7 状态

- Gate 1 新增 `tests/integration/test_all_flags_off_baseline.py`。
- Gate 2 新增 `tests/integration/test_fallback_paths.py`。
- B4/B5 profile artifacts 均保持 `evicted_blocks=0`。
- B5 profile：`fused_kernel_calls=15512`、`fused_kernel_fallbacks=0`、`parity_check_calls=0`。
