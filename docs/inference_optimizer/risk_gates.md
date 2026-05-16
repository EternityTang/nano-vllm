# 风险门控

## Risk Gates 总原则

以下 7 条 Risk Gates 是 **merge gate**，不是 README 提醒。任何 gate 未通过，对应 Phase 不允许 merge 到主线。关闭所有 optimizer flags 后，full-only fallback 必须始终可用。Gate 1 是跨阶段常驻门：P-1 之后每个 Phase merge 前都必须重新跑 all-flags-off regression，而不是只在 P0/P7 抽查。

---

## Gate 1：All Optimizer Flags Default-Off Merge Gate

### Gate 名称

```text
All optimizer feature flags default off
```

### 阻断条件

```text
任一 optimizer flag 默认 True
关闭 flags 后仍走 memory-aware scheduler / metadata mutation / mixed-KV backend
```

### 必须通过的测试或 benchmark

```bash
python -m pytest tests/integration/test_all_flags_off_baseline.py -v
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 8 --output-json results/gate1_all_off.json
```

### 失败时的 rollback / fallback

```text
重置所有 optimizer flags 默认 False
full-only scheduler + full-only KV + full-only attention
```

### 负责拦截的 Phase

```text
P-1, P0, P1, P2, P3, P4a, P4b, P5, P6a, P6b, P7
```

---

## Gate 2：FULL-only Fallback Path Merge Gate

### Gate 名称

```text
full-only fallback path always available
```

### 阻断条件

```text
mixed-KV fallback/kernel 出错时无法回退
kernel unsupported shape 导致 serving crash
P4/P6 code path 删除或破坏 full-only attention
```

### 必须通过的测试或 benchmark

```bash
python -m pytest tests/integration/test_fallback_paths.py -v
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 8 --output-json results/gate2_fallback.json --disable-all-optimizer-flags
```

### 失败时的 rollback / fallback

```text
关闭 enable_mixed_kv_fallback
关闭 enable_triton_gather_dequant
关闭 enable_mixed_kv_decode_kernel
退回 full-only attention
```

### 负责拦截的 Phase

```text
P4a, P4b, P6a, P6b, P7
```

---

## Gate 3：Metadata Invariants Merge Gate

### Gate 名称

```text
logical / physical / visible table invariants
```

### 阻断条件

```text
PhysicalBlockMeta 被合并回单一 KVBlockMeta
SequenceKVRef 被删除
shared-prefix block 被建模为单 seq owner
logical_context_len 与 visible_context_len 混用
slot_mapping 指向 QUANT/EVICT
VisibleBlockTable 用于写路径
```

### 必须通过的测试或 benchmark

```bash
python -m pytest tests/engine/test_kv_meta.py -v
python -m pytest tests/engine/test_visible_tables.py -v
python -m pytest tests/integration/test_decode_mixed_kv_fallback.py -v
```

### 失败时的 rollback / fallback

```text
关闭 enable_arkv_metadata
关闭 enable_kv_q8_runtime
退回原 full-only block manager
```

但禁止通过合并 PhysicalBlockMeta 和 SequenceKVRef 来“修复”。

### 负责拦截的 Phase

```text
P2, P3, P4a, P4b
```

---

## Gate 4：Two-Phase Quant Commit Merge Gate

### Gate 名称

```text
quantize_from_full rollback-safe two-phase commit
```

### 阻断条件

```text
quantize 失败后 FULL 被释放
metadata 更新部分成功但没有 rollback
SequenceKVRef / PhysicalBlockMeta / VisibleBlockTable 不一致
FULL->QUANT 后无 mixed-KV read path 却释放 full block
```

### 必须通过的测试或 benchmark

```bash
python -m pytest tests/engine/test_quant_commit.py -v
python -m pytest tests/integration/test_full_reuse_after_quant.py -v
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 8 --output-json results/gate4_quant_commit.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback
```

### 失败时的 rollback / fallback

```text
rollback transaction
保留 FULL
释放 quant slot
disable enable_kv_q8_runtime
保留 enable_kv_q8_shadow only
```

### 负责拦截的 Phase

```text
P3, P4a
```

---

## Gate 5：KV Budget Accounting Merge Gate

### Gate 名称

```text
fixed total_kv_budget_bytes split
```

### 阻断条件

```text
quant pool 在 full pool 之外额外追加
scratch / scale / metadata 未计入 total_kv_budget_bytes
headline 只报告 raw peak VRAM
effective KV memory / free full blocks / OOM rate / max stable concurrency 缺失
```

### 必须通过的测试或 benchmark

```bash
python -m pytest tests/engine/test_kv_budget.py -v
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/gate5_budget.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback
```

### 失败时的 rollback / fallback

```text
disable quant runtime
recompute full/quant/scale/scratch/metadata split
do not publish memory headline
```

### 负责拦截的 Phase

```text
P3, P4a, P7
```

---

## Gate 6：EVICT Quality and Phase Boundary Merge Gate

### Gate 名称

```text
EVICT locked to P5 quality gate
```

### 阻断条件

```text
P3/P4 policy 生成 EVICT entry
B3/B4/B5 默认启用 EVICT
allow_direct_full_evict 默认 True
B2c 未过 quality gate 却进入 headline
EVICT shared-prefix / sink / recent / inflight-write / unfinished-prefill
```

### 必须通过的测试或 benchmark

```bash
python -m pytest tests/engine/test_evict_policy_guards.py -v
python -m pytest tests/quality/test_evict_quality_gate.py -v
python benchmarks/benchmark_serving.py --workload quality_passkey --concurrency 8 --output-json results/gate6_evict_quality.json --enable-kv-evict --enable-quality-gate
```

### 失败时的 rollback / fallback

```text
disable enable_kv_evict
disable enable_direct_full_evict
remove B2c from headline
continue with B3 QUANT-only
```

### 负责拦截的 Phase

```text
P5, P7
```

---

## Gate 7：Kernel Parity and Fallback Merge Gate

### Gate 名称

```text
Triton kernel parity and fallback
```

### 阻断条件

```text
kernel 无 torch reference
无 numerical diff test
无 microbench
unsupported shape 不 fallback
kernel parity fail 仍进入 headline
```

### 必须通过的测试或 benchmark

```bash
python -m pytest tests/kernels/test_q8_kv.py -v
python -m pytest tests/kernels/test_gather_dequant.py -v
python -m pytest tests/kernels/test_mixed_kv_decode_attention.py -v
python benchmarks/microbench_gather_dequant.py --head-dim 128 --block-size <calibrated_block_size>
python benchmarks/microbench_mixed_kv_decode.py --head-dim 128 --block-size <calibrated_block_size>
```

### 失败时的 rollback / fallback

```text
disable enable_triton_q8_kv
disable enable_triton_gather_dequant
disable enable_mixed_kv_decode_kernel
fallback to torch reference / mixed-KV fallback
```

### 负责拦截的 Phase

```text
P3, P6a, P6b, P7
```
