# 最终交付

## Ablation Plan：B0-B5 全 8 组

B0-B5 必须全部保留。B2c 是 optional，只有质量门通过才允许进入 headline。B4/B5 默认基于 B3 QUANT-only，不默认依赖 EVICT；`evicted_block_ratio` 只在 B2c 或 optional runs 中报告。

### B0：Nano-VLLM baseline

| 项 | 内容 |
|---|---|
| 打开的 feature flags | `enable_metrics_hooks=True` only |
| 禁用的 feature flags | all optimizer flags |
| 对应验证问题 | 原始 Nano-vLLM 在目标 workload 上的 latency / throughput / VRAM / OOM 基线是什么？ |
| headline metrics | TTFT、TPOT、throughput、raw peak VRAM、OOM rate、free full blocks、max stable concurrency |
| 是否进入 headline | 是，作为 baseline |
| 失败时如何解释 | benchmark harness 或 baseline config 不稳定，不能进入后续 ablation |

### B1：scheduler only

| 项 | 内容 |
|---|---|
| 打开的 feature flags | `enable_memory_aware_scheduler=True`, `enable_admission_controller=True` |
| 禁用的 feature flags | `enable_arkv_metadata`, `enable_kv_q8_runtime`, `enable_mixed_kv_fallback`, `enable_kv_evict`, all kernels |
| 对应验证问题 | 仅靠 decode-first + adaptive chunked prefill + admission 是否能改善 TTFT / SLO-goodput？ |
| headline metrics | TTFT p95/p99、TPOT、SLO-goodput、admission shrink/defer、queue depth |
| 是否进入 headline | 是 |
| 失败时如何解释 | scheduler policy 未带来收益，需用 queue/backpressure metrics 解释；不影响 P2/P3 correctness |

### B2a：naive Q8 tiering

| 项 | 内容 |
|---|---|
| 打开的 feature flags | `enable_arkv_metadata=True`, `enable_kv_q8_runtime=True`, `enable_mixed_kv_fallback=True`, `reclaim_policy=naive_age_q8` |
| 禁用的 feature flags | `enable_memory_aware_scheduler` 可按实验隔离关闭，`enable_kv_evict=False`, all acceleration kernels off |
| 对应验证问题 | 只是把冷 block 做成 Q8，是否已经足够？ |
| headline metrics | effective KV memory、free full blocks、quantized block ratio、OOM rate、token agreement、TPOT |
| 是否进入 headline | 是 |
| 失败时如何解释 | naive Q8 可能引入读放大或错误候选；用于证明 naive baseline 不足或 Q8 本身成本 |

### B2b：ARKV-policy Q8 tiering

| 项 | 内容 |
|---|---|
| 打开的 feature flags | `enable_arkv_metadata=True`, `enable_kv_q8_runtime=True`, `enable_mixed_kv_fallback=True`, `reclaim_policy=arkv_q8` |
| 禁用的 feature flags | `enable_kv_evict=False`, all acceleration kernels off |
| 对应验证问题 | importance-aware policy 是否比 naive age/LRU Q8 更好？ |
| headline metrics | effective KV memory、free full blocks、OOM rate、token agreement、reclaim trigger count、TPOT |
| 是否进入 headline | 是 |
| 失败时如何解释 | policy score 权重或 protection 过强/过弱；不得删除 policy baseline，应调参或解释 |

### B2c：ARKV-policy Q8 + EVICT optional

| 项 | 内容 |
|---|---|
| 打开的 feature flags | `enable_arkv_metadata=True`, `enable_kv_q8_runtime=True`, `enable_mixed_kv_fallback=True`, `enable_kv_evict=True`, `enable_quality_gate=True`, `reclaim_policy=arkv_q8_evict` |
| 禁用的 feature flags | `enable_direct_full_evict=False` by default, acceleration kernels off unless optional run |
| 对应验证问题 | 在 strict quality gate 下，EVICT 的边际容量收益是多少，质量代价是什么？ |
| headline metrics | quality gate result、passkey/retrieval drop、greedy token agreement、effective KV memory、evicted_block_ratio、max stable concurrency |
| 是否进入 headline | Optional；只有 quality gate 通过才进入 headline |
| 失败时如何解释 | EVICT 改变 attention 语义导致质量不可接受；保留 QUANT-only headline，不影响 B3/B4/B5 |

### B3：scheduler + ARKV-policy Q8

| 项 | 内容 |
|---|---|
| 打开的 feature flags | `enable_memory_aware_scheduler=True`, `enable_admission_controller=True`, `enable_arkv_metadata=True`, `enable_kv_q8_runtime=True`, `enable_mixed_kv_fallback=True`, `reclaim_policy=arkv_q8` |
| 禁用的 feature flags | `enable_kv_evict=False`, all acceleration kernels off |
| 对应验证问题 | memory savings 是否真的被 admission 转化成更高 stable concurrency / goodput / 更低 OOM？ |
| headline metrics | SLO-goodput、max stable concurrency、OOM rate、free full blocks、effective KV memory、TTFT、TPOT |
| 是否进入 headline | 是，MVP 关键结果 |
| 失败时如何解释 | Q8 有 savings 但 admission 未利用，或 fallback read 放大抵消收益；需拆看 B1/B2b |

### B4：B3 + Triton gather/dequant

| 项 | 内容 |
|---|---|
| 打开的 feature flags | B3 flags + `enable_triton_gather_dequant=True` |
| 禁用的 feature flags | `enable_kv_evict=False`, `enable_mixed_kv_decode_kernel=False` |
| 对应验证问题 | fallback mixed-KV 的额外 materialization 成本是否能被 Triton gather/dequant 部分抹平？ |
| headline metrics | TPOT、decode step latency、gather/dequant GB/s、effective KV memory、OOM rate |
| 是否进入 headline | 是，如果 kernel parity gate 通过 |
| 失败时如何解释 | kernel 不够快或 shape 不匹配；fallback path 仍是正确系统路径 |

### B5：B3 + mixed-KV decode attention kernel

| 项 | 内容 |
|---|---|
| 打开的 feature flags | B3 flags + `enable_mixed_kv_decode_kernel=True`; optional `enable_attention_mass_output=True` only in separate run |
| 禁用的 feature flags | `enable_kv_evict=False` by default |
| 对应验证问题 | fused decode kernel 是否进一步恢复 TPOT，使 FULL/QUANT 系统不只是省显存，还能恢复 decode 性能？ |
| headline metrics | TPOT、decode latency、throughput、kernel fallback rate、effective KV memory、token agreement |
| 是否进入 headline | 是，如果 parity/perf gate 通过 |
| 失败时如何解释 | kernel 不成熟；B3/B4 仍可作为 release headline，B5 标为 experimental |

## Metrics and Benchmark Workloads

### Headline Metrics

```text
TTFT mean/p50/p95/p99
TPOT mean/p50/p95/p99
request throughput
output token throughput
SLO-goodput
effective KV memory
free full block ratio
active full blocks
quantized block ratio
OOM rate
max stable concurrency
admission reject/shrink/defer count
reclaim trigger count
greedy token agreement
retrieval/passkey quality proxy
```

### Non-headline / Observational Metrics

```text
raw peak VRAM
kernel compile time
kernel fallback count
workspace high-watermark
```

### EVICT-only Metrics

只在 B2c 或 optional runs 报告：

```text
evicted_block_ratio
visible_context_len / logical_context_len ratio
passkey drop
retrieval drop
```

### Workloads

| Workload | 目的 |
|---|---|
| scheduler_stress | mixed short/long prompts，验证 scheduler-only、chunk policy、lane fairness |
| long_context_pressure | 高 KV pressure，验证 Q8、free full blocks、OOM、stable concurrency |
| shared_prefix | 验证 prefix cache、shared-block protection、owner_refs/ref_count |
| slo_goodput_stress | 并发 sweep / open-loop arrival，找到 backpressure turning point |
| quality_passkey | passkey retrieval、document retrieval、token agreement，验证 EVICT quality gate |

### Benchmark 统一规则

```text
fixed seed
greedy decoding
warmup runs
repeated runs
report variance
same model / same prompt set / same output length where possible
B4/B5 no EVICT by default
```

## 原文风险-替代方案讨论保留与强化

### 风险 1：QUANT 不降低 raw peak VRAM

原因：

```text
如果在已有 full pool 旁边额外加 quant pool，raw peak VRAM 可能上升。
```

替代方案：

```text
使用固定 total_kv_budget_bytes，在其中切 full pool / quant pool / scale / scratch / metadata。
headline 使用 effective KV memory、free full blocks、OOM rate、max stable concurrency。
raw peak VRAM 单独报告。
```

### 风险 2：mixed-KV read path 未落地时释放 FULL 会 silent corruption

原因：

```text
FULL->QUANT 后如果 old full block 被复用，而 attention 仍按 full block id 读取，会 stale read。
```

替代方案：

```text
quantize_from_full 两阶段提交。
P3 shadow mode 不释放 FULL。
P4a decode mixed-KV fallback 通过后才允许 release FULL。
```

### 风险 3：EVICT 改变 attention 语义

原因：

```text
QUANT 保留 token，只引入数值误差；EVICT 删除 visible context，会改变模型行为。
```

替代方案：

```text
EVICT 锁到 P5。
P5 必须 quality gate。
B2c optional。
B3/B4/B5 默认 QUANT-only。
```

### 风险 4：shared-prefix block 被错误归属给单 seq

原因：

```text
prefix cache 共享 physical block，多 seq owner 下单 owner 模型会破坏 ref_count 与 transition update。
```

替代方案：

```text
PhysicalBlockMeta.owner_refs + ref_count。
SequenceKVRef 保存 per-sequence logical mapping。
shared-prefix 默认禁止 EVICT，MVP 默认不 QUANT，除非能原子更新所有 owner refs。
```

### 风险 5：scheduler-only 收益掩盖 KV tiering 收益

原因：

```text
B1 可能已经改善 TTFT，使 B3 的 marginal gain 不清晰。
```

替代方案：

```text
保留 B0/B1/B2a/B2b/B3 分组。
B2a/B2b 隔离 policy/Q8。
B3 验证 scheduler + Q8 是否转化成 stable concurrency / OOM rate 改善。
```

### 风险 6：kernel 加速变成 correctness 风险

原因：

```text
Triton kernel 可能 shape 不支持、数值 mismatch、runtime compile fail。
```

替代方案：

```text
Triton-first but fallback-required。
每个 kernel 必须有 torch reference、numerical diff、microbench。
kernel flags 默认关闭。
```

### 风险 7：metadata 重构范围大导致工程风险

原因：

```text
logical/physical/visible/write 分离会触及 scheduler、block manager、attention、runner。
```

替代方案：

```text
P2 先 dry-run metadata。
P3 shadow quant。
P4a 才开启 runtime release。
每个 Phase 都有 flag rollback。
禁止通过合并表结构来“简化”。
```

## 面试回答模板

### Q1：这个项目解决了什么问题？

回答模板：

```text
这个项目解决的是长上下文和高并发推理时 KV cache 成为显存瓶颈的问题。
我没有只做一个 int8 KV cache demo，而是把问题拆成三层：
第一层是 scheduler/admission，决定什么时候 prefill、decode、admit、shrink 或 defer；
第二层是 KV manager，把 logical token timeline、physical storage 和 attention-visible view 分开；
第三层是 mixed-KV attention backend，让 FULL 和 QUANT blocks 能在真实 serving path 中被正确读取。
最终目标是把 KV savings 转化成可测的 max stable concurrency、OOM rate 和 SLO-goodput 改善。
```

### Q2：为什么不能只有 scheduler-only？

回答模板：

```text
scheduler-only 可以改善 TTFT 和队列行为，但它不能改变 KV cache 容量上限。
在 long-context pressure 下，如果 full blocks 用完，scheduler 只能 shrink、defer 或 OOM。
所以我保留 B1 作为 scheduler-only baseline，但真正的 B3 是 scheduler + ARKV-policy Q8：
它验证释放出来的 full blocks 是否被 admission 用来提高 stable concurrency。
```

### Q3：为什么要拆 PhysicalBlockMeta 和 SequenceKVRef？

回答模板：

```text
因为 shared-prefix block 不是单个 sequence 的私有 block。
PhysicalBlockMeta 是 storage truth，保存 FULL/QUANT/EVICT 状态、full/quant id、ref_count 和 owner_refs。
SequenceKVRef 是 sequence truth，保存某个 seq 的 logical block 到 storage_id 的映射，以及 is_sink、is_recent、is_inflight_write 等 per-seq protection。
如果把它们合成一个 KVBlockMeta，会在 shared-prefix、EVICT guard 和 atomic transition 上出错。
```

### Q4：为什么要分 logical / physical / visible 三张表？

回答模板：

```text
logical table 表示 token 时间线，必须 append-only，并驱动 position ids 和 sequence progress；
physical table 表示真实存储位置和状态；
visible table 是每个 step attention 实际读到的 view。
QUANT 不改变 logical 或 visible token 数，只改变数值精度；
EVICT 不改变 logical_context_len，但会减少 visible_context_len。
如果三者混用，position、attention 和 allocator 会互相污染。
```

### Q5：为什么 P4a 和 P4b 要拆开？

回答模板：

```text
decode-only mixed-KV 是最小可闭环路径：decode attention 读 FULL/QUANT，FULL->QUANT 后释放 full block，并让 scheduler/admission 使用这些容量。
prefill-prefix 和 unfinished-prefill 的 mixed read 更复杂，涉及 chunk split、prefix cache 和写路径并存。
所以 P4a 通过就能证明 MVP 的 B2/B3 价值，P4b 失败也不应该阻断核心 headline。
```

### Q6：为什么 EVICT 不能提前做？

回答模板：

```text
QUANT 保留全部 token，只引入数值误差；EVICT 会改变 attention 可见上下文，是语义变化。
因此 EVICT 必须锁到 P5 quality gate。
P4a/P4b backend 可以识别 EVICT entry 并跳过，但 policy 在 P5 前不能生成 EVICT。
B2c 也是 optional，只有 passkey/retrieval/token agreement 质量门通过才进入 headline。
```

### Q7：为什么需要 fallback 和 fused kernel 两条路径？

回答模板：

```text
fallback path 是 correctness path：它让 FULL/QUANT/EVICT 语义先跑通，并作为 kernel reference。
fused kernel 是 performance path：在 correctness 已经稳定后减少 mixed-KV 的 dequant/gather overhead。
如果只有 kernel，没有 fallback，调试和回滚风险太高；如果只有 fallback，可能省了显存但 TPOT 变差。
所以我设计了 P4 fallback、P6a gather/dequant、P6b fused decode kernel 的分阶段路线。
```

## README 最终内容要求

README 至少包含：

```text
1. 架构 ASCII 图
2. MVP vs Full Target 对照表
3. Feature flags 默认值
4. B0-B5 ablation 命令
5. Metrics schema
6. Risk Gates
7. Rollback instructions
8. Known limitations
9. Recommended RTX 4070 Ti config
10. Interview narrative link
```

## Final Deliverables

### Code

```text
nanovllm/engine/admission.py
nanovllm/engine/tasks.py
nanovllm/engine/kv_meta.py
nanovllm/engine/kv_policy.py
nanovllm/engine/visible_tables.py
nanovllm/engine/arkv_kv_manager.py
nanovllm/engine/quant_cache.py
nanovllm/layers/mixed_kv_fallback.py
nanovllm/kernels/q8_kv.py
nanovllm/kernels/triton_gather_dequant.py
nanovllm/kernels/mixed_kv_decode_attention.py
```

### Tests

```text
metadata invariant tests
capability smoke tests
scheduler/admission unit tests
quant two-phase commit tests
mixed-KV fallback integration tests
EVICT quality gate tests
kernel parity tests
all-flags-off regression tests
fallback path tests
```

### Benchmarks

```text
benchmark_serving.py
capability_probe.py
scheduler_stress
long_context_pressure
shared_prefix
slo_goodput_stress
quality_passkey
microbench_q8_kv
microbench_gather_dequant
microbench_mixed_kv_decode
ablation report generator
```

### Documentation

```text
README.md
docs/memory_aware_optimizer.md
docs/benchmark_ablation.md
docs/risk_gates.md
docs/interview_narrative.md
```

## 最终判断

Nano-vLLM 的正确升级方向不是“在现有 scheduler 上补一点 int8 accounting”，而是把它重构成一个以 task abstraction 为骨架、以 memory-aware scheduler/admission 为调度中枢、以 ARKV-inspired tri-state KV manager 为显存核心、以 mixed-KV fallback/fused backend 为执行底座、并以 benchmark/ablation/risk gates 为验收标准的完整推理系统。

MVP 的关键闭环是：

```text
P-1 capability calibration
  -> P0 metrics
  -> P1 homogeneous memory-aware scheduler
  -> P2 PhysicalBlockMeta + SequenceKVRef + visible tables
  -> P3 FULL->QUANT two-phase commit
  -> P4a decode-only mixed-KV fallback
  -> B2/B3 headline
```

Full Target 的增强闭环是：

```text
P4b prefill mixed read
  -> P5 EVICT quality gate
  -> P6a Triton gather/dequant
  -> P6b mixed-KV decode attention kernel
  -> P7 release hardening
```

所有 optimizer feature flags 默认关闭，full-only fallback path 始终可用，任何 Phase 失败都必须能回滚到前一条可靠路径。
