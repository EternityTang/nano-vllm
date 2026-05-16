# 执行决策

## Source of Truth 与本次升级边界

本 code plan 以已上传的 **Nano-VLLM Memory-Aware Inference Optimizer Code Plan** 原文作为唯一 product/source-of-truth，并在其基础上补齐端到端数据流、Phase 依赖、函数签名、验证命令、完成定义、风险-回滚矩阵与时间估算。原文的系统主线必须完整保留：memory-aware scheduler、adaptive chunked prefill、ARKV-inspired KV tiering、FULL/QUANT/EVICT 三态、importance-aware reclaim、admission 将 KV savings 转化为 goodput，以及 benchmark/ablation 证明系统价值。

## Executive Decision

这份计划不是“保守 demo 化”，而是把 Nano-vLLM 重构成一套可以分阶段落地、可以真实跑通、可以验收、可以回滚的 inference systems 工程方案。允许对 Nano-vLLM 做大规模改造，而且这是合理的：Nano-vLLM 是轻量实现，核心逻辑集中在 scheduler、block manager、model runner、attention、config 等少量文件中，因此适合作为 memory-aware inference optimizer 的可控重构底座。

核心执行判断如下：

1. **不做技术降级。** PhysicalBlockMeta + SequenceKVRef 双表拆分、logical / physical / visible 三表分离、FULL/QUANT/EVICT 三态、watermark+hysteresis、两阶段提交、Triton-first kernel 规划、B0-B5 全 8 组 ablation 与 7 条 Risk Gates 都是硬约束。
2. **MVP 必须是真闭环。** MVP 不是 metadata accounting demo，而是必须跑通 scheduler → admission → FULL→QUANT → release full blocks → mixed-KV fallback read → benchmark B2/B3 的真实 serving 闭环。
3. **P1 不要求 true mixed batch。** P1 只输出 homogeneous batch，全 decode 或全 prefill；true mixed decode/prefill 执行推迟到 P4a/P4b 之后的 runner/backend 演进。
4. **P4 必须拆成 P4a 与 P4b。** P4a 先完成 decode-only mixed-KV fallback，形成 B2/B3 MVP 关键闭环；P4b 再扩展 prefill-prefix / unfinished-prefill mixed read。P4b 失败不得阻断 P4a headline。
5. **EVICT 锁死在 P5 quality gate。** P4a/P4b 的 attention backend 可以识别并跳过 EVICT entry，但 policy 在 P5 之前不允许生成任何 EVICT entry。
6. **QUANT pool 必须从统一 KV budget 中切走。** quant pool 不是 full pool 之外的额外显存追加；headline 指标优先使用 effective KV memory、free full blocks、OOM rate、max stable concurrency，raw peak VRAM 只作为独立观察项。
7. **所有 optimizer feature flags 默认关闭。** full-only fallback path 始终可用，关闭全部 flags 后必须回到 baseline 行为。
8. **先做 P-1 仓库能力校准。** 在 P0 前必须冻结当前 Nano-vLLM 的真实运行边界：Qwen3-first 模型支持、KV block size 约束、mixed-KV fallback reference 可行性、CUDA graph / eager 策略与本地显存配置。P-1 不降低系统目标，只防止把研究假设误当成当前代码事实。
9. **benchmark 先服从当前 repo 能跑的模型。** 当前实现以 Qwen3 runner 为主，因此主线 benchmark 先用 Qwen3 系列；Llama/TinyLlama/OPT 只在通用模型适配层落地后进入正式对照。

## Final System Goal

最终系统应具备如下闭环能力：scheduler 在每个 step 基于 decode backlog、token budget、KV full-block budget、quant-pool capacity、reclaimable full-equivalent blocks 与 future decode reserve，选择 decode / prefill tasks；ARKV-inspired KV manager 按 block importance 执行 FULL/QUANT/EVICT tiering，同时保护 shared-prefix、sink、recent 与 inflight-write blocks；attention backend 正确读取 FULL blocks、反量化 QUANT blocks，并在 EVICT 时只改变 attention 可见上下文而不破坏 logical token 时间线；admission controller 把 reclaim 释放出来的 full-block 容量转化为更高 stable concurrency、更低 OOM rate 与更高 SLO-goodput。

## MVP vs Full Target

| 范围 | 必须落地的系统路径 | 默认开关策略 | 主要价值 | 不允许降级点 |
|---|---|---|---|---|
| MVP | decode-first scheduler → admission → PhysicalBlockMeta / SequenceKVRef → FULL/QUANT tiering → quantize_from_full 两阶段提交 → decode-only mixed-KV fallback attention | 所有 optimizer flags 默认关闭，实验时显式打开 | 先证明“KV savings 能转成 goodput / stable concurrency / OOM 降低” | 不允许只做 accounting；不允许释放 FULL 后没有 mixed-KV read；不允许把 QUANT pool 作为额外追加显存 |
| Full Target | MVP 全部 + prefill-prefix mixed read + EVICT quality gate + Triton gather/dequant + decode-only mixed-KV attention kernel + 可选 block attention mass | 按 quality/perf gate 分阶段打开 | 证明 tri-state 与 kernel 加速都能带来额外收益 | EVICT 只能在 P5 之后生成；B4/B5 默认基于 B3 QUANT-only，不混入 EVICT 收益 |

## 仓库校准后的推荐模型与 RTX 4070 Ti 显存配置

Assumption：以下配置用于单卡工程验证与 ablation 排优先级，目标不是最大化模型质量，而是让 memory pressure、chunked prefill、KV reclaim、mixed-KV read 与 OOM/stable-concurrency 指标在一张消费级 GPU 上可重复触发。该 Assumption 服从 P-1 校准结果：当前 runner、flash-attn block size、CUDA graph、显存容量与本地模型可用性优先于论文级理想配置。

| 用途 | 推荐模型 | 理由 |
|---|---|---|
| 主 benchmark / serving stress | Qwen3-0.6B / Qwen3-1.7B 起步；若本地显存允许再上 Qwen3-4B 级别 | 当前 `ModelRunner` 直接实例化 Qwen3，因此 Qwen3-first 是可执行路径；用长上下文和并发 sweep 制造 KV pressure，而不是先假设 7B/8B 一定能装进 12GB |
| correctness / kernel parity | Qwen3-0.6B 或 small random Qwen3-compatible causal LM | 单测、CI、kernel 数值对拍速度快；避免在 P-1 前引入非 Qwen3 模型适配变量 |
| long-context stress | Qwen3 系列中本地可运行且支持目标 context 的最大模型 | 更容易触发 free full blocks 下降、reclaim、admission shrink/defer 与 workspace planning；实际型号由 P-1 显存校准决定 |
| quality proxy | passkey retrieval / document retrieval prompt on 与主 benchmark 相同的 Qwen3 模型 | EVICT 会改变 visible context，质量 proxy 必须与 headline 模型一致，避免模型差异污染结论 |
| 扩展模型对照 | Llama / TinyLlama / OPT | 只有在通用 model adapter 或多模型 runner 已经实现后才进入正式 ablation；否则作为 future work，不阻塞 Qwen3 主线 |

| 项 | 建议值 | 选择理由 |
|---|---:|---|
| GPU | RTX 4070 Ti 12GB | 显存紧张，适合暴露 KV pressure、OOM rate、stable concurrency 差异 |
| weight dtype | fp16 或 bf16，按模型支持选择 | 保持 FULL KV 与主流推理路径一致，避免引入额外变量 |
| KV block size | 256 tokens 起步；P-1 证明 flash-attn 与当前 config 支持后再 sweep 16/32 | 当前 Nano-vLLM 配置约束 `kvcache_block_size % 256 == 0`，正式 benchmark 必须报告实际 block size |
| max model len | 4096 起步；stress run 按模型能力和显存提高到 8192/16384 | 让 KV cache 成为主要瓶颈，而不是只测短 prompt 调度 |
| max_num_batched_tokens | 1024/2048 sweep | 用于验证 adaptive chunked prefill 是否能在 decode backlog 与 prefill throughput 间平衡 |
| max_num_seqs | 16/32/64 sweep | 用并发扫描找到 max stable concurrency |
| full pool / quant pool | 由 total_kv_budget_bytes 统一切分 | 避免“quant pool 额外加显存”的错误结论 |
| scratch budget | 固定纳入 total_kv_budget_bytes | mixed-KV fallback dequant-to-scratch 必须提前预算 |
| benchmark mode | fixed seed、greedy、warmup、repeated runs | 确保 TTFT/TPOT、token agreement、quality proxy 可比较 |

RTX 4070 Ti 12GB 的价值在于它会迫使系统面对真实内存压力：如果只用大显存卡，B0/B1 可能不容易 OOM，B2/B3 的 capacity benefit 会被弱化；如果只用很小模型且短上下文，KV cache 不是主要瓶颈，QUANT/EVICT/tiering 的价值又不明显。因此本项目的实际验证组合应是 **Qwen3-first + 长上下文 + 并发 sweep + 固定 KV budget**。7B/8B fp16/bf16 只能在 P-1 证明本地可加载、或已有权重量化/更大显存环境后进入正式 benchmark。

## Feature Flags 总原则

所有 optimizer feature flags 必须默认关闭。默认配置必须是 full-only baseline。

```python
@dataclass
class OptimizerFeatureFlags:
    enable_memory_aware_optimizer: bool = False
    enable_metrics_hooks: bool = False

    enable_memory_aware_scheduler: bool = False
    enable_admission_controller: bool = False

    enable_arkv_metadata: bool = False
    enable_arkv_policy_dry_run: bool = False

    enable_kv_q8_shadow: bool = False
    enable_kv_q8_runtime: bool = False
    enable_mixed_kv_fallback: bool = False
    enable_prefill_mixed_kv_fallback: bool = False

    enable_kv_evict: bool = False
    enable_direct_full_evict: bool = False
    enable_quality_gate: bool = False

    enable_triton_q8_kv: bool = False
    enable_triton_gather_dequant: bool = False
    enable_mixed_kv_decode_kernel: bool = False
    enable_attention_mass_output: bool = False
```

关闭所有 flags 时，系统必须满足：

```text
scheduler = legacy scheduler
kv_manager = original full-only manager
slot_mapping = original full-only write mapping
visible tables = ignored
attention backend = full-only
benchmark behavior = B0 baseline
```

# 架构与数据结构

## 系统架构

```text
                        +----------------------------+
                        |        Request Ingress     |
                        +-------------+--------------+
                                      |
                                      v
                        +----------------------------+
                        |     AdmissionController    |
                        |  - estimate KV demand      |
                        |  - shrink / defer / admit  |
                        |  - future decode reserve   |
                        +-------------+--------------+
                                      |
                                      v
                        +----------------------------+
                        |  MemoryAwareScheduler      |
                        |  - decode-first            |
                        |  - adaptive chunk prefill  |
                        |  - short/long lanes        |
                        |  - starvation guard        |
                        +-------------+--------------+
                                      |
                                      v
                        +----------------------------+
                        |        BatchPlan           |
                        |  DecodeTask / PrefillTask  |
                        |  slot_mapping              |
                        |  visible_block_tables      |
                        |  workspace_plan            |
                        +-------------+--------------+
                                      |
                                      v
                        +----------------------------+
                        |        ModelRunner         |
                        |  run_step(batch, context)  |
                        +-------------+--------------+
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
        +----------------------------+   +----------------------------+
        |   Attention Backend        |   |   ARKVPagedKVManager      |
        | full-only / mixed fallback |   | - metadata store          |
        | / fused mixed-kv decode    |   | - reclaim planner         |
        +-------------+--------------+   | - FULL/QUANT/EVICT ops    |
                      |                  +------+-----------+---------+
                      |                         |           |
                      v                         v           v
        +----------------------------+   +---------+   +----------+
        |  FULL KV Cache             |   | Quant   |   | Prefix   |
        |  paged blocks              |   | KV Cache|   | Hash/Refs|
        +----------------------------+   +---------+   +----------+

                        +----------------------------+
                        |   Metrics + Bench Harness  |
                        |   TTFT/TPOT/goodput/VRAM   |
                        +----------------------------+
```

## 分层改造落点

```text
nanovllm/
  config.py
  engine/
    admission.py
    arkv_kv_manager.py
    block_manager.py
    kv_meta.py
    kv_policy.py
    llm_engine.py
    model_runner.py
    scheduler.py
    scheduler_metrics.py
    sequence.py
    tasks.py
    visible_tables.py
    quant_cache.py
  layers/
    attention.py
    mixed_kv_fallback.py
  kernels/
    q8_kv.py
    triton_q8_kv.py
    triton_gather_dequant.py
    mixed_kv_decode_attention.py
benchmarks/
  benchmark_serving.py
  workloads/
  report.py
tests/
  engine/
  integration/
  kernels/
  quality/
```

## 三个架构不变量

### 不变量 1：写路径只写 FULL

新的 prefill/decode token 永远只写入 FULL writable blocks。`slot_mapping` 只允许指向 FULL block 的 physical slot，不允许指向 QUANT 或 EVICT。

```text
token write
  -> WriteBlockTable
  -> slot_mapping
  -> FULL KV cache
```

任何违反条件都必须 hard fail：

```python
if physical_meta.state is not KVBlockState.FULL:
    raise InvalidWriteTargetError("slot_mapping must only target FULL blocks")
```

### 不变量 2：读路径只读 VisibleBlockTable

attention backend 不直接读取 logical table，也不直接依赖 allocator 的 full block ids。读路径只认当前 step 已冻结的 `VisibleBlockTable`。

```text
attention read
  -> VisibleBlockTable[seq_id]
  -> FULL direct read / QUANT dequant-to-scratch / EVICT skip
```

### 不变量 3：reclaim 只在 step 边界提交

FULL→QUANT 或 QUANT→EVICT 不允许在 in-flight batch 中途修改可见状态。reclaim 必须在 step 边界执行：

```text
end of step N
  -> collect metrics
  -> plan reclaim
  -> execute quantize_from_full / evict transition
  -> atomic metadata commit
  -> rebuild VisibleBlockTable for step N+1
  -> release FULL block only after commit
```

## logical / physical / visible 三表分离

| 表 | 语义 | 是否 append-only | 用途 |
|---|---|---:|---|
| LogicalBlockTable | token 时间线的逻辑真相 | 是 | position ids、sequence progress、EOS/sampling、logical_context_len |
| PhysicalBlockTable | storage 真相 | 否 | FULL/QUANT/EVICT 状态、full_block_id、quant_block_id、ref_count、owner_refs |
| VisibleBlockTable | 当前 step attention read view | 否 | attention backend 按 logical order 读取 FULL/QUANT，跳过 EVICT |

`logical_context_len` 与 `visible_context_len` 不可混用。

```text
logical_context_len:
  - 表示序列真实 token 时间线长度
  - 不因 EVICT 减少
  - 驱动 position ids / sampling / sequence progress

visible_context_len:
  - 表示本 step attention 实际可见 token 数
  - EVICT 后会减少
  - 只用于 attention read
```

## 数据结构字段定义

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Literal


class KVBlockState(Enum):
    FULL = "full"       # 全精度；attention 语义不变
    QUANT = "quant"     # 低精度；attention 语义不变，仅有数值误差
    EVICT = "evict"     # 被省略；attention 可见上下文改变


@dataclass
class PhysicalBlockMeta:
    storage_id: int
    layer_id: int | None
    state: KVBlockState

    num_tokens: int
    full_block_id: int | None
    quant_block_id: int | None

    ref_count: int
    owner_refs: set[tuple[int, int]]  # (seq_id, logical_block_id)

    prefix_hash: int | None
    is_shared_prefix: bool

    create_step: int
    last_access_step: int
    transition_epoch: int

    # Full target extension fields.
    layer_stats: dict[int, float] = field(default_factory=dict)
    layer_sensitivity: dict[int, float] = field(default_factory=dict)

    # Debug / safety.
    last_transition_error: str | None = None
    pinned_reason: str | None = None


@dataclass
class SequenceKVRef:
    seq_id: int
    logical_block_id: int
    storage_id: int

    logical_start: int
    logical_end: int

    is_sink: bool
    is_recent: bool
    is_inflight_write: bool

    attention_mass_ema: float
    recency_score: float
    sink_score: float
    shared_score: float
    importance_score: float

    transition_epoch: int


@dataclass
class VisibleBlockEntry:
    seq_id: int
    logical_block_id: int
    storage_id: int

    state: KVBlockState
    logical_start: int
    logical_end: int
    visible_start: int
    visible_end: int
    num_tokens: int

    full_block_id: int | None
    quant_block_id: int | None

    # For EVICT: logical span exists but attention skips it.
    is_visible: bool
    evicted_logical_tokens: int = 0


@dataclass
class WriteCursor:
    seq_id: int
    logical_block_id: int
    storage_id: int
    full_block_id: int
    write_offset: int
    remaining_slots: int


@dataclass
class SequenceKVState:
    seq_id: int

    logical_context_len: int
    visible_context_len: int

    logical_blocks: list[int]
    visible_entries: list[VisibleBlockEntry]

    write_tail_ref: SequenceKVRef | None
    protected_sink_blocks: int
    protected_recent_blocks: int
    unfinished_prefill: bool
    shared_prefix_refcnt: int


PhysicalBlockTable = dict[int, PhysicalBlockMeta]
SequenceKVRefTable = dict[tuple[int, int], SequenceKVRef]
LogicalBlockTable = dict[int, list[SequenceKVRef]]
VisibleBlockTable = dict[int, list[VisibleBlockEntry]]
WriteBlockTable = dict[int, WriteCursor]
SlotMapping = list[tuple[int, int, int]]  # batch_token -> (seq_id, logical_block_id, full_slot)
```

## PhysicalBlockMeta + SequenceKVRef 双表拆分

shared-prefix block 不能被建模成单一 `seq_id` owner。必须使用：

```text
PhysicalBlockMeta.owner_refs = {(seq_a, logical_block_x), (seq_b, logical_block_y), ...}
PhysicalBlockMeta.ref_count = len(owner_refs)
SequenceKVRef(seq_id, logical_block_id) -> storage_id
```

这样做解决三个问题：

1. shared-prefix block 的 ownership 是多序列集合，不是单 owner。
2. QUANT/EVICT transition 必须原子更新所有受影响 SequenceKVRef 与 VisibleBlockTable。
3. per-sequence protection，例如 `is_sink/is_recent/is_inflight_write`，不能塞进 physical block 单表里，否则 shared block 会产生冲突。

## KV 显存预算公式

必须保留并在 P3 实装：

```text
full_block_bytes =
  2 * num_layers * block_size * num_kv_heads * head_dim * full_dtype_bytes

quant_block_bytes =
  2 * num_layers * block_size * num_kv_heads * head_dim * int8_bytes

scale_block_bytes =
  2 * num_layers * num_kv_heads * scale_dtype_bytes

total_kv_budget_bytes =
  full_pool_bytes
  + quant_pool_bytes
  + scale_bytes
  + scratch_budget
  + metadata_budget
```

等价展开：

```text
total_kv_budget_bytes =
  full_pool_blocks * full_block_bytes
  + quant_pool_blocks * quant_block_bytes
  + quant_pool_blocks * scale_block_bytes
  + max_mixed_kv_scratch_bytes
  + kv_metadata_budget_bytes
```

重要约束：

```text
quant_pool_bytes 从 total_kv_budget_bytes 中切走；
quant pool 不是 full pool 之外的额外追加；
headline 不使用 raw peak VRAM 作为唯一收益指标；
raw peak VRAM 只作为独立观察项。
```

## 端到端数据流

### Step 0：配置与预算初始化

```text
Config
  -> total_kv_budget_bytes
  -> full_pool_blocks
  -> quant_pool_blocks
  -> scale storage
  -> scratch budget
  -> metadata budget
```

初始化时输出：

```json
{
  "total_kv_budget_bytes": 10000000000,
  "full_pool_blocks": 2048,
  "quant_pool_blocks": 1024,
  "scale_bytes": 33554432,
  "scratch_budget": 268435456,
  "metadata_budget": 67108864,
  "effective_kv_capacity_bytes": 0
}
```

### Step 1：scheduler 生成 BatchPlan

```text
Request queue
  -> AdmissionController
  -> MemoryAwareScheduler
  -> BatchPlan
```

P1 中 `BatchPlan` 只能是 homogeneous：

```text
BatchPlan.kind = "decode" OR "prefill"
```

P4a 后 decode path 可以读取 mixed-KV；P4b 后 prefill-prefix / unfinished-prefill 可以读取 mixed-KV。

### Step 2：slot_mapping 只写 FULL

```text
BatchPlan
  -> allocate_write_blocks()
  -> WriteBlockTable
  -> slot_mapping
```

`slot_mapping` 中每个 entry 必须满足：

```text
target PhysicalBlockMeta.state == FULL
target ref.is_inflight_write == True
```

### Step 3：VisibleBlockTable 作为 read view

在 batch 执行前冻结 visible view：

```text
SequenceKVRefTable + PhysicalBlockTable
  -> build_visible_table(seq_id)
  -> VisibleBlockTable[seq_id]
```

规则：

```text
FULL  -> visible read entry, direct full_block_id
QUANT -> visible read entry, quant_block_id + scales
EVICT -> logical span retained, attention skip
```

P4a/P4b backend 能识别 EVICT，但 P5 前 policy 不生成 EVICT。

### Step 4：ModelRunner 执行

```text
ModelRunner.run_step(batch_plan, step_context)
  -> embedding / layers
  -> attention backend
  -> write new KV to FULL slots
  -> logits / sampling
```

attention backend：

```text
full-only flags off:
  read original FULL table

mixed fallback on:
  read VisibleBlockTable
  FULL: direct gather
  QUANT: dequant-to-scratch
  EVICT: skip
```

### Step 5：quantize_from_full 两阶段提交

在 step boundary，reclaim planner 选出 FULL→QUANT candidates：

```text
snapshot
  -> compute_block_score
  -> watermark+hysteresis target
  -> ReclaimPlan.quantize
```

两阶段提交：

```text
Phase A: prepare
  allocate quant slot
  write int8 K/V
  write scales
  validate quant output

Phase B: commit
  atomically update PhysicalBlockMeta
  atomically update SequenceKVRefTable if storage view changes
  rebuild / update VisibleBlockTable
  mark old FULL block releasable
  release FULL block to free pool

Rollback:
  if any prepare/commit failure:
    free quant slot if allocated
    keep PhysicalBlockMeta.state = FULL
    keep SequenceKVRef pointing to original storage
    do not release FULL block
```

### Step 6：QUANT dequant-to-scratch 与 workspace planning

```text
BatchPlan
  -> count visible QUANT blocks
  -> estimate scratch bytes
  -> MixedKVWorkspacePlan
```

如果超过预算：

```text
decode batch:
  shrink batch or fallback to full-only if no QUANT committed

prefill batch:
  split chunk before execution

runtime:
  never allocate unbounded scratch
```

### Step 7：释放 full block 回收

FULL block 只能在 commit 成功后回到 free pool：

```text
quantize commit success
  -> remove full_block_id from PhysicalBlockMeta
  -> free_full_block(full_block_id)
  -> scheduler snapshot sees higher free_full_blocks
  -> admission can admit/shrink less aggressively
```

这一步是 B3 的关键：KV savings 必须变成 scheduler/admission 可用容量，而不是只停留在统计指标。

# 策略与分层设计

## ARKV-Inspired Policy 总体原则

本计划不要求逐公式复刻 ARKV 论文，但必须保留关键思想：

```text
tri-state:
  FULL / QUANT / EVICT

budget-aware:
  reclaim 由 full block pressure 与 total KV budget 驱动

importance-aware:
  recent / sink / shared-prefix / attention mass / layer sensitivity

layer-aware-upgradeable:
  MVP 可 page-global tiering
  Full target 保留 per-layer sensitivity 扩展
```

## compute_block_score 伪代码

必须保留并作为 `kv_policy.py` 中的核心 policy 函数。

```python
def compute_block_score(ref: SequenceKVRef,
                        block: PhysicalBlockMeta,
                        seq_state: SequenceKVState,
                        layer_sensitivity: float = 1.0,
                        cfg: PolicyConfig = CFG) -> float:
    if ref.is_inflight_write:
        return float("inf")
    if block.is_shared_prefix and block.ref_count > 1:
        return float("inf")
    if ref.is_sink and not cfg.allow_sink_reclaim:
        return float("inf")
    if ref.is_recent and not cfg.allow_recent_reclaim:
        return float("inf")

    recency = exp(-distance_from_tail(ref, seq_state) / cfg.recency_tau)
    sink = 1.0 if ref.is_sink else 0.0
    shared = min(log2(block.ref_count + 1) / log2(cfg.ref_norm + 1), 1.0)
    attn = ref.attention_mass_ema if cfg.enable_attention_mass else 0.0

    score = (
        cfg.w_recency * recency +
        cfg.w_sink * sink +
        cfg.w_shared * shared +
        cfg.w_attn * attn
    )
    return score * layer_sensitivity
```

默认权重：

```text
w_recency = 0.35
w_sink    = 0.20
w_shared  = 0.25
w_attn    = 0.20
```

## reclaim_blocks 伪代码

watermark+hysteresis 必须保留，且 reclaim 必须拉到目标水位，而不是“缺多少补多少”。

```python
def select_blocks_to_quantize(candidates, required_full_equiv, quant_pool_free):
    eligible = [
        c for c in candidates
        if c.block.state == KVBlockState.FULL and can_quantize(c)
    ]
    eligible.sort(
        key=lambda c: (
            compute_block_score(c.ref, c.block, c.seq_state),
            c.block.last_access_step,
        )
    )

    picked, reclaimed = [], 0
    for candidate in eligible:
        if len(picked) >= quant_pool_free:
            break
        gain = full_equiv_gain_if_quantized(candidate.block)
        if gain <= 0:
            continue
        picked.append(candidate)
        reclaimed += gain
        if reclaimed >= required_full_equiv:
            break

    return picked, reclaimed


def select_blocks_to_evict(candidates, required_full_equiv, seq_states, cfg):
    eligible = [
        c for c in candidates
        if can_evict(c, seq_states[c.ref.seq_id], cfg)
    ]
    eligible.sort(
        key=lambda c: (
            compute_block_score(c.ref, c.block, c.seq_state),
            0 if c.block.state == KVBlockState.QUANT else 1,
            c.block.last_access_step,
        )
    )

    picked, reclaimed = [], 0
    for candidate in eligible:
        gain = full_equiv_gain_if_evicted(candidate.block)
        if gain <= 0:
            continue
        picked.append(candidate)
        reclaimed += gain
        if reclaimed >= required_full_equiv:
            break

    return picked, reclaimed


def reclaim_blocks(snapshot, required_full_equiv, cfg):
    if snapshot.free_full_equiv >= required_full_equiv and snapshot.free_ratio > cfg.mid_wm:
        return ReclaimPlan.none()

    target = max(required_full_equiv, hysteresis_target(snapshot, cfg))

    quant_plan, q_gain = select_blocks_to_quantize(
        snapshot.candidates,
        target,
        snapshot.free_quant_blocks,
    )

    remaining = max(0, target - q_gain)

    evict_plan, e_gain = [], 0
    if remaining > 0 and cfg.enable_evict and snapshot.free_ratio <= cfg.low_wm:
        evict_plan, e_gain = select_blocks_to_evict(
            snapshot.candidates,
            remaining,
            snapshot.seq_states,
            cfg,
        )

    return ReclaimPlan(
        quantize=quant_plan,
        evict=evict_plan,
        total_gain=q_gain + e_gain,
    )
```

## Watermark + Hysteresis Reclaim

配置建议：

```python
@dataclass
class ReclaimWatermarks:
    high_wm: float = 0.30
    mid_wm: float = 0.20
    low_wm: float = 0.10
    target_after_mid_pressure: float = 0.25
    target_after_low_pressure: float = 0.20
```

行为：

```text
free_full_ratio > high_wm:
  no reclaim

mid_wm >= free_full_ratio > low_wm:
  FULL -> QUANT only
  reclaim until target_after_mid_pressure

free_full_ratio <= low_wm:
  FULL -> QUANT first
  if still insufficient and P5 gate passed:
    QUANT -> EVICT
  direct FULL -> EVICT only if allow_direct_full_evict=True
```

## EVICT 策略硬约束

EVICT 必须遵循：

```text
默认只优先选择已 QUANT 的低分 block
allow_direct_full_evict=False
FULL -> EVICT 直跳只允许：
  - 极端压力
  - P5 quality gate 已通过
  - enable_direct_full_evict=True
  - no QUANT candidate can satisfy target
```

P5 前：

```text
policy 不允许生成任何 EVICT entry
attention backend 可识别 EVICT 并跳过
benchmark 不报告 evicted_block_ratio，除非 optional B2c
```

## naive policy baseline 内建

必须内建两个 baseline，不允许 benchmark 阶段临时拼：

```python
class ReclaimPolicyName(Enum):
    NAIVE_AGE_Q8 = "naive_age_q8"
    NAIVE_RECENT_GUARD_EVICT = "naive_recent_guard_evict"
    ARKV_Q8 = "arkv_q8"
    ARKV_Q8_EVICT = "arkv_q8_evict"
```

### naive_age_q8

```text
候选：
  FULL block
  非 recent
  非 sink
  非 inflight write
  默认跳过 shared-prefix

排序：
  oldest first by last_access_step

动作：
  FULL -> QUANT
```

### naive_recent_guard_evict

```text
候选：
  QUANT block first
  非 recent
  非 sink
  非 inflight write
  默认跳过 shared-prefix

排序：
  oldest first

动作：
  QUANT -> EVICT
  direct FULL -> EVICT only when explicitly enabled
```

B2a/B2b 必须回答：

```text
收益来自任何 Q8 tiering，
还是来自 importance-aware ARKV policy 本身？
```

## Scheduler and Admission Design

调度策略：

```text
decode-first
adaptive chunked prefill
short/long lane
starvation guard
future decode reserve
reclaim-aware admission
```

P1 只输出 homogeneous batch：

```text
BatchPlan.kind == DECODE_ONLY
or
BatchPlan.kind == PREFILL_ONLY
```

P4a/P4b 后再扩展 mixed read，不把 P1 变成 true mixed batch 阶段。

## decide_admission 伪代码

```python
def decide_admission(req, sched_snapshot, kv_snapshot, cfg):
    want_chunk = choose_prefill_chunk(req, sched_snapshot, kv_snapshot, cfg)
    need = estimate_full_blocks(req, want_chunk) + estimate_future_decode_reserve(req, cfg)

    if kv_snapshot.free_full_equiv >= need:
        return AdmitDecision.admit(chunk=want_chunk)

    reclaimable = kv_snapshot.conservative_reclaimable_full_equiv
    if kv_snapshot.free_full_equiv + reclaimable >= need:
        return AdmitDecision.admit_after_reclaim(chunk=want_chunk)

    shrunk = shrink_chunk_until_fit(req, sched_snapshot, kv_snapshot, cfg)
    if shrunk is not None:
        return AdmitDecision.shrink(chunk=shrunk)

    if cfg.temporary_backpressure:
        return AdmitDecision.defer()

    return AdmitDecision.reject_temp()
```

## Admission 状态

```python
class AdmitDecisionKind(Enum):
    ADMIT = "admit"
    ADMIT_AFTER_RECLAIM = "admit_after_reclaim"
    SHRINK = "shrink"
    DEFER = "defer"
    REJECT_TEMP = "reject_temp"


@dataclass
class AdmitDecision:
    kind: AdmitDecisionKind
    chunk_tokens: int | None
    required_full_blocks: int
    reclaim_plan_id: str | None
    reason: str
```

## Scheduler Chunk 策略

chunk size 由三类压力共同决定：

```text
decode_backlog_pressure:
  decode backlog 越高，prefill chunk 越小

free_full_block_ratio:
  free full blocks 越紧张，prefill chunk 越小

reclaim_cost_estimate:
  reclaim 成本越高，prefill chunk 越保守
```

starvation guard：

```text
每个 waiting request 维护 skip_count
skip_count > threshold:
  至少分配 min_prefill_chunk
```

short/long lane：

```text
short prompt lane:
  优先降低 TTFT

long prompt lane:
  受 max_long_partial_prefills 限制
  防止长 prompt 占满 token budget
```

# Attention Backend 与 Kernel 规划

## Backend 分层

```text
Layer 0: full_only_backend
  - legacy path
  - default path
  - all optimizer flags off

Layer 1: mixed_kv_fallback_backend
  - MVP critical path
  - FULL direct read
  - QUANT dequant-to-scratch
  - EVICT skip support
  - P4a decode-only
  - P4b prefill-prefix / unfinished-prefill

Layer 2: accelerated backend
  - P6a Triton gather/dequant
  - P6b decode-only mixed-KV attention kernel
  - optional block_attn_mass
  - runtime fallback to Layer 1
```

## mixed-KV fallback read 规则

```text
for entry in VisibleBlockTable[seq_id]:
  if entry.state == FULL:
    gather full cache view

  elif entry.state == QUANT:
    dequant quant cache into scratch
    gather scratch view

  elif entry.state == EVICT:
    skip attention read
    preserve logical span accounting
```

P4a/P4b 中：

```text
backend can parse EVICT
policy cannot generate EVICT
```

## Workspace Planning

```python
@dataclass
class MixedKVWorkspacePlan:
    batch_id: int
    num_visible_full_blocks: int
    num_visible_quant_blocks: int
    num_visible_evict_entries: int

    per_layer_scratch_bytes: int
    total_scratch_bytes: int
    max_allowed_scratch_bytes: int

    requires_split: bool
    split_reason: str | None
```

规则：

```text
workspace_plan.total_scratch_bytes <= max_mixed_kv_scratch_bytes

if overflow:
  decode:
    shrink batch or disable runtime QUANT commit before execution

  prefill:
    split chunk before execution
```

## P4a decode-only mixed-KV fallback

目标：

```text
Decode attention reads visible FULL/QUANT entries.
FULL -> QUANT 后释放出来的 full blocks 能被复用。
B2/B3 闭环成立。
```

关键限制：

```text
only decode read
no true mixed prefill execution requirement
no EVICT policy generation
```

## P4b prefill-prefix / unfinished-prefill mixed read

目标：

```text
prefill 可以读取 QUANT prefix
unfinished-prefill 可以读取 QUANT blocks
slot_mapping 仍只写 FULL
unfinished-prefill 禁止 EVICT
```

P4b 失败不阻断：

```text
P4a headline
B2a/B2b/B3
MVP 关键闭环
```

## Triton-first, CUDA-optional Kernel Plan

Triton-first 必须保留。CUDA 只作为 optional future optimization，不作为 MVP 依赖。

### Kernel 1：Q8 quant/dequant

```text
Input:
  full K/V pages
  scale granularity
  block metadata

Output:
  int8 K/V pages
  k_scale / v_scale

Edge cases:
  partial block
  all-zero block
  fp16 / bf16 input
  head_dim 64 / 128
```

### Kernel 2：gather+dequant

```text
Input:
  quant cache
  visible entries
  scale tensors
  scratch pointer

Output:
  contiguous scratch K/V

Edge cases:
  ragged seq
  partial last block
  mixed visible FULL/QUANT
```

### Kernel 3：mixed-KV decode attention

```text
Input:
  q
  FULL cache
  QUANT cache
  visible entries
  seq lens
  head map

Output:
  attention output
  optional block_attn_mass

Initial support:
  decode-only
  causal
  single query
  head_dim 64 / 128
```

## Kernel 测试要求

所有 kernel 必须具备：

```text
torch reference
numerical diff tests
microbench
runtime fallback
shape support checks
```

标准命令：

```bash
python -m pytest tests/kernels/test_q8_kv.py -v
python -m pytest tests/kernels/test_gather_dequant.py -v
python -m pytest tests/kernels/test_mixed_kv_decode_attention.py -v
python benchmarks/microbench_q8_kv.py --dtype fp16 --head-dim 128 --block-size <calibrated_block_size>
python benchmarks/microbench_gather_dequant.py --block-size <calibrated_block_size> --quant-blocks 1024 --seq-len 4096
python benchmarks/microbench_mixed_kv_decode.py --head-dim 128 --block-size <calibrated_block_size> --kv-mix-ratio 0.5
```

# 实施阶段

## Phase Dependency Graph

```text
P-1 repo capability calibration
 |
 v
P0  baseline harness / flags / metrics
 |
 +--> P1  scheduler homogeneous batch
 |
 +--> P2  metadata tables + dry-run policy
       |
       v
P3  QUANT path + budget split + two-phase commit
       |
       v
P4a decode-only mixed-KV fallback  ---> MVP critical loop / B2-B3
       |
       v
P4b prefill-prefix / unfinished-prefill mixed read
       |
       v
P5  EVICT quality gate
       |
       v
P6a Triton gather/dequant
       |
       v
P6b decode-only mixed-KV attention kernel
       |
       v
P7  benchmark / ablation / release hardening / README
```

可并行项：

```text
P-1 必须先完成，冻结模型、block size、fallback reference 与 CUDA graph 策略。
P0 与部分 P2 metadata schema design 可并行。
P1 scheduler abstraction 与 P2 metadata dry-run 可部分并行，但 integration 依赖 P0 metrics。
P6a/P6b 的 torch reference 与 microbench scaffold 可在 P4a 后提前准备。
P7 report 脚本骨架可在 P0 后并行开发。
```

严格串行项：

```text
P-1 -> P0 -> P3 -> P4a -> P4b -> P5 -> P6a -> P6b -> P7

原因：
  P-1 必须先确定当前 repo 可执行边界，否则 P0 benchmark 与 P3/P4a kernel/fallback 目标会漂移。
  P4a 必须依赖 P3 的真实 FULL->QUANT commit。
  P4b 必须依赖 P4a 的 fallback read correctness。
  P5 必须依赖 P4a/P4b backend 能识别 EVICT。
  P6a 必须以 P4 fallback 为 reference。
  P6b 必须以 P4/P6a correctness 为 fallback。
  P7 headline 必须在 B0-B5 全部可跑后完成。
```

---

## P-1：repo capability calibration / smoke spike

### Objective

在进入 P0 之前，冻结当前 Nano-vLLM 的可执行边界：本仓模型支持、KV block size、mixed-KV fallback reference、CUDA graph 策略、本地显存与 benchmark 默认模型。P-1 只做最小 smoke / spike，不引入 optimizer 行为，不改变 full-only baseline。

### Dependencies / Parallelism

```text
Dependencies:
  none

Can run in parallel with:
  documentation cleanup only

Must finish before:
  P0 benchmark harness
  P1 scheduler validation config
  P3 quant pool/block-size decisions
  P4a mixed-KV fallback implementation
```

### Files to Add / Modify

```text
Modify:
  inference_systems_code_plan.md only if calibration changes assumptions
  nanovllm/config.py only if adding non-behavioral validation/logging

Add:
  tests/bench/test_capability_smoke.py
  benchmarks/capability_probe.py
```

### Calibration Questions

```text
1. Which Qwen3 model paths are locally available and loadable on target GPU?
2. What is the largest Qwen3 model / max_model_len / max_num_seqs tuple that can run B0 smoke on RTX 4070 Ti 12GB?
3. Is kvcache_block_size below 256 supported by current config + flash-attn path? If not, benchmark default remains 256.
4. Can a decode-only torch reference materialize visible FULL KV into scratch and match the existing full-only decode output?
5. Does mixed-KV fallback require enforce_eager=True until P6b fused kernel? Default answer is yes unless proven otherwise.
6. Are Llama/TinyLlama/OPT runners available? If not, they stay out of formal ablation.
```

### Key Implementation Steps

1. Run import/config smoke without loading a model.
2. Probe local model path availability and record selected Qwen3 benchmark model.
3. Probe `kvcache_block_size` support for 256 and any smaller candidate only if current config allows it.
4. Build a tiny decode-only materialization reference: FULL cache -> scratch -> attention reference, compared against current full-only decode for controlled tensors.
5. Decide CUDA graph policy: `enable_mixed_kv_fallback=True` implies `enforce_eager=True` until P6b unless a graph-safe path is explicitly validated.
6. Write `benchmarks/capability_probe.py` output JSON with chosen model, block size, max context, eager policy, and skipped assumptions.

### Feature Flags

```text
No optimizer feature flags may be enabled in P-1.
```

### Validation Commands

```bash
python -c "import nanovllm; print(nanovllm.__all__)"
python benchmarks/capability_probe.py --dry-run --output-json results/p_minus_1_capability.json
python -m pytest tests/bench/test_capability_smoke.py -v
```

### Definition of Done

- Formal benchmark model is Qwen3-first and recorded in `results/p_minus_1_capability.json`.
- Formal benchmark block size is recorded; default remains 256 unless smaller sizes are proven valid.
- Mixed-KV fallback reference has a minimal tensor-level parity test or is explicitly marked blocked before P4a.
- `enable_mixed_kv_fallback=True` defaults to eager execution until P6b graph safety is proven.
- Any non-Qwen3 model use is marked optional unless model adapter support exists.

### Failure Modes and Rollback

| Failure mode | Rollback operation | Impact on later phases |
|---|---|---|
| No local model can load | Keep dry-run benchmark only and require model path before B0 | P0 dry-run can proceed, real B0 blocked |
| Smaller block sizes fail | Freeze benchmark block size at 256 | Kernel/microbench commands use calibrated block size |
| Decode materialization reference mismatches | Keep mixed-KV fallback unimplemented and investigate before P3 runtime release | P4a blocked |
| CUDA graph cannot handle dynamic mixed visible tables | Force `enforce_eager=True` for mixed fallback | P4a can proceed, P6b owns graph recovery |
| 7B/8B does not fit 12GB | Use smaller Qwen3 + longer context/concurrency sweep | Headline remains valid if KV pressure is demonstrated |

### Estimated Days

```text
1-2 days
```

### Codex Implementation Prompt

```text
Calibrate the current Nano-vLLM repository before feature work. Produce a dry-run
capability probe that records the formal Qwen3 benchmark model, supported KV block
size, max smoke config, mixed-KV fallback reference status, and CUDA graph/eager
policy. Do not enable optimizer behavior. If smaller block sizes or non-Qwen3
models are unsupported, mark them optional rather than using them in formal
benchmark commands.
```

---

## P0：baseline benchmark harness + feature flag scaffolding + metrics skeleton

### Objective

建立统一 benchmark/metrics 基线，不改变任何算法路径。P0 必须使用 P-1 冻结的模型、block size、eager/graph 策略产出可跑、含 dry-run 的 benchmark 脚本，并记录 B0 baseline 数据。

### Dependencies / Parallelism

```text
Dependencies:
  P-1 capability calibration

Can run in parallel with:
  P2 metadata schema design
  P7 report schema draft

Must finish before:
  P1 validation
  all ablation reporting
```

### Files to Add / Modify

```text
Modify:
  nanovllm/config.py
  nanovllm/engine/llm_engine.py
  nanovllm/engine/scheduler.py
  nanovllm/engine/block_manager.py

Add:
  benchmarks/benchmark_serving.py
  benchmarks/workloads/scheduler_stress.py
  benchmarks/workloads/long_context_pressure.py
  benchmarks/workloads/shared_prefix.py
  benchmarks/workloads/quality_passkey.py
  benchmarks/report.py
  tests/bench/test_metrics_smoke.py
```

### Public Interfaces / Function Signatures

```python
@dataclass
class RequestMetrics:
    request_id: str
    arrival_ts: float
    scheduled_ts: float | None
    first_token_ts: float | None
    finish_ts: float | None
    prompt_tokens: int
    output_tokens: int
    oom: bool
    error: str | None


@dataclass
class KVPoolMetrics:
    step: int
    free_full_blocks: int
    active_full_blocks: int
    active_quant_blocks: int
    evicted_blocks: int
    free_full_block_ratio: float
    effective_kv_memory_bytes: int
    raw_peak_vram_bytes: int


def record_request_event(
    request_id: str,
    event: Literal["arrival", "scheduled", "first_token", "finish", "oom"],
    timestamp: float,
) -> None:
    """Record request lifecycle event.
    Raises:
        MetricsStateError: if event order is invalid.
    """


def collect_kv_pool_metrics(step: int) -> KVPoolMetrics:
    """Collect current KV pool metrics from block manager.
    Raises:
        MetricsUnavailableError: if block manager state cannot be read.
    """


def run_serving_benchmark(
    workload_name: str,
    model: str,
    concurrency: int,
    max_requests: int,
    output_json: str,
    dry_run: bool = False,
) -> dict:
    """Run serving benchmark or dry-run workload generation.
    Raises:
        BenchmarkConfigError: invalid workload or model config.
        BenchmarkRuntimeError: benchmark fails after startup.
    """
```

### Key Implementation Steps

1. Add feature flag scaffolding with all optimizer flags defaulting to `False`.
2. Add metrics hooks for request arrival, scheduled time, first token, finish time, OOM, queue depth, full block stats.
3. Implement dry-run benchmark mode that generates workload and validates config without loading model.
4. Implement B0 baseline command path.
5. Emit JSON and CSV reports with fixed schema.

### Feature Flags

```text
enable_metrics_hooks=False by default
enable_memory_aware_optimizer=False by default
all optimizer flags=False
```

P0 may enable metrics explicitly in benchmark command, but must not change serving semantics.

### Validation Commands

```bash
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 1 --dry-run --output-json /tmp/b0_dryrun.json
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 16 --output-json results/b0_scheduler_stress.json
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 8 --output-json results/b0_long_context.json
python -m pytest tests/bench/test_metrics_smoke.py -v
```

### Definition of Done

- Dry-run benchmark succeeds without loading model.
- B0 baseline produces JSON and CSV with TTFT、TPOT、throughput、OOM、free full blocks、raw peak VRAM fields。
- Closing all optimizer flags preserves baseline output path.
- Metrics hooks do not mutate scheduler decisions.
- At least one B0 scheduler_stress run is saved under `results/`.

### Failure Modes and Rollback

| Failure mode | Rollback operation | Impact on later phases |
|---|---|---|
| Metrics hook changes timing too much | Disable `enable_metrics_hooks`; keep dry-run harness | P1/P2 can continue, but ablation cannot be accepted |
| JSON schema unstable | Freeze schema in `benchmarks/report.py` and regenerate | Later benchmark parsers blocked until fixed |
| Benchmark OOM before workload starts | Lower default dry-run / smoke config only; do not alter algorithm | P0 valid for dry-run, real B0 needs documented smaller config |
| Feature flags accidentally default on | Hard reset defaults to False and add regression test | Blocks all later merge gates |

### Estimated Days

```text
2-4 days
```

### Codex Implementation Prompt

```text
Implement baseline observability for Nano-vLLM without changing serving semantics.
Add request lifecycle timestamps, KV/block/memory stats, feature flag scaffolding,
and a repeatable benchmark harness that outputs TTFT/TPOT/throughput/VRAM/OOM
in JSON and CSV. Include dry-run mode. Keep all optimizer behavior disabled by
default and preserve the full-only baseline path.
```

---

## P1：memory-aware scheduler，homogeneous batch only

### Objective

引入 task abstraction、decode-first + adaptive chunked prefill、short/long lanes、starvation guard 与 reclaim-aware admission 的框架。但 P1 只输出 homogeneous batch，不要求 ModelRunner 执行 true mixed batch。

### Dependencies / Parallelism

```text
Dependencies:
  P0 metrics and feature flag scaffolding

Can run in parallel with:
  P2 metadata dry-run internals, after interface alignment

Must finish before:
  B1 scheduler-only ablation
  P3 admission integration
```

### Files to Add / Modify

```text
Modify:
  nanovllm/config.py
  nanovllm/engine/scheduler.py
  nanovllm/engine/sequence.py
  nanovllm/engine/llm_engine.py
  nanovllm/engine/model_runner.py

Add:
  nanovllm/engine/tasks.py
  nanovllm/engine/admission.py
  nanovllm/engine/scheduler_metrics.py
  tests/engine/test_scheduler_tasks.py
  tests/engine/test_admission.py
```

### Public Interfaces / Function Signatures

```python
class TaskKind(Enum):
    DECODE = "decode"
    PREFILL = "prefill"


@dataclass
class DecodeTask:
    seq_id: int
    request_id: str
    num_tokens: int = 1


@dataclass
class PrefillTask:
    seq_id: int
    request_id: str
    start_pos: int
    chunk_tokens: int
    is_long_prefill: bool
    skip_count: int


@dataclass
class BatchPlan:
    batch_id: int
    kind: Literal["decode", "prefill"]
    decode_tasks: list[DecodeTask]
    prefill_tasks: list[PrefillTask]
    token_budget: int
    slot_mapping: SlotMapping | None
    visible_block_tables: dict[int, list[VisibleBlockEntry]] | None
    workspace_plan: MixedKVWorkspacePlan | None


def build_batch_plan(
    waiting: list[Sequence],
    running: list[Sequence],
    sched_snapshot: SchedulerSnapshot,
    kv_snapshot: KVSnapshot,
    cfg: SchedulerConfig,
) -> BatchPlan:
    """Build a homogeneous decode-only or prefill-only batch plan.
    Raises:
        SchedulerInvariantError: if mixed decode/prefill tasks are emitted in P1 mode.
    """


def choose_prefill_chunk(
    req: Sequence,
    sched_snapshot: SchedulerSnapshot,
    kv_snapshot: KVSnapshot,
    cfg: SchedulerConfig,
) -> int:
    """Choose adaptive prefill chunk size.
    Raises:
        AdmissionError: if request cannot be scheduled or shrunk.
    """


def estimate_future_decode_reserve(
    req: Sequence,
    cfg: SchedulerConfig,
) -> int:
    """Estimate full blocks reserved for future decode.
    Raises:
        ValueError: invalid max_new_tokens or block size.
    """


def decide_admission(
    req: Sequence,
    sched_snapshot: SchedulerSnapshot,
    kv_snapshot: KVSnapshot,
    cfg: SchedulerConfig,
) -> AdmitDecision:
    """Return ADMIT / ADMIT_AFTER_RECLAIM / SHRINK / DEFER / REJECT_TEMP.
    Raises:
        AdmissionStateError: inconsistent request state.
    """
```

### Key Implementation Steps

1. Add `DecodeTask`、`PrefillTask`、`BatchPlan`.
2. Refactor scheduler to emit homogeneous decode-only or prefill-only batch.
3. Add decode-first selection.
4. Add short/long prefill lanes.
5. Add starvation guard via per-request `skip_count`.
6. Add admission decisions: admit、admit_after_reclaim、shrink、defer、reject_temp.
7. Keep legacy runner path unchanged for actual execution.

### Feature Flags

```text
enable_memory_aware_scheduler=False
enable_admission_controller=False
```

### Validation Commands

```bash
python -m pytest tests/engine/test_scheduler_tasks.py -v
python -m pytest tests/engine/test_admission.py -v
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 16 --output-json results/b1_scheduler_only.json --enable-memory-aware-scheduler --enable-admission-controller
python benchmarks/benchmark_serving.py --workload shared_prefix --concurrency 16 --output-json results/b1_shared_prefix.json --enable-memory-aware-scheduler --enable-admission-controller
```

### Definition of Done

- P1 scheduler never emits true mixed decode/prefill batch.
- Closing P1 flags restores legacy scheduler.
- B1 can run on scheduler_stress workload and produce comparable metrics against B0.
- Admission shrink/defer/reject counts are recorded.
- Starvation guard has deterministic unit tests.

### Failure Modes and Rollback

| Failure mode | Rollback operation | Impact on later phases |
|---|---|---|
| Scheduler emits mixed batch in P1 | Add invariant check and force split into decode-only or prefill-only | Blocks P1 merge |
| TTFT worsens due to over-shrinking | Tune chunk policy behind flag; keep legacy scheduler fallback | B1 may fail but P2/P3 metadata can continue |
| Admission over-defer reduces throughput | Disable `enable_admission_controller` while keeping task abstraction | P3 integration delayed |
| Starvation guard breaks decode-first | Cap forced prefill chunks; add regression test | B1 headline blocked until stable |

### Estimated Days

```text
4-7 days
```

### Codex Implementation Prompt

```text
Refactor the scheduler around DecodeTask, PrefillTask, BatchPlan, and an explicit
AdmissionController. Preserve legacy full-only execution. Implement decode-first
scheduling, adaptive chunked prefill, short/long lanes, starvation guard, and
KV-reserve-aware admit/shrink/defer decisions. In P1, BatchPlan must be
homogeneous: all decode or all prefill. Do not require true mixed batch execution.
All new behavior must be behind default-off feature flags.
```

---

## P2：metadata tables，PhysicalBlockMeta + SequenceKVRef，三表分离

### Objective

实现 PhysicalBlockMeta + SequenceKVRef 双表拆分，并保持 logical / physical / visible 三表分离。P2 只做 metadata truth 与 policy dry-run，不改变 runtime attention 语义。

### Dependencies / Parallelism

```text
Dependencies:
  P0 feature flags and metrics schema

Can run in parallel with:
  P1 scheduler implementation after shared KVSnapshot interface is agreed

Must finish before:
  P3 quantize_from_full
  P4a visible read path
```

### Files to Add / Modify

```text
Modify:
  nanovllm/engine/block_manager.py
  nanovllm/engine/sequence.py
  nanovllm/config.py

Add:
  nanovllm/engine/kv_meta.py
  nanovllm/engine/kv_policy.py
  nanovllm/engine/visible_tables.py
  tests/engine/test_kv_meta.py
  tests/engine/test_visible_tables.py
  tests/engine/test_kv_policy_dry_run.py
```

### Public Interfaces / Function Signatures

```python
def register_full_block(
    seq_id: int,
    logical_block_id: int,
    full_block_id: int,
    logical_start: int,
    logical_end: int,
    prefix_hash: int | None,
    is_shared_prefix: bool,
) -> int:
    """Register a FULL physical block and return storage_id.
    Raises:
        MetadataConsistencyError: duplicate logical ref or invalid block span.
    """


def add_owner_ref(
    storage_id: int,
    seq_id: int,
    logical_block_id: int,
) -> None:
    """Attach a sequence logical block to a physical block.
    Raises:
        MetadataConsistencyError: storage_id does not exist or duplicate owner.
    """


def build_visible_block_table(
    seq_id: int,
    logical_refs: list[SequenceKVRef],
    physical_table: PhysicalBlockTable,
    cfg: VisibleTableConfig,
) -> list[VisibleBlockEntry]:
    """Build attention read view in logical order.
    Raises:
        VisibleTableError: missing physical block, non-monotonic logical span.
    """


def validate_kv_tables(
    physical_table: PhysicalBlockTable,
    ref_table: SequenceKVRefTable,
    visible_table: VisibleBlockTable,
) -> None:
    """Validate logical / physical / visible table invariants.
    Raises:
        MetadataConsistencyError: invariant violation.
    """


def plan_reclaim_dry_run(
    snapshot: KVSnapshot,
    required_full_equiv: int,
    policy_name: ReclaimPolicyName,
    cfg: PolicyConfig,
) -> ReclaimPlan:
    """Compute reclaim plan without mutating physical storage.
    Raises:
        PolicyError: invalid policy config.
    """
```

### Key Implementation Steps

1. Add `KVBlockState`、`PhysicalBlockMeta`、`SequenceKVRef`、`SequenceKVState`、`VisibleBlockEntry`.
2. Introduce `LogicalBlockTable`、`PhysicalBlockTable`、`SequenceKVRefTable`、`VisibleBlockTable`、`WriteBlockTable`.
3. Ensure shared-prefix block uses `owner_refs` and `ref_count`.
4. Implement table validators.
5. Implement `compute_block_score` and dry-run reclaim planning.
6. Emit metrics: protected ratio、candidate count、conservative reclaimable blocks.

### Feature Flags

```text
enable_arkv_metadata=False
enable_arkv_policy_dry_run=False
```

### Validation Commands

```bash
python -m pytest tests/engine/test_kv_meta.py -v
python -m pytest tests/engine/test_visible_tables.py -v
python -m pytest tests/engine/test_kv_policy_dry_run.py -v
python benchmarks/benchmark_serving.py --workload shared_prefix --concurrency 16 --output-json results/p2_metadata_dryrun.json --enable-arkv-metadata --enable-arkv-policy-dry-run
```

### Definition of Done

- PhysicalBlockMeta + SequenceKVRef 双表存在并通过 shared-prefix tests。
- logical / physical / visible 三表分离存在并有 invariant tests。
- `slot_mapping` 与 `VisibleBlockTable` 不混用。
- dry-run reclaim plan deterministic。
- Runtime output 与 full-only baseline 一致。

### Failure Modes and Rollback

| Failure mode | Rollback operation | Impact on later phases |
|---|---|---|
| Shared-prefix ref_count 不一致 | Disable metadata flag; add owner_refs reconstruction test | P3 blocked |
| visible_context_len 与 logical_context_len 混用 | Add explicit type/field checks and invariant tests | P4a blocked |
| Dry-run policy mutates state | Freeze snapshot objects or deep-copy before planning | P2 merge blocked |
| Metadata overhead too high | Optimize tables, but do not merge PhysicalBlockMeta and SequenceKVRef | Later perf may be delayed, correctness preserved |

### Estimated Days

```text
5-8 days
```

### Codex Implementation Prompt

```text
Add tri-state KV metadata and separate logical / physical / visible / write views.
Implement PhysicalBlockMeta with owner_refs/ref_count/state and SequenceKVRef with
per-sequence logical_block_id -> storage_id plus protection flags. Implement
visible table construction and ARKV-inspired scoring in dry-run mode only. Do not
change runtime attention semantics. Add invariant tests for shared-prefix, sink,
recent, inflight-write, logical_context_len, and visible_context_len separation.
```

---

## P3：QUANT path，两阶段提交，统一 KV budget 切分

### Objective

实现 FULL→QUANT 真实物理迁移链路、quant pool、scale storage、scratch budget 与 `quantize_from_full` 两阶段提交。按 `total_kv_budget_bytes` 公式切分 full pool 和 quant pool。P3 可以先 shadow / controlled runtime，但不得在没有 mixed-KV read path 时释放 FULL 给 serving 复用。

### Dependencies / Parallelism

```text
Dependencies:
  P-1 calibrated block size and formal benchmark model
  P2 metadata tables
  P0 metrics schema

Can run in parallel with:
  P4a interface draft, but not runtime integration

Must finish before:
  P4a decode-only mixed-KV fallback
```

### Files to Add / Modify

```text
Modify:
  nanovllm/config.py
  nanovllm/engine/block_manager.py
  nanovllm/engine/model_runner.py

Add:
  nanovllm/engine/arkv_kv_manager.py
  nanovllm/engine/quant_cache.py
  nanovllm/kernels/q8_kv.py
  tests/engine/test_quant_commit.py
  tests/kernels/test_q8_kv.py
```

### Public Interfaces / Function Signatures

```python
@dataclass
class KVCacheBudget:
    total_kv_budget_bytes: int
    full_pool_bytes: int
    quant_pool_bytes: int
    scale_bytes: int
    scratch_budget: int
    metadata_budget: int
    full_pool_blocks: int
    quant_pool_blocks: int


def compute_kv_cache_budget(
    model_cfg: ModelConfig,
    cache_cfg: CacheConfig,
    optimizer_cfg: OptimizerConfig,
) -> KVCacheBudget:
    """Compute full/quant/scale/scratch/metadata split from total KV budget.
    Raises:
        BudgetConfigError: if budget cannot fit minimum full pool or scratch.
    """


def quantize_from_full(
    storage_id: int,
    reason: str,
    step: int,
    allow_release_full: bool,
) -> QuantizeCommitResult:
    """Two-phase FULL->QUANT transition.
    Prepare: allocate quant slot and write int8/scales.
    Commit: atomically update metadata and visible tables.
    Release FULL only if commit succeeds and allow_release_full is True.
    Raises:
        QuantPoolExhaustedError: no quant slot available.
        QuantizationKernelError: quant kernel/reference failed.
        MetadataCommitError: atomic metadata swap failed.
    """


def dequantize_to_scratch(
    quant_block_ids: list[int],
    layer_id: int,
    scratch: torch.Tensor,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Reference dequantization path into scratch buffer.
    Raises:
        ScratchOverflowError: scratch is insufficient.
        QuantCacheError: quant block id is invalid.
    """


def rollback_quantize_prepare(
    transaction_id: str,
) -> None:
    """Rollback failed quantize prepare or commit.
    Raises:
        RollbackError: transaction cannot be safely rolled back.
    """
```

### Key Implementation Steps

1. Implement budget split with explicit logging.
2. Allocate full pool and quant pool from the same `total_kv_budget_bytes`.
3. Add quant cache tensors and scale tensors.
4. Implement torch reference Q8 quant/dequant.
5. Implement two-phase commit transaction object.
6. Add rollback on allocation, kernel, metadata, visible table update failure.
7. In P3, keep `allow_release_full=False` for serving unless P4a path is enabled.

### Feature Flags

```text
enable_kv_q8_shadow=False
enable_kv_q8_runtime=False
enable_triton_q8_kv=False
```

### Validation Commands

```bash
python -m pytest tests/kernels/test_q8_kv.py -v
python -m pytest tests/engine/test_quant_commit.py -v
python benchmarks/microbench_q8_kv.py --dtype fp16 --head-dim 128 --block-size <calibrated_block_size>
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 8 --output-json results/p3_q8_shadow.json --enable-arkv-metadata --enable-kv-q8-shadow
```

### Definition of Done

- `total_kv_budget_bytes` 公式在 config/init 中实现并被日志输出。
- P3 uses P-1 calibrated block size; microbench commands do not assume unsupported 16-token blocks.
- quant pool 从 total budget 中切分，不是额外追加。
- `quantize_from_full` 两阶段提交通过 failure injection tests。
- 中途失败时 FULL 保留且不进入 free pool。
- P3 shadow mode 可报告 potential reclaimed full-equivalent blocks。

### Failure Modes and Rollback

| Failure mode | Rollback operation | Impact on later phases |
|---|---|---|
| quant kernel 数值错误 | Disable `enable_triton_q8_kv`; use torch reference | P4a can proceed with reference only |
| quant pool 预算导致 full pool 太小 | Adjust budget ratios within total budget; do not add extra pool | Bench numbers delayed |
| commit 后 visible table 不一致 | Rollback transaction; keep FULL | P4a blocked |
| FULL 提前释放导致 stale read | Add hard gate: release only when mixed-KV read enabled and tests pass | P3 merge blocked |
| metadata commit partially succeeds | Transaction rollback or fail-stop; no silent recovery | P4a blocked |

### Estimated Days

```text
6-10 days
```

### Codex Implementation Prompt

```text
Implement the physical infrastructure for FULL->QUANT transitions. Add total KV
budget splitting into full_pool_bytes, quant_pool_bytes, scale_bytes,
scratch_budget, and metadata_budget. Add quant cache and scale storage. Implement
torch reference Q8 quant/dequant and quantize_from_full with rollback-safe
two-phase commit: allocate quant slot, write int8/scales, atomically update
PhysicalBlockMeta, SequenceKVRefTable, and VisibleBlockTable, then release FULL
only after successful commit. Do not release FULL into serving reuse until P4a
mixed-KV read path is enabled.
```

---

## P4a：decode-only mixed-KV fallback，形成 B2/B3 闭环

### Objective

实现 decode-only mixed-KV fallback attention，让 decode 阶段真实读取 FULL/QUANT visible entries。P4a 必须以 P-1 的 tensor-level materialization reference 为起点；在 graph safety 未验证前，`enable_mixed_kv_fallback=True` 默认强制 eager execution。P4a 通过即 MVP 关键闭环成立：B2/B3 可跑，FULL→QUANT 后释放的 full blocks 能被 scheduler/admission 使用。

### Dependencies / Parallelism

```text
Dependencies:
  P-1 decode materialization reference and eager/graph decision
  P3 quantize_from_full
  P2 VisibleBlockTable
  P1 admission for B3

Can run in parallel with:
  P6a reference microbench scaffold

Must finish before:
  P4b
  P5
  B2/B3 headline
```

### Files to Add / Modify

```text
Modify:
  nanovllm/layers/attention.py
  nanovllm/engine/model_runner.py
  nanovllm/engine/llm_engine.py
  nanovllm/engine/arkv_kv_manager.py
  nanovllm/engine/visible_tables.py

Add:
  nanovllm/layers/mixed_kv_fallback.py
  tests/integration/test_decode_mixed_kv_fallback.py
  tests/integration/test_full_reuse_after_quant.py
  tests/integration/test_workspace_planning.py
```

### Public Interfaces / Function Signatures

```python
def plan_mixed_kv_workspace(
    batch_plan: BatchPlan,
    visible_tables: VisibleBlockTable,
    cache_cfg: CacheConfig,
) -> MixedKVWorkspacePlan:
    """Estimate scratch needed for mixed-KV fallback read.
    Raises:
        WorkspacePlanningError: invalid visible table or unsupported shape.
    """


def run_decode_mixed_kv_fallback(
    q: torch.Tensor,
    visible_entries: list[list[VisibleBlockEntry]],
    full_k_cache: torch.Tensor,
    full_v_cache: torch.Tensor,
    quant_cache: QuantKVCache,
    workspace: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    """Decode-only fallback attention for FULL/QUANT visible entries.
    EVICT entries are skipped if present.
    Raises:
        MixedKVReadError: missing full/quant source or invalid entry ordering.
        ScratchOverflowError: workspace too small.
    """


def materialize_visible_kv_for_decode(
    visible_entries: list[VisibleBlockEntry],
    full_cache: FullKVCache,
    quant_cache: QuantKVCache,
    workspace: torch.Tensor,
) -> MaterializedKV:
    """Build contiguous K/V view for one decode sequence.
    Raises:
        MixedKVReadError: invalid state transition or missing block.
    """


def enable_full_reuse_after_quant(
    storage_id: int,
    commit_result: QuantizeCommitResult,
) -> None:
    """Release old FULL block after P4a read path is available.
    Raises:
        FullReuseSafetyError: mixed-KV read path is disabled or commit invalid.
    """
```

### Key Implementation Steps

1. Decode path reads only `VisibleBlockTable`.
2. FULL entries direct-read full cache.
3. QUANT entries dequantize to scratch.
4. EVICT entries are skipped if present, but P4a policy cannot generate them.
5. Add workspace planning before runner execution.
6. Force eager execution for mixed-KV fallback unless P-1/P6b explicitly proves graph safety.
7. After successful P4a validation, allow committed FULL blocks to return to free pool.
8. Run B2a/B2b/B3.

### Feature Flags

```text
enable_mixed_kv_fallback=False
enable_kv_q8_runtime=False
enable_memory_aware_scheduler=False
enable_admission_controller=False
```

Until P6b graph safety is validated:

```text
enable_mixed_kv_fallback=True implies enforce_eager=True
```

B2a/B2b/B3 commands explicitly enable required flags.

### Validation Commands

```bash
python -m pytest tests/integration/test_decode_mixed_kv_fallback.py -v
python -m pytest tests/integration/test_full_reuse_after_quant.py -v
python -m pytest tests/integration/test_workspace_planning.py -v

python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b2a_naive_q8.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy naive_age_q8

python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b2b_arkv_q8.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8

python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 16 --output-json results/b3_scheduler_arkv_q8.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8
```

### Definition of Done

- Decode mixed-KV fallback matches full-only reference within defined numerical tolerance for controlled tests。
- FULL→QUANT commit can release old full block and later reuse it without stale read。
- B2a/B2b/B3 all run end-to-end。
- P4a does not generate EVICT entries。
- Mixed-KV fallback runs in eager mode unless graph safety has explicit evidence。
- Closing `enable_mixed_kv_fallback` or `enable_kv_q8_runtime` returns to full-only path。

### Failure Modes and Rollback

| Failure mode | Rollback operation | Impact on later phases |
|---|---|---|
| decode output diverges beyond tolerance | Disable `enable_kv_q8_runtime`; keep shadow mode | MVP blocked |
| scratch OOM | Scheduler shrinks batch/chunk; otherwise fallback to full-only before commit | B2/B3 delayed |
| FULL reuse causes stale read | Disable full reuse after quant; keep QUANT shadow only | P4a merge blocked |
| VisibleBlockTable ordering bug | Rebuild table from logical refs each step and validate monotonic spans | P4b blocked |
| CUDA graph capture/replay corrupts mixed visible tables | Force `enforce_eager=True` when mixed fallback is enabled | B2/B3 can proceed, P6b owns graph recovery |
| EVICT appears before P5 | Hard assert policy output contains no EVICT | P4a merge blocked |

### Estimated Days

```text
7-12 days
```

### Codex Implementation Prompt

```text
Implement a real decode-only mixed-KV fallback attention path. Start from the P-1
tensor-level materialization reference, run mixed fallback in eager mode unless
graph safety is proven, read VisibleBlockTable only, gather FULL blocks directly,
dequantize QUANT blocks into bounded scratch, tolerate EVICT entries by skipping
them, and never generate EVICT policy entries in P4a. After correctness tests pass,
allow FULL blocks from successful quantize_from_full commits to be released and
reused. Run B2a, B2b, and B3 end-to-end. Preserve full-only fallback behind
default-off flags.
```

---

## P4b：prefill-prefix / unfinished-prefill mixed read

### Objective

在 P4a 已形成 B2/B3 闭环后，扩展 prefill-prefix 与 unfinished-prefill 对 QUANT blocks 的读取能力。P4b 失败不阻断 P4a headline。

### Dependencies / Parallelism

```text
Dependencies:
  P4a decode mixed-KV fallback

Can run in parallel with:
  P6a gather/dequant kernel implementation after P4a fallback reference stabilizes

Must finish before:
  P5 full EVICT quality gate for prefill-related workloads
```

### Files to Add / Modify

```text
Modify:
  nanovllm/layers/attention.py
  nanovllm/layers/mixed_kv_fallback.py
  nanovllm/engine/model_runner.py
  nanovllm/engine/visible_tables.py
  nanovllm/engine/scheduler.py

Add:
  tests/integration/test_prefill_prefix_mixed_kv.py
  tests/integration/test_unfinished_prefill_mixed_kv.py
  tests/integration/test_prefill_chunk_split_workspace.py
```

### Public Interfaces / Function Signatures

```python
def run_prefill_mixed_kv_fallback(
    q: torch.Tensor,
    visible_entries: list[list[VisibleBlockEntry]],
    slot_mapping: SlotMapping,
    full_k_cache: torch.Tensor,
    full_v_cache: torch.Tensor,
    quant_cache: QuantKVCache,
    workspace: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> torch.Tensor:
    """Prefill fallback attention with QUANT prefix support.
    Writes still target FULL slots through slot_mapping only.
    Raises:
        MixedKVReadError: invalid visible entries.
        InvalidWriteTargetError: slot_mapping points to non-FULL block.
        ScratchOverflowError: workspace budget exceeded.
    """


def split_prefill_for_workspace(
    prefill_task: PrefillTask,
    workspace_plan: MixedKVWorkspacePlan,
    cfg: SchedulerConfig,
) -> list[PrefillTask]:
    """Split prefill task if mixed-KV materialization exceeds scratch budget.
    Raises:
        WorkspacePlanningError: cannot split below minimum chunk.
    """


def validate_unfinished_prefill_policy(
    seq_state: SequenceKVState,
    reclaim_plan: ReclaimPlan,
) -> None:
    """Ensure unfinished-prefill sequence is not subject to EVICT.
    Raises:
        PolicyInvariantError: EVICT candidate found for unfinished prefill.
    """
```

### Key Implementation Steps

1. Extend fallback materialization to prefill prefix.
2. Preserve invariant: `slot_mapping` only writes FULL.
3. Allow unfinished-prefill to read QUANT.
4. Forbid EVICT on unfinished-prefill sequences.
5. Add chunk split if workspace budget exceeded.
6. Add shared-prefix + QUANT prefill tests.

### Feature Flags

```text
enable_prefill_mixed_kv_fallback=False
enable_mixed_kv_fallback=False
enable_kv_q8_runtime=False
```

### Validation Commands

```bash
python -m pytest tests/integration/test_prefill_prefix_mixed_kv.py -v
python -m pytest tests/integration/test_unfinished_prefill_mixed_kv.py -v
python -m pytest tests/integration/test_prefill_chunk_split_workspace.py -v

python benchmarks/benchmark_serving.py --workload shared_prefix --concurrency 16 --output-json results/p4b_shared_prefix_mixed_prefill.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-prefill-mixed-kv-fallback --reclaim-policy arkv_q8
```

### Definition of Done

- Prefill-prefix can read QUANT blocks via fallback。
- Unfinished-prefill can read QUANT but cannot be EVICTed。
- `slot_mapping` never points to QUANT/EVICT。
- Workspace overflow causes chunk split, not runtime OOM。
- P4b can be disabled without disabling P4a decode-only headline。

### Failure Modes and Rollback

| Failure mode | Rollback operation | Impact on later phases |
|---|---|---|
| Prefill mixed read incorrect | Disable `enable_prefill_mixed_kv_fallback` | P4a/B3 headline unaffected |
| Chunk split causes starvation | Tune starvation guard; cap split count | P5 prefill quality workloads delayed |
| unfinished-prefill EVICT appears | Hard fail policy validation; disable EVICT for all unfinished prefill | Blocks P5 |
| shared-prefix QUANT update inconsistent | Default protect shared-prefix from QUANT until fixed | Some memory savings reduced, correctness preserved |

### Estimated Days

```text
5-9 days
```

### Codex Implementation Prompt

```text
Extend mixed-KV fallback from decode-only to prefill-prefix and unfinished-prefill
reads. Preserve FULL-only slot_mapping for writes. Allow QUANT reads for prefix
and unfinished-prefill sequences, but forbid EVICT for unfinished-prefill. Add
workspace planning and chunk splitting so scratch overflow is handled before
execution. Keep P4b behind its own flag so P4a/B2/B3 headline remains valid if
P4b is disabled.
```

---

## P5：EVICT policy quality gate

### Objective

只有在 P5 才允许 policy 生成 EVICT entry。EVICT 必须通过 quality gate 才能进入 optional B2c headline；否则只保留为 disabled / benchmark-only path。B4/B5 默认基于 B3 QUANT-only，不默认依赖 EVICT。

### Dependencies / Parallelism

```text
Dependencies:
  P4a backend can skip EVICT
  P4b prefill safeguards if prefill workloads are included
  P0 quality benchmark harness

Can run in parallel with:
  P6a kernel acceleration, but EVICT results must not contaminate B4/B5

Must finish before:
  optional B2c headline
  any EVICT-enabled release note
```

### Files to Add / Modify

```text
Modify:
  nanovllm/engine/kv_policy.py
  nanovllm/engine/arkv_kv_manager.py
  nanovllm/engine/scheduler.py
  nanovllm/engine/visible_tables.py
  benchmarks/workloads/quality_passkey.py
  benchmarks/report.py

Add:
  tests/quality/test_evict_quality_gate.py
  tests/engine/test_evict_policy_guards.py
  tests/integration/test_evict_visible_context.py
```

### Public Interfaces / Function Signatures

```python
@dataclass
class QualityGateResult:
    passed: bool
    passkey_drop_abs: float
    retrieval_drop_abs: float
    greedy_token_agreement: float
    slo_goodput_delta: float
    reason: str


def select_blocks_to_evict(
    candidates: list[ReclaimCandidate],
    required_full_equiv: int,
    seq_states: dict[int, SequenceKVState],
    cfg: PolicyConfig,
) -> tuple[list[ReclaimCandidate], int]:
    """Select EVICT candidates, preferring already-QUANT low-score blocks.
    Raises:
        PolicyInvariantError: direct FULL->EVICT selected while disabled.
    """


def apply_evict_transition(
    storage_id: int,
    step: int,
    reason: str,
) -> EvictCommitResult:
    """Transition QUANT or explicitly allowed FULL block to EVICT.
    Updates PhysicalBlockMeta and VisibleBlockTable.
    Raises:
        EvictNotAllowedError: P5 gate disabled or protected block.
        MetadataCommitError: visible/logical update failed.
    """


def run_quality_gate(
    benchmark_results: list[dict],
    thresholds: QualityGateThresholds,
) -> QualityGateResult:
    """Evaluate passkey/retrieval/token-agreement quality gate.
    Raises:
        QualityGateConfigError: missing required metrics.
    """
```

### Key Implementation Steps

1. Implement `select_blocks_to_evict`.
2. Enforce guards: shared-prefix、sink、recent、inflight-write、unfinished-prefill.
3. Prefer QUANT→EVICT; direct FULL→EVICT disabled by default.
4. Update VisibleBlockTable: logical span retained, visible span skipped.
5. Implement quality gate benchmark and fail-close behavior.
6. Add optional B2c run; do not mix into B3/B4/B5 default.

### Feature Flags

```text
enable_kv_evict=False
enable_direct_full_evict=False
enable_quality_gate=False
```

### Validation Commands

```bash
python -m pytest tests/engine/test_evict_policy_guards.py -v
python -m pytest tests/integration/test_evict_visible_context.py -v
python -m pytest tests/quality/test_evict_quality_gate.py -v

python benchmarks/benchmark_serving.py --workload quality_passkey --concurrency 8 --output-json results/p5_quality_gate.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-kv-evict --enable-quality-gate --reclaim-policy arkv_q8_evict

python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b2c_optional_evict.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-kv-evict --enable-quality-gate --reclaim-policy arkv_q8_evict
```

### Definition of Done

- P5 前所有 EVICT generation tests fail closed。
- P5 后 EVICT only when `enable_kv_evict=True` and quality gate enabled/passed。
- EVICT prefers QUANT low-score blocks。
- `allow_direct_full_evict=False` default enforced。
- B2c marked optional and only enters headline if quality gate passes。

### Failure Modes and Rollback

| Failure mode | Rollback operation | Impact on later phases |
|---|---|---|
| Quality drop exceeds threshold | Disable `enable_kv_evict`; keep B2c optional/non-headline | B3/B4/B5 unaffected |
| EVICT selects shared-prefix | Hard fail and fix guard | P5 blocked |
| direct FULL->EVICT occurs by default | Reset flag false and add regression | P5 blocked |
| visible_context_len corrupts logical timeline | Rollback EVICT transition; rebuild visible from logical table | P5 blocked |
| EVICT improves memory but hurts SLO-goodput | Do not include B2c in headline | Kernel phases proceed on QUANT-only |

### Estimated Days

```text
5-8 days
```

### Codex Implementation Prompt

```text
Add EVICT as a quality-gated feature only in P5. The policy must prefer already
QUANT low-importance blocks and keep allow_direct_full_evict=False by default.
Enforce hard guards for shared-prefix, sink, recent, inflight-write, and
unfinished-prefill blocks. EVICT updates VisibleBlockTable by removing attention
visibility but preserving logical_context_len. Wire passkey/retrieval/token
agreement quality gates so EVICT fails closed and B2c remains optional unless
all gates pass.
```

---

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

## P7：benchmark / ablation / release hardening / README

### Objective

完成 B0-B5 全 8 组 ablation、metrics report、README、limitations、rollback docs 与面试回答模板。P7 是 release hardening，不允许新增未门控的 optimizer 行为。

### Dependencies / Parallelism

```text
Dependencies:
  P-1-P6b

Can run partially in parallel:
  report script after P0
  README skeleton after P2
  interview narrative after P4a

Must finish last:
  final release
```

### Files to Add / Modify

```text
Modify:
  README.md
  benchmarks/report.py
  nanovllm/config.py

Add:
  docs/memory_aware_optimizer.md
  docs/benchmark_ablation.md
  docs/risk_gates.md
  docs/interview_narrative.md
  tests/integration/test_all_flags_off_baseline.py
  tests/integration/test_fallback_paths.py
```

### Public Interfaces / Function Signatures

```python
def run_ablation_suite(
    model: str,
    workloads: list[str],
    output_dir: str,
    concurrency_sweep: list[int],
    include_optional_evict: bool = False,
) -> AblationSuiteResult:
    """Run B0-B5 ablation suite.
    Raises:
        AblationConfigError: invalid flags or missing required workload.
        AblationGateError: required risk gate failed.
    """


def generate_ablation_report(
    results_dir: str,
    output_markdown: str,
    output_csv: str,
) -> None:
    """Generate markdown and CSV ablation report.
    Raises:
        ReportSchemaError: missing metrics or incompatible schema.
    """


def validate_release_gates(
    results: AblationSuiteResult,
    gates: list[RiskGate],
) -> None:
    """Validate all merge gates before release.
    Raises:
        MergeGateError: any hard risk gate fails.
    """
```

### Key Implementation Steps

1. Add all-flags-off regression.
2. Run B0/B1/B2a/B2b/B2c/B3/B4/B5.
3. Keep B2c optional and gate-controlled.
4. Ensure B4/B5 default based on B3 QUANT-only.
5. Generate report with metrics and failure interpretation.
6. Write README with flags、commands、fallback、limitations。
7. Add interview narrative.

### Feature Flags

No new flags. P7 verifies defaults:

```text
all optimizer feature flags default False
full-only fallback always available
```

### Validation Commands

```bash
python -m pytest tests/integration/test_all_flags_off_baseline.py -v
python -m pytest tests/integration/test_fallback_paths.py -v
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 16 --output-json results/final_b0.json
python benchmarks/run_ablation_suite.py --model <model> --output-dir results/ablation --concurrency-sweep 1,2,4,8,16,32
python benchmarks/report.py --results-dir results/ablation --output-markdown docs/benchmark_ablation.md --output-csv results/ablation_summary.csv
```

### Definition of Done

- B0-B5 全 8 组 ablation schema 完整。
- B2c clearly optional，且未过 quality gate 时不进 headline。
- 7 条 Risk Gates 全部作为 merge gate 通过。
- README 包含架构图、feature flags、运行命令、rollback、limitations。
- 面试回答模板完整可用。

### Failure Modes and Rollback

| Failure mode | Rollback operation | Impact on later phases |
|---|---|---|
| Ablation missing required metric | Regenerate benchmark with fixed schema | Release blocked |
| B2c quality gate fails | Mark optional/non-headline | Release can proceed without EVICT headline |
| B4/B5 accidentally include EVICT | Re-run QUANT-only flags | Release blocked until corrected |
| all-flags-off differs from baseline | Fix defaults/fallback | Release blocked |
| README command stale | Add CI smoke for docs commands | Release blocked |

### Estimated Days

```text
4-7 days
```

### Codex Implementation Prompt

```text
Finalize the project with reproducible B0-B5 ablations, benchmark scripts,
report generation, README, limitations, rollback docs, risk gate validation, and
an interview-ready narrative. Ensure all optimizer feature flags default off and
full-only fallback is always available. B2c must be optional and quality-gated;
B4/B5 must default to B3 QUANT-only. Do not add new un-gated optimizer behavior.
```

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
