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
