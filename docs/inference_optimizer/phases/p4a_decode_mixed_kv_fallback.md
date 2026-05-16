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
