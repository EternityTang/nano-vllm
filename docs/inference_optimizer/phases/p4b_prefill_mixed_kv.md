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
