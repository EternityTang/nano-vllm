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
