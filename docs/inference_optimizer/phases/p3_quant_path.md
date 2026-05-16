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
