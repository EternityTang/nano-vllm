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
