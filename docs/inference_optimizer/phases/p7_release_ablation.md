## P7：benchmark / ablation / release hardening / README

### Objective

完成 B0-B5 全 8 组 ablation、metrics report、README、limitations、rollback docs 与面试回答模板。P7 是 release hardening，不允许新增未门控的 optimizer 行为。

### Dependencies / Parallelism

```text
Dependencies:
  P-1-P6c

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
5. Include P6c profile artifacts as the B3/B4/B5 performance explanation appendix.
6. Generate report with metrics and failure interpretation.
7. Write README with flags、commands、fallback、limitations。
8. Add interview narrative.

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

### P7 Status Update - 2026-05-17

Generated P7 artifacts:

```text
benchmarks/run_ablation_suite.py
docs/memory_aware_optimizer.md
docs/benchmark_ablation.md
docs/risk_gates.md
docs/interview_narrative.md
results/ablation_summary.csv
tests/integration/test_all_flags_off_baseline.py
tests/integration/test_fallback_paths.py
```

Implemented:

- `--disable-all-optimizer-flags` now actively clears optimizer flags in `benchmarks/benchmark_serving.py`.
- `benchmarks/report.py` can generate markdown/csv ablation summaries from benchmark JSON artifacts.
- `benchmarks/run_ablation_suite.py` emits reproducible B0/B1/B2a/B2b/B3/B4/B5 and optional B2c commands.
- README links release docs and includes B5/profile report commands.
- Release docs cover architecture, flags, rollback, limitations, risk gates, and interview narrative.

Validation passed:

```bash
python -m pytest tests/integration/test_all_flags_off_baseline.py tests/integration/test_fallback_paths.py -v
python benchmarks/report.py --results-dir results --output-markdown docs/benchmark_ablation.md --output-csv results/ablation_summary.csv
python benchmarks/run_ablation_suite.py --output-dir results/ablation --concurrency-sweep 16 --include-optional-evict --plan-only
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 1 --max-requests 1 --dry-run --output-json /tmp/p7_all_off_gate.json --disable-all-optimizer-flags --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-kv-evict --enable-quality-gate
python -m pytest tests -q
```

Real P7 result:

- P7 targeted release gate tests: `5 passed`.
- Full suite: `83 passed`.
- Ablation report generation status: `ok`.
- Suite plan-only includes B4/B5 QUANT-only commands with no EVICT flags.
- Dry-run all-off override status: `ok`, `active_quant_blocks=0`, `evicted_blocks=0`, `fused_kernel_calls=0`.

Final ablation smoke sweep on concurrency 16:

| Group | Workload | SLO-goodput tok/s | Active QUANT | EVICT | Quality gate | Notes |
|---|---|---:|---:|---:|---|---|
| B0 | `scheduler_stress` | `243.95` | `0` | `0` | `false` | all optimizer flags disabled |
| B1 | `scheduler_stress` | `241.47` | `0` | `0` | `false` | scheduler/admission only |
| B2a | `long_context_pressure` | `27.68` | `16` | `0` | `false` | naive Q8 |
| B2b | `long_context_pressure` | `28.14` | `16` | `0` | `false` | ARKV Q8 |
| B2c | `quality_passkey` | `67.13` | `0` | `31` | `true` | optional quality-gated EVICT |
| B3 | `long_context_pressure` | `24.41` | `16` | `0` | `false` | scheduler + ARKV Q8 |
| B4 | `long_context_pressure` | `24.22` | `16` | `0` | `false` | QUANT-only Triton gather/dequant |
| B5 | `long_context_pressure` | `40.44` | `16` | `0` | `false` | QUANT-only fused decode, `fused_kernel_calls=15512`, `fused_kernel_fallbacks=0` |

Final report regeneration:

```bash
python benchmarks/report.py --results-dir results --output-markdown docs/benchmark_ablation.md --output-csv results/ablation_summary.csv
```

Post-sweep verification:

- `results/ablation_summary.csv` has 8 rows.
- B2c is optional and quality-gated: `quality_gate_passed=true`.
- B4/B5 remain QUANT-only: `evicted_blocks=0`.
- Required summary metrics exist for all B0/B1/B2a/B2b/B2c/B3/B4/B5 artifacts.
- `docs/benchmark_ablation.md` represents all 7 risk gates and states current limitations.

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
