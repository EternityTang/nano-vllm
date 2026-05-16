# Current Task Context

Last updated: 2026-05-15

## Current Phase

```text
P2 metadata tables, PhysicalBlockMeta + SequenceKVRef, three-table separation
```

P1 scheduler/admission validation is complete. Do not start P3 quantization or P4 visible read-path work until P2 metadata dry-run is complete.

## Read These Files First

```text
inference_systems_code_plan.md
docs/inference_optimizer/overview.md
docs/inference_optimizer/phases/p2_metadata_tables.md
omx_wiki/memory-aware-inference-implementation-log.md
```

Only read the full archive if these files do not answer a required question:

```text
docs/inference_optimizer/archive/full_code_plan_2026-05-15.md
```

## Target Result

Produce metadata truth tables and dry-run policy that record:

- PhysicalBlockMeta + SequenceKVRef split
- logical / physical / visible / write view separation
- shared-prefix owner refs and ref counts
- visible table invariant validation
- deterministic reclaim policy dry-run metrics

## Expected Files

```text
nanovllm/engine/kv_meta.py
nanovllm/engine/kv_policy.py
nanovllm/engine/visible_tables.py
tests/engine/test_kv_meta.py
tests/engine/test_visible_tables.py
tests/engine/test_kv_policy_dry_run.py
```

Optional only if needed:

```text
nanovllm/config.py
nanovllm/engine/block_manager.py
nanovllm/engine/sequence.py
```

All P2 behavior must remain disabled by default and must not change runtime attention semantics.

## Validation Commands

```bash
python -m pytest tests/engine/test_kv_meta.py -v
python -m pytest tests/engine/test_visible_tables.py -v
python -m pytest tests/engine/test_kv_policy_dry_run.py -v
python benchmarks/benchmark_serving.py --workload shared_prefix --concurrency 16 --output-json results/p2_metadata_dryrun.json --enable-arkv-metadata --enable-arkv-policy-dry-run
```

## Acceptance Criteria

- PhysicalBlockMeta + SequenceKVRef split exists and passes shared-prefix tests.
- Logical / physical / visible table separation exists and has invariant tests.
- `slot_mapping` and `VisibleBlockTable` are not mixed.
- Dry-run reclaim plan is deterministic and non-mutating.
- Runtime output remains full-only baseline when P2 flags are closed.

## Relevant Existing Code

Use these files for P2 metadata dry-run checks:

```text
nanovllm/config.py
nanovllm/engine/block_manager.py
nanovllm/engine/sequence.py
nanovllm/engine/kv_meta.py
nanovllm/engine/kv_policy.py
nanovllm/engine/visible_tables.py
benchmarks/benchmark_serving.py
benchmarks/workloads/
```

## Stop Condition

Stop P2 only when metadata/visible-table/policy tests pass, dry-run report exists, default-off fallback is tested, and any metadata blocker is documented here.

## P-1 Status Update - 2026-05-16

Generated P-1 dry-run artifacts:

```text
benchmarks/capability_probe.py
tests/bench/test_capability_smoke.py
results/p_minus_1_capability.json
```

Dry-run calibration recorded:

- formal benchmark model: `/home/tang/nano-vllm/weight/Qwen3-1.7B`
- correctness/smoke model: `/home/tang/nano-vllm/weight/Qwen3-0.6B`
- formal KV block size: `256`
- smaller block sizes: unsupported by current `Config.__post_init__`
- mixed-KV fallback reference: CPU tensor-level materialization/decode parity passed
- CUDA graph policy: mixed-KV fallback requires eager execution until P6b proves graph safety
- optimizer behavior enabled: no

Retest result on 2026-05-16:

- `python` is now `3.12.13`, satisfying project `>=3.10,<3.13`
- `flash-attn`, `pytest`, `torch`, `triton`, and `transformers` are installed in the active interpreter
- P-1 import smoke, dry-run probe, and pytest smoke tests pass
- non-sandbox GPU probe sees `NVIDIA GeForce RTX 4070 Ti` with `12282 MiB` total memory
- `results/p_minus_1_capability.json` currently has no probe blockers

P-1 dry-run calibration is complete. Real B0/model-load benchmarking remains a P0 validation step, not a P-1 blocker.

## P0 Status Update - 2026-05-16

Generated P0 dry-run artifacts:

```text
benchmarks/benchmark_serving.py
benchmarks/report.py
benchmarks/workloads/
tests/bench/test_metrics_smoke.py
results/b0_scheduler_stress.json
results/b0_scheduler_stress.csv
results/b0_long_context.json
results/b0_long_context.csv
```

Validation passed:

```bash
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 1 --dry-run --output-json /tmp/b0_dryrun.json
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 16 --max-requests 16 --dry-run --output-json results/b0_scheduler_stress.json
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 8 --max-requests 8 --dry-run --output-json results/b0_long_context.json
python -m pytest tests/bench/test_capability_smoke.py tests/bench/test_metrics_smoke.py -v
```

Real B0 result:

- corrected formal model is `/home/tang/nano-vllm/weight/Qwen3-1.7B`
- `scheduler_stress` real B0 passed and saved to `results/b0_scheduler_stress.json` / `.csv`
- `long_context_pressure` real B0 passed and saved to `results/b0_long_context.json` / `.csv`
- previous `/home/tang/models/Qwen3-4B` selection was a probe search-priority bug; repo-local `weight/` models now take precedence

## P1 Status Update - 2026-05-16

Generated P1 artifacts:

```text
nanovllm/engine/tasks.py
nanovllm/engine/admission.py
nanovllm/engine/scheduler_metrics.py
tests/engine/test_scheduler_tasks.py
tests/engine/test_admission.py
results/b1_scheduler_only.json
results/b1_scheduler_only.csv
results/b1_shared_prefix.json
results/b1_shared_prefix.csv
```

Validation passed:

```bash
python -m pytest tests/bench/test_capability_smoke.py tests/bench/test_metrics_smoke.py tests/engine/test_scheduler_tasks.py tests/engine/test_admission.py -v
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 16 --output-json results/b1_scheduler_only.json --enable-memory-aware-scheduler --enable-admission-controller
python benchmarks/benchmark_serving.py --workload shared_prefix --concurrency 16 --output-json results/b1_shared_prefix.json --enable-memory-aware-scheduler --enable-admission-controller
```

Real B1 result:

- `scheduler_stress`: passed, `throughput_tokens_per_s=222.81`, `raw_peak_vram_bytes=9383980544`
- `shared_prefix`: passed, `throughput_tokens_per_s=225.12`, `raw_peak_vram_bytes=9741792768`
- admission counts are present in both reports
- P1 scheduler remains homogeneous: decode-only or prefill-only; true mixed execution is still not required
