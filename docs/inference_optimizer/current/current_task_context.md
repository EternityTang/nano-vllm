# Current Task Context

Last updated: 2026-05-16

## Current Phase

```text
P3 QUANT path, two-phase commit, unified KV budget split
```

P2 metadata dry-run validation is complete. Do not start P4 visible read-path work until P3 quant shadow/two-phase commit is complete.

## Read These Files First

```text
inference_systems_code_plan.md
docs/inference_optimizer/overview.md
docs/inference_optimizer/phases/p3_quant_path.md
omx_wiki/memory-aware-inference-implementation-log.md
```

Only read the full archive if these files do not answer a required question:

```text
docs/inference_optimizer/archive/full_code_plan_2026-05-15.md
```

## Target Result

Produce a rollback-safe FULL->QUANT physical path that records:

- unified `total_kv_budget_bytes` split
- full/quant/scale/scratch/metadata budget accounting
- torch reference Q8 quant/dequant path
- `quantize_from_full` two-phase prepare/commit/rollback
- shadow-mode reclaimed full-equivalent reporting

## Expected Files

```text
nanovllm/engine/arkv_kv_manager.py
nanovllm/engine/quant_cache.py
nanovllm/kernels/q8_kv.py
tests/engine/test_quant_commit.py
tests/kernels/test_q8_kv.py
```

Optional only if needed:

```text
nanovllm/config.py
nanovllm/engine/block_manager.py
nanovllm/engine/model_runner.py
benchmarks/microbench_q8_kv.py
```

P3 may implement shadow / controlled runtime. Do not release FULL blocks into serving reuse until P4a mixed-KV read path is enabled and tested.

## Validation Commands

```bash
python -m pytest tests/kernels/test_q8_kv.py -v
python -m pytest tests/engine/test_quant_commit.py -v
python benchmarks/microbench_q8_kv.py --dtype fp16 --head-dim 128 --block-size 256
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 8 --output-json results/p3_q8_shadow.json --enable-arkv-metadata --enable-kv-q8-shadow
```

## Acceptance Criteria

- `total_kv_budget_bytes` is split into full/quant/scale/scratch/metadata budgets.
- Quant pool is carved from the total KV budget, not added beside it.
- Q8 reference quant/dequant tests pass for calibrated block size `256`.
- `quantize_from_full` is rollback-safe under failure injection.
- FULL blocks are retained unless mixed-KV read path is explicitly available.

## Relevant Existing Code

Use these files for P3 quant shadow checks:

```text
nanovllm/config.py
nanovllm/engine/block_manager.py
nanovllm/engine/kv_meta.py
nanovllm/engine/visible_tables.py
nanovllm/engine/arkv_kv_manager.py
nanovllm/engine/quant_cache.py
nanovllm/kernels/q8_kv.py
benchmarks/benchmark_serving.py
```

## Stop Condition

Stop P3 only when Q8 reference tests pass, quant commit rollback tests pass, shadow report exists, default-off fallback is tested, and any quant blocker is documented here.

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

## P2 Status Update - 2026-05-16

Generated P2 artifacts:

```text
nanovllm/engine/kv_meta.py
nanovllm/engine/kv_policy.py
nanovllm/engine/visible_tables.py
tests/engine/test_kv_meta.py
tests/engine/test_visible_tables.py
tests/engine/test_kv_policy_dry_run.py
results/p2_metadata_dryrun.json
results/p2_metadata_dryrun.csv
```

Validation passed:

```bash
python -m pytest tests/bench/test_capability_smoke.py tests/bench/test_metrics_smoke.py tests/engine/test_scheduler_tasks.py tests/engine/test_admission.py tests/engine/test_kv_meta.py tests/engine/test_visible_tables.py tests/engine/test_kv_policy_dry_run.py -v
python benchmarks/benchmark_serving.py --workload shared_prefix --concurrency 16 --output-json results/p2_metadata_dryrun.json --enable-arkv-metadata --enable-arkv-policy-dry-run
```

Real P2 result:

- P-1/P0/P1/P2 regression suite passed: `30 passed`.
- `shared_prefix` real P2 metadata dry-run passed, `throughput_tokens_per_s=224.08`, `raw_peak_vram_bytes=9741792768`.
- `metadata_policy` summary is present: `candidate_count=168`, `conservative_reclaimable_blocks=1`, `protected_ratio_max=1.0`.
- P2 remains metadata/policy dry-run only; runtime attention semantics and FULL block reuse are unchanged.
- `enable_arkv_metadata` and `enable_arkv_policy_dry_run` remain default-off.
