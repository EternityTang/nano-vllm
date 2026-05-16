# Current Task Context

Last updated: 2026-05-16

## Current Phase

```text
P4a decode-only mixed-KV fallback
```

P3 quant shadow/two-phase commit validation is complete. Do not start P4b prefill mixed-KV work until P4a decode-only fallback and full-reuse safety are complete.

## Read These Files First

```text
inference_systems_code_plan.md
docs/inference_optimizer/overview.md
docs/inference_optimizer/phases/p4a_decode_mixed_kv_fallback.md
omx_wiki/memory-aware-inference-implementation-log.md
```

Only read the full archive if these files do not answer a required question:

```text
docs/inference_optimizer/archive/full_code_plan_2026-05-15.md
```

## Target Result

Produce a decode-only mixed-KV fallback path that:

- reads FULL/QUANT entries through `VisibleBlockTable`
- dequantizes QUANT blocks into bounded scratch
- matches full-only decode reference within tolerance in controlled tests
- permits FULL reuse only after mixed-KV read correctness is proven
- keeps all mixed-KV behavior default-off and eager-only until graph safety is proven

## Expected Files

```text
nanovllm/layers/mixed_kv_fallback.py
tests/integration/test_decode_mixed_kv_fallback.py
tests/integration/test_full_reuse_after_quant.py
tests/integration/test_workspace_planning.py
```

Likely modifications:

```text
nanovllm/layers/attention.py
nanovllm/engine/arkv_kv_manager.py
nanovllm/engine/llm_engine.py
nanovllm/engine/model_runner.py
nanovllm/engine/visible_tables.py
```

P4a may enable controlled FULL reuse only after decode mixed-KV correctness tests pass. P4a must not generate EVICT entries.

## Validation Commands

```bash
python -m pytest tests/integration/test_decode_mixed_kv_fallback.py -v
python -m pytest tests/integration/test_full_reuse_after_quant.py -v
python -m pytest tests/integration/test_workspace_planning.py -v
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b2a_naive_q8.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy naive_age_q8
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b2b_arkv_q8.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 16 --output-json results/b3_scheduler_arkv_q8.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8
```

## Acceptance Criteria

- Decode mixed-KV fallback matches full-only reference within defined tolerance.
- QUANT entries dequantize through bounded scratch; scratch overflow is rejected.
- FULL entries read directly from full cache.
- FULL blocks from successful quant commits can be released and reused without stale reads.
- Closing `enable_mixed_kv_fallback` or `enable_kv_q8_runtime` returns to full-only behavior.

## Relevant Existing Code

Use these files for P4a decode mixed-KV checks:

```text
nanovllm/config.py
nanovllm/layers/attention.py
nanovllm/layers/mixed_kv_fallback.py
nanovllm/engine/arkv_kv_manager.py
nanovllm/engine/visible_tables.py
nanovllm/engine/quant_cache.py
nanovllm/kernels/q8_kv.py
benchmarks/benchmark_serving.py
```

## Stop Condition

Stop P4a only when decode mixed-KV fallback tests pass, full reuse after quant is safe under tests, workspace planning bounds scratch use, B2a/B2b/B3 run end-to-end, and any mixed-KV blocker is documented here.

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

## P3 Status Update - 2026-05-16

Generated P3 artifacts:

```text
nanovllm/engine/arkv_kv_manager.py
nanovllm/engine/quant_cache.py
nanovllm/kernels/q8_kv.py
benchmarks/microbench_q8_kv.py
tests/engine/test_quant_commit.py
tests/kernels/test_q8_kv.py
results/p3_q8_shadow.json
results/p3_q8_shadow.csv
```

Implemented:

- unified `total_kv_budget_bytes` split into full/quant/scale/scratch/metadata budgets
- Q8 torch reference quant/dequant with per-head-vector scales
- quant cache allocation, scale storage, and scratch dequant path
- rollback-safe `ARKVKVManager.quantize_from_full` prepare/commit/rollback path
- hard release gate: FULL blocks are retained unless mixed-KV read availability is explicitly true
- q8 shadow benchmark report fields for potential reclaimed full-equivalent blocks

Validation passed:

```bash
python -m pytest tests/kernels/test_q8_kv.py -v
python -m pytest tests/engine/test_quant_commit.py -v
python benchmarks/microbench_q8_kv.py --dtype fp16 --head-dim 128 --block-size 256
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 8 --output-json results/p3_q8_shadow.json --enable-arkv-metadata --enable-kv-q8-shadow
python -m pytest tests/bench/test_capability_smoke.py tests/bench/test_metrics_smoke.py tests/engine/test_scheduler_tasks.py tests/engine/test_admission.py tests/engine/test_kv_meta.py tests/engine/test_visible_tables.py tests/engine/test_kv_policy_dry_run.py tests/kernels/test_q8_kv.py tests/engine/test_quant_commit.py -v
```

Real P3 shadow result:

- P-1/P0/P1/P2/P3 regression suite passed: `39 passed`.
- Q8 microbench passed on CPU fallback for calibrated block size `256`, `max_abs_error=0.015625`.
- `long_context_pressure` real P3 q8 shadow passed and saved to `results/p3_q8_shadow.json` / `.csv`.
- q8 shadow summary is present: `candidate_count=488`, `potential_reclaimed_full_equiv_blocks=416`, `quantized_shadow_blocks=416`, `full_blocks_retained=true`.
- P3 still does not release FULL blocks into serving reuse; P4a mixed-KV read path remains required before enabling release.

## P4a Status Update - 2026-05-16

Generated P4a artifacts:

```text
nanovllm/layers/mixed_kv_fallback.py
tests/integration/test_decode_mixed_kv_fallback.py
tests/integration/test_full_reuse_after_quant.py
tests/integration/test_p4a_serving_activation.py
tests/integration/test_workspace_planning.py
benchmarks/workloads/b3_reclaim_pressure.py
results/b2a_naive_q8.json
results/b2b_arkv_q8.json
results/b3_scheduler_arkv_q8.json
results/b1_reclaim_pressure.json
results/b3_reclaim_pressure_arkv_q8.json
```

Implemented:

- decode-only mixed-KV fallback that reads `VisibleBlockEntry` lists and materializes FULL/QUANT entries
- Q8 QUANT dequantization into bounded scratch through `QuantCache.dequantize_to_scratch`
- torch decode attention fallback with GQA head repeat support
- `VisibleBlockEntry.quant_block_id` so visible tables carry the QUANT read pointer
- eager-only model runner context wiring for mixed fallback when runtime flags and per-sequence visible entries are present
- controlled FULL reuse validation after successful quant commit when `mixed_kv_read_available=True`
- serving-loop activation for FULL allocation -> policy selection -> quant commit -> FULL release -> visible QUANT -> decode fallback QUANT read
- P4a runtime metrics: `quantized_block_ratio`, `reclaim_trigger_count`, `quant_commits_success`, `quant_commits_rollback`, `full_blocks_released_after_quant`, `mixed_kv_quant_reads`, `visible_quant_entries`
- B3 reclaim-pressure workload plus `--require-arkv-q8-reclaim` benchmark assertion for scheduler/admission + ARKV Q8 activation
- benchmark CLI support for P4a runtime flags and `--reclaim-policy`

Validation passed:

```bash
python -m pytest tests/integration/test_decode_mixed_kv_fallback.py -v
python -m pytest tests/integration/test_full_reuse_after_quant.py -v
python -m pytest tests/integration/test_workspace_planning.py -v
python -m pytest tests -v
python -c "import nanovllm; print(nanovllm.__all__)"
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b2a_naive_q8.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy naive_age_q8
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b2b_arkv_q8.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 16 --output-json results/b3_scheduler_arkv_q8.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8
python benchmarks/benchmark_serving.py --workload b3_reclaim_pressure --concurrency 16 --output-json results/b1_reclaim_pressure.json --enable-memory-aware-scheduler --enable-admission-controller
python benchmarks/benchmark_serving.py --workload b3_reclaim_pressure --concurrency 16 --output-json results/b3_reclaim_pressure_arkv_q8.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8 --require-arkv-q8-reclaim
```

Real P4a result:

- full test suite passed: `48 passed`
- B2a `long_context_pressure` status `ok`, `throughput_tokens_per_s=31.02`, `active_quant_blocks=16`, `quant_commits_success=16`, `full_blocks_released_after_quant=16`, `mixed_kv_quant_reads=68320`, `visible_quant_entries=16`, `evicted_blocks=0`, `raw_peak_vram_bytes=10043701760`
- B2b `long_context_pressure` status `ok`, `throughput_tokens_per_s=30.57`, `active_quant_blocks=16`, `quant_commits_success=16`, `full_blocks_released_after_quant=16`, `mixed_kv_quant_reads=68320`, `visible_quant_entries=16`, `evicted_blocks=0`, `raw_peak_vram_bytes=10043701760`
- B3 `scheduler_stress` status `ok`, `throughput_tokens_per_s=76.38`, admission `admitted=16`, `active_quant_blocks=0`, `quant_commits_success=0`, `evicted_blocks=0`, `raw_peak_vram_bytes=9328406016`
- B3 has no eligible old non-recent reclaim candidates under the current 256-token block size and P4a sink/recent/write protection rules, so quant activation is correctly absent there.
- B1 reclaim-pressure comparison status `ok`, `oom_requests=0`, `max_stable_concurrency=16`, `slo_goodput_tokens_per_s=279.78`, `active_quant_blocks=0`, `raw_peak_vram_bytes=10099382784`
- B3 reclaim-pressure ARKV Q8 status `ok`, `oom_requests=0`, `max_stable_concurrency=16`, `slo_goodput_tokens_per_s=45.22`, `active_quant_blocks=16`, `quant_commits_success=16`, `full_blocks_released_after_quant=16`, `mixed_kv_quant_reads=29232`, `visible_quant_entries=16`, `free_full_blocks_reclaim_delta=1`, `evicted_blocks=0`, `raw_peak_vram_bytes=10043808256`
- Under the measured reclaim-pressure workload, ARKV Q8 proves scheduler/admission reclaim activation and slightly lower peak VRAM; it does not improve OOM rate, measured max stable concurrency, or SLO-goodput yet because P4a uses the slow Python fallback.
- P4a does not generate EVICT entries.
- Mixed fallback remains default-off and eager-only; when no per-sequence visible entries are attached, runtime falls back to the existing full-only decode path.
