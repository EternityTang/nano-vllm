# Current Task Context

Last updated: 2026-05-17

## Current Phase

```text
P7 release ablation/documentation hardening in progress
```

P6c end-to-end profiling and low-risk stabilization is complete. P7 release hardening is now preparing reproducible ablation docs, release risk gates, fallback/default-off tests, and interview/release narrative. B3/B4/B5 remain QUANT-only by default and do not enable EVICT.

## Read These Files First

```text
inference_systems_code_plan.md
docs/inference_optimizer/overview.md
docs/inference_optimizer/phases/p6b_mixed_kv_decode_kernel.md
omx_wiki/memory-aware-inference-implementation-log.md
```

Only read the full archive if these files do not answer a required question:

```text
docs/inference_optimizer/archive/full_code_plan_2026-05-15.md
```

## Target Result

P7 release hardening should finish without adding optimizer behavior. It must:

- keep EVICT optional-only; do not mix B2c into B4/B5 default results
- preserve B3/B4 QUANT-only behavior and metrics as the comparison base
- keep fallback/reference path available for correctness
- keep all optimizer/kernel flags default-off
- provide reproducible ablation/report commands and release docs

## Expected Files

```text
docs/memory_aware_optimizer.md
docs/benchmark_ablation.md
docs/risk_gates.md
docs/interview_narrative.md
tests/integration/test_all_flags_off_baseline.py
tests/integration/test_fallback_paths.py
```

P6c touched these areas:

```text
nanovllm/kernels/
nanovllm/layers/mixed_kv_fallback.py
nanovllm/engine/model_runner.py
benchmarks/benchmark_serving.py
tests/integration/
tests/kernels/
```

P4a, P4b, P5, and P6a are complete and must remain default-off unless their flags are explicitly enabled. P6b must not change reclaim policy or make EVICT part of B5 defaults.

## Validation Commands

```bash
python -m pytest tests/integration/test_all_flags_off_baseline.py tests/integration/test_fallback_paths.py -v
python benchmarks/report.py --results-dir results --output-markdown docs/benchmark_ablation.md --output-csv results/ablation_summary.csv
python benchmarks/run_ablation_suite.py --output-dir results/ablation --concurrency-sweep 16 --include-optional-evict --plan-only
python -m pytest tests -q
```

## Acceptance Criteria

- P4a/P4b/P5/P6a regression remains green.
- B3/B4 QUANT-only behavior remains the default comparison path.
- P6b fused decode output matches the fallback/reference path within tolerance.
- All optimizer flags remain default-off.
- Release docs and generated ablation summary are present.
- All-flags-off and fallback path release gate tests pass.

## P7 Final Ablation Smoke - 2026-05-17

Final smoke sweep used real executions, not plan-only:

```bash
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 16 --output-json results/final_b0.json --disable-all-optimizer-flags
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 16 --output-json results/b1_scheduler_only.json --enable-memory-aware-scheduler --enable-admission-controller
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b2a_naive_q8.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy naive_age_q8
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b2b_arkv_q8.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b3_profile.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b4_profile.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-triton-gather-dequant --reclaim-policy arkv_q8
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b5_profile.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-mixed-kv-decode-kernel --reclaim-policy arkv_q8
python benchmarks/benchmark_serving.py --workload quality_passkey --concurrency 16 --output-json results/b2c_optional_evict.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-kv-evict --enable-quality-gate --reclaim-policy arkv_q8_evict
python benchmarks/report.py --results-dir results --output-markdown docs/benchmark_ablation.md --output-csv results/ablation_summary.csv
```

Final smoke summary:

- B0: `slo_goodput_tokens_per_s=243.95`, `evicted_blocks=0`.
- B1: `slo_goodput_tokens_per_s=241.47`, `evicted_blocks=0`.
- B2a: `slo_goodput_tokens_per_s=27.68`, `active_quant_blocks=16`, `evicted_blocks=0`.
- B2b: `slo_goodput_tokens_per_s=28.14`, `active_quant_blocks=16`, `evicted_blocks=0`.
- B2c optional: `workload=quality_passkey`, `quality_gate_passed=true`, `evicted_blocks=31`.
- B3: `slo_goodput_tokens_per_s=24.41`, `active_quant_blocks=16`, `evicted_blocks=0`.
- B4: `slo_goodput_tokens_per_s=24.22`, `active_quant_blocks=16`, `evicted_blocks=0`.
- B5: `slo_goodput_tokens_per_s=40.44`, `active_quant_blocks=16`, `evicted_blocks=0`, `fused_kernel_calls=15512`, `fused_kernel_fallbacks=0`.

Post-sweep verification passed:

- `docs/benchmark_ablation.md` and `results/ablation_summary.csv` regenerated.
- `results/ablation_summary.csv` has 8 rows.
- All required summary metrics exist in B0/B1/B2a/B2b/B2c/B3/B4/B5 reports.
- Report represents all 7 risk gates.
- Report states current limitations: block-level/rule-driven ARKV policy, future `attention_mass_ema`/`layer_sensitivity`, and mixed-KV still below original full-cache fast path.

## Relevant Existing Code

Use these files for the next phase:

```text
nanovllm/config.py
nanovllm/kernels/triton_gather_dequant.py
nanovllm/layers/mixed_kv_fallback.py
nanovllm/engine/model_runner.py
benchmarks/benchmark_serving.py
```

## Stop Condition

Stop P6c only when profiling fields are emitted for B3/B4/B5, low-risk hot-path fixes are verified, and B4/B5 remain QUANT-only with `evicted_blocks=0`.

## P6c Status Update - 2026-05-17

Generated P6c artifacts:

```text
nanovllm/engine/profiler.py
results/b3_profile.json
results/b4_profile.json
results/b5_profile.json
```

Implemented:

- Added a default-passive runtime profiler used by benchmark/metrics paths.
- Added profile summary fields: `scheduler_ms`, `admission_ms`, `reclaim_planning_ms`, `quantize_from_full_ms`, `visible_table_build_ms`, `visible_table_tensor_pack_ms`, `workspace_planning_ms`, `gather_dequant_ms`, `mixed_kv_decode_kernel_ms`, `model_forward_non_attention_ms`, `cuda_sync_ms`, `fused_kernel_calls`, `fused_kernel_fallbacks`, `fallback_count`, `parity_check_calls`, `avg_step_ms`.
- Moved P6b visible metadata tensor packing out of per-layer attention calls and into decode preparation, so packed metadata is shared across all layers in the decode step.
- Added packed visible metadata signature caching for unchanged decode batches.
- Changed visible metadata tensor packing to batch on CPU and transfer once instead of issuing per-field CUDA scalar writes.
- Kept parity validation disabled by default; benchmark path reports `parity_check_calls=0` unless explicit validation env vars are set.
- Kept B4/B5 QUANT-only; no EVICT flags are enabled.

Validation passed:

```bash
python -m pytest tests/kernels/test_mixed_kv_decode_attention.py tests/kernels/test_attention_mass_output.py -q
python -m pytest tests/integration/test_decode_mixed_kv_fallback.py -q
python -m pytest tests -q
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b3_profile.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b4_profile.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-triton-gather-dequant --reclaim-policy arkv_q8
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b5_profile.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-mixed-kv-decode-kernel --reclaim-policy arkv_q8
```

Real P6c result:

- Targeted kernel tests: `5 passed`.
- Decode mixed fallback regression: `2 passed`.
- Full suite: `78 passed`.
- B3 profile: `slo_goodput_tokens_per_s=23.04`, `avg_step_ms=199.99`, `fallback_count=15512`, `fused_kernel_calls=0`, `active_quant_blocks=16`, `evicted_blocks=0`.
- B4 profile: `slo_goodput_tokens_per_s=25.16`, `avg_step_ms=183.14`, `fallback_count=15512`, `gather_dequant_ms=8757.12`, `fused_kernel_calls=0`, `active_quant_blocks=16`, `evicted_blocks=0`.
- B5 profile: `slo_goodput_tokens_per_s=38.64`, `avg_step_ms=119.23`, `fused_kernel_calls=15512`, `fused_kernel_fallbacks=0`, `fallback_count=0`, `visible_table_tensor_pack_ms=55.64`, `mixed_kv_decode_kernel_ms=4074.25`, `parity_check_calls=0`, `active_quant_blocks=16`, `evicted_blocks=0`.
- P6c explains the previous B5 gap: before this change, B5 packed Python visible metadata inside every layer's attention call; after caching/step-level packing, B5 dispatches fused decode attention for every mixed-KV decode attention call without fallback.
- Remaining gap is not unsupported shape or parity validation. It is dominated by model forward non-attention/launch overhead and small per-layer fused decode kernels.

## P6b Status Update - 2026-05-17

Generated P6b artifacts:

```text
nanovllm/kernels/mixed_kv_decode_attention.py
tests/kernels/test_mixed_kv_decode_attention.py
tests/kernels/test_attention_mass_output.py
benchmarks/microbench_mixed_kv_decode.py
results/b5_mixed_kv_decode_kernel.json
```

Implemented:

- Decode-only fused mixed-KV attention kernel behind `enable_mixed_kv_decode_kernel=False`.
- Torch reference `mixed_kv_decode_attention_reference()` and support check `mixed_kv_decode_attention_supported()`.
- Runtime dispatch from `Attention`/`ModelRunner` through `mixed_kv_fallback.py`, with fallback to P4/P6a materialization on unsupported shape/device, compile/runtime failure, or optional parity mismatch.
- Encoded visible read view via `visible_entries_to_tensor()`; the kernel reads FULL and QUANT entries from `VisibleBlockEntry` semantics without changing `VisibleBlockTable`.
- Optional block attention mass output behind `enable_attention_mass_output=False`.
- Benchmark CLI flags `--enable-mixed-kv-decode-kernel` and `--enable-attention-mass-output`.
- B5 command uses B3 QUANT-only flags and no EVICT flags.

Validation passed:

```bash
python -m pytest tests/kernels/test_mixed_kv_decode_attention.py -v
python -m pytest tests/kernels/test_attention_mass_output.py -v
python -m pytest tests -q
python benchmarks/microbench_mixed_kv_decode.py --head-dim 128 --block-size 16 --kv-mix-ratio 0.5
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b5_mixed_kv_decode_kernel.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-mixed-kv-decode-kernel --reclaim-policy arkv_q8
```

Real P6b result:

- Mixed decode kernel tests: `3 passed`.
- Attention mass tests: `2 passed`.
- Full suite: `78 passed`.
- CUDA microbench: `triton_supported=true`, `max_abs_diff=6.103515625e-05`, reference `6.3050 ms`, Triton `0.3251 ms`, speedup `19.39x`.
- B5 `long_context_pressure` status `ok`, `active_quant_blocks=16`, `quant_commits_success=16`, `full_blocks_released_after_quant=16`, `mixed_kv_quant_reads=71344`, `visible_quant_entries=16`, `free_full_blocks_reclaim_delta=1`, `evicted_blocks=0`, `oom_requests=0`, `max_stable_concurrency=16`, `slo_goodput_tokens_per_s=10.02`, `raw_peak_vram_bytes=10058285568`.
- P6b did not change `PhysicalBlockMeta`, `SequenceKVRef`, or `VisibleBlockTable` semantics; `slot_mapping` still writes FULL only.
- All optimizer/kernel flags remain default-off.

## P6a Status Update - 2026-05-17

Generated P6a artifacts:

```text
nanovllm/kernels/triton_gather_dequant.py
tests/kernels/test_gather_dequant.py
benchmarks/microbench_gather_dequant.py
results/b4_triton_gather_dequant.json
```

Implemented:

- Triton gather/dequant kernel for Q8 KV blocks, default-off behind `enable_triton_gather_dequant`.
- Torch reference `gather_dequant_reference()` and support check `triton_gather_dequant_supported()`.
- Runtime dispatch from `mixed_kv_fallback.py` with automatic fallback on unsupported shape/device, compile/runtime failure, or optional parity validation mismatch.
- Context and model-runner plumbing so only explicit `--enable-triton-gather-dequant` attempts Triton materialization.
- Microbench for reference vs Triton materialization latency/bandwidth.
- B4 benchmark CLI support using B3 QUANT-only flags; no EVICT flags are enabled.

Validation passed:

```bash
python -m pytest tests/kernels/test_gather_dequant.py -v
python -m pytest tests -q
python benchmarks/microbench_gather_dequant.py --head-dim 128 --block-size 16 --quant-blocks 1024
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b4_triton_gather_dequant.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-triton-gather-dequant --reclaim-policy arkv_q8
```

Real P6a result:

- CUDA P6a kernel tests passed: `3 passed`.
- Full suite passed with CUDA-visible execution: `73 passed`.
- Microbench on CUDA: `triton_supported=true`, `max_abs_diff=0.0`, reference `0.4598 ms / 55.30 GB/s`, Triton `0.0754 ms / 337.16 GB/s`.
- B4 `long_context_pressure` status `ok`, `active_quant_blocks=16`, `mixed_kv_quant_reads=71344`, `visible_quant_entries=16`, `evicted_blocks=0`, `quality_gate_passed=false`, `oom_requests=0`, `max_stable_concurrency=16`, `slo_goodput_tokens_per_s=23.49`, `raw_peak_vram_bytes=10058285568`.
- P6a did not change PhysicalBlockMeta, SequenceKVRef, VisibleBlockTable semantics, slot_mapping writes, or reclaim policy.

## P5 Status Update - 2026-05-17

Generated P5 artifacts:

```text
nanovllm/engine/quality_gate.py
tests/engine/test_evict_policy_guards.py
tests/integration/test_evict_visible_context.py
tests/integration/test_p5_evict_serving_activation.py
tests/quality/test_evict_quality_gate.py
results/p5_quality_gate.json
results/b2c_optional_evict.json
results/b3_reclaim_pressure_arkv_q8_p5_regression.json
```

Implemented:

- P5-only `arkv_q8_evict` policy mode, gated by `allow_evict=True` and passed quality gate.
- QUANT-first EVICT selection with shared-prefix, sink, recent, inflight-write, unfinished-prefill, and direct-FULL guards.
- `ARKVKVManager.apply_evict_transition()` for rollback-safe QUANT->EVICT metadata and visible-table updates.
- VisibleBlockTable EVICT skip semantics: logical refs remain intact while visible spans are compacted.
- Quality gate result schema and benchmark fail-close behavior for `--enable-kv-evict --enable-quality-gate --reclaim-policy arkv_q8_evict`.
- Decode runtime activation for quality-gated QUANT->EVICT; prefill reclaim remains QUANT-only and does not EVICT unfinished prefill.
- Benchmark report fields `quality_gate_passed` and `quality_gate_reason`.

Validation passed:

```bash
python -m pytest tests/engine/test_evict_policy_guards.py tests/integration/test_evict_visible_context.py tests/quality/test_evict_quality_gate.py -v
python -m pytest tests/integration/test_workspace_planning.py tests/integration/test_p5_evict_serving_activation.py -v
python -m pytest tests -q
python benchmarks/benchmark_serving.py --workload quality_passkey --concurrency 8 --output-json results/p5_quality_gate.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-kv-evict --enable-quality-gate --reclaim-policy arkv_q8_evict
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 16 --output-json results/b2c_optional_evict.json --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-kv-evict --enable-quality-gate --reclaim-policy arkv_q8_evict
python benchmarks/benchmark_serving.py --workload b3_reclaim_pressure --concurrency 16 --output-json results/b3_reclaim_pressure_arkv_q8_p5_regression.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8 --require-arkv-q8-reclaim
```

Real P5 result:

- Full suite passed after P5: `70 passed`.
- `quality_passkey` P5 gate status `ok`, `quality_gate_passed=true`, `evicted_blocks=24`, `quant_commits_success=24`, `full_blocks_released_after_quant=24`, `oom_requests=0`, `max_stable_concurrency=8`, `slo_goodput_tokens_per_s=49.73`, `raw_peak_vram_bytes=9634911232`.
- Optional B2c `long_context_pressure` status `ok`, `quality_gate_passed=true`, `evicted_blocks=186`, `quant_commits_success=186`, `full_blocks_released_after_quant=186`, `oom_requests=0`, `max_stable_concurrency=16`, `slo_goodput_tokens_per_s=44.55`, `raw_peak_vram_bytes=10043701760`.
- B3 QUANT-only regression after P5 status `ok`, `active_quant_blocks=16`, `mixed_kv_quant_reads=29232`, `visible_quant_entries=16`, `evicted_blocks=0`, `quality_gate_passed=false`, `slo_goodput_tokens_per_s=47.28`, `raw_peak_vram_bytes=10043808256`.
- EVICT remains default-off and optional; B3/B4/B5 defaults remain QUANT-only.

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

## P4b Status Update - 2026-05-16

Generated P4b artifacts:

```text
tests/integration/test_prefill_prefix_mixed_kv.py
tests/integration/test_unfinished_prefill_mixed_kv.py
tests/integration/test_prefill_chunk_split_workspace.py
results/p4b_shared_prefix_mixed_prefill.json
results/p4b_shared_prefix_mixed_prefill.csv
```

Implemented:

- `enable_prefill_mixed_kv_fallback=False` default-off flag, gated behind `enable_mixed_kv_fallback`
- prefill mixed-KV fallback that reads FULL/QUANT visible entries and dequantizes QUANT through bounded scratch
- prefill query-span metadata so prefix and unfinished-prefill chunks attend causally over the visible prefix plus current FULL writes
- explicit `slot_mapping` validation that rejects writes to QUANT/EVICT/non-FULL blocks
- prefill runtime wiring in `LLMEngine`, `ModelRunner`, `Attention`, and context state
- prefill metadata sync that only registers written/cached context, protects current prefill write blocks as inflight, and allows old prefix QUANT reads
- unfinished-prefill EVICT invariant helper; P4b still generates no EVICT entries
- prefill workspace planning plus chunk split helper for pre-runtime scratch overflow handling
- benchmark CLI flag `--enable-prefill-mixed-kv-fallback`

Validation passed:

```bash
python -m pytest tests/integration/test_prefill_prefix_mixed_kv.py tests/integration/test_unfinished_prefill_mixed_kv.py tests/integration/test_prefill_chunk_split_workspace.py -v
python -m pytest tests -v
python benchmarks/benchmark_serving.py --workload shared_prefix --concurrency 16 --output-json results/p4b_shared_prefix_mixed_prefill.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --enable-prefill-mixed-kv-fallback --reclaim-policy arkv_q8
python benchmarks/benchmark_serving.py --workload b3_reclaim_pressure --concurrency 16 --output-json results/b3_reclaim_pressure_arkv_q8_p4b_regression.json --enable-memory-aware-scheduler --enable-admission-controller --enable-arkv-metadata --enable-kv-q8-runtime --enable-mixed-kv-fallback --reclaim-policy arkv_q8 --require-arkv-q8-reclaim
```

Real P4b result:

- P4b targeted tests passed: `9 passed`.
- Full suite passed after P4b: `57 passed`.
- `shared_prefix` P4b benchmark status `ok`, `throughput_tokens_per_s=13.26`, `slo_goodput_tokens_per_s=13.26`, `active_quant_blocks=16`, `quant_commits_success=16`, `full_blocks_released_after_quant=16`, `mixed_kv_quant_reads=32312`, `visible_quant_entries=16`, `free_full_blocks_reclaim_delta=1`, `evicted_blocks=0`, `raw_peak_vram_bytes=9686218240`.
- P4a reclaim-pressure regression after P4b status `ok`, `active_quant_blocks=16`, `quant_commits_success=16`, `full_blocks_released_after_quant=16`, `mixed_kv_quant_reads=29232`, `visible_quant_entries=16`, `free_full_blocks_reclaim_delta=1`, `evicted_blocks=0`.
- P4b remains a correctness/activation fallback, not a performance optimization. The Python prefill fallback is slow and should be replaced only in later kernel phases.
- Next safe entry is P5 EVICT quality gate. Do not implement P6 acceleration or generate EVICT outside P5.
