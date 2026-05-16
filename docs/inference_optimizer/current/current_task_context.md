# Current Task Context

Last updated: 2026-05-15

## Current Phase

```text
P-1 repo capability calibration / smoke spike
```

P-1 means "P minus one"; it comes before P0. Do not confuse it with P1 scheduler/admission.

Do not start P0, P1 scheduler, metadata, quantization, or attention backend work until P-1 is complete.

## Read These Files First

```text
inference_systems_code_plan.md
docs/inference_optimizer/overview.md
docs/inference_optimizer/phases/p_minus_1_capability_calibration.md
omx_wiki/memory-aware-inference-implementation-log.md
```

Only read the full archive if these files do not answer a required question:

```text
docs/inference_optimizer/archive/full_code_plan_2026-05-15.md
```

## Target Result

Produce a dry-run capability probe that records:

- formal Qwen3 benchmark model
- supported KV block size
- maximum smoke config
- mixed-KV fallback reference status
- CUDA graph / eager policy

## Expected Files

```text
benchmarks/capability_probe.py
tests/bench/test_capability_smoke.py
results/p_minus_1_capability.json
```

Optional only if needed:

```text
nanovllm/config.py
```

Only add non-behavioral validation/logging to `config.py` during P-1.

## Validation Commands

```bash
python -c "import nanovllm; print(nanovllm.__all__)"
python benchmarks/capability_probe.py --dry-run --output-json results/p_minus_1_capability.json
python -m pytest tests/bench/test_capability_smoke.py -v
```

## Acceptance Criteria

- Formal benchmark model is Qwen3-first and recorded in `results/p_minus_1_capability.json`.
- Formal benchmark block size is recorded.
- Default block size remains 256 unless smaller sizes are proven valid.
- Mixed-KV fallback reference has a minimal tensor-level parity test or is explicitly marked blocked before P4a.
- `enable_mixed_kv_fallback=True` implies eager execution until P6b proves graph safety.
- Non-Qwen3 models are optional unless model adapter support exists.
- No optimizer behavior is enabled.

## Relevant Existing Code

Use these files for P-1 repo checks:

```text
nanovllm/config.py
nanovllm/engine/model_runner.py
nanovllm/layers/attention.py
nanovllm/engine/scheduler.py
nanovllm/engine/block_manager.py
nanovllm/utils/context.py
example.py
bench.py
```

## Stop Condition

Stop P-1 only when the probe, test, and result JSON exist and the validation commands pass, or when a real blocker is documented in this file and in the probe output.

## P-1 Status Update - 2026-05-16

Generated P-1 dry-run artifacts:

```text
benchmarks/capability_probe.py
tests/bench/test_capability_smoke.py
results/p_minus_1_capability.json
```

Dry-run calibration recorded:

- formal benchmark model: `/home/tang/models/Qwen3-4B`
- formal KV block size: `256`
- smaller block sizes: unsupported by current `Config.__post_init__`
- mixed-KV fallback reference: CPU tensor-level materialization/decode parity passed
- CUDA graph policy: mixed-KV fallback requires eager execution until P6b proves graph safety
- optimizer behavior enabled: no

Current blockers recorded in `results/p_minus_1_capability.json`:

- active `python` is `3.13.12`, outside project `>=3.10,<3.13`
- `flash-attn` is not installed in the active interpreter
- `pytest` is not installed in the active interpreter
- `nvidia-smi` cannot access NVML, so real GPU/model B0 smoke is blocked

Do not start P0 real benchmark claims until these environment blockers are cleared or an equivalent validated environment is used.
