# Memory-Aware Inference Optimizer Overview

This is the compact project overview. It is the first file to read before opening a phase package.

## Source Of Truth

- Root entrypoint: `inference_systems_code_plan.md`
- Current task context: `docs/inference_optimizer/current/current_task_context.md`
- Current phase package: `docs/inference_optimizer/phases/p_minus_1_capability_calibration.md`
- Full archived plan: `docs/inference_optimizer/archive/full_code_plan_2026-05-15.md`
- Handoff log: `omx_wiki/memory-aware-inference-implementation-log.md`

## Goal

Build a memory-aware Nano-VLLM inference optimizer with:

- decode-first memory-aware scheduler
- reclaim-aware admission controller
- ARKV-inspired FULL / QUANT / EVICT KV tiering
- logical / physical / visible KV table separation
- decode-only mixed-KV fallback path for MVP
- quality-gated EVICT
- Triton acceleration paths after correctness is proven
- B0-B5 benchmark and ablation evidence

## Non-Negotiable Decisions

1. All optimizer feature flags default to off.
2. Full-only baseline must always be recoverable.
3. MVP must be a real serving loop, not metadata-only accounting.
4. FULL blocks can be released after quantization only after mixed-KV read correctness is proven.
5. P4a decode mixed-KV and P4b prefill mixed-KV stay split.
6. EVICT is locked to P5 quality gate.
7. B3/B4/B5 default results are QUANT-only, not EVICT-mixed.
8. Current repo is Qwen3-first until model adapter support exists.
9. KV block size defaults to the repo-supported calibrated value, currently expected to be 256 unless P-1 proves smaller values.
10. Mixed-KV fallback defaults to eager execution until graph safety is proven.

## Phase Order

```text
P-1 capability calibration
  -> P0 baseline harness / flags / metrics
  -> P1 scheduler / admission
  -> P2 metadata tables / dry-run policy
  -> P3 QUANT path / budget split / two-phase commit
  -> P4a decode-only mixed-KV fallback
  -> P4b prefill-prefix / unfinished-prefill mixed read
  -> P5 EVICT quality gate
  -> P6a Triton gather/dequant
  -> P6b mixed-KV decode attention kernel
  -> P7 ablation / release hardening
```

## Minimal Context Protocol

For a normal development turn, read only:

1. `docs/inference_optimizer/current/current_task_context.md`
2. the current phase package under `docs/inference_optimizer/phases/`
3. `docs/inference_optimizer/risk_gates.md` only for gates that apply to the current phase
4. exact code files touched by the phase

Read the full archive only when a required invariant, rationale, or historical detail is missing from the split files.

## Current Phase

Current phase: `P-1 capability calibration`.

P-1 means "P minus one"; it comes before P0 and is not P1 scheduler/admission.

Current phase package:

```text
docs/inference_optimizer/phases/p_minus_1_capability_calibration.md
```

Current task context:

```text
docs/inference_optimizer/current/current_task_context.md
```
