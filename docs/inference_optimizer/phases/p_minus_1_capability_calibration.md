## P-1：repo capability calibration / smoke spike

### Objective

在进入 P0 之前，冻结当前 Nano-vLLM 的可执行边界：本仓模型支持、KV block size、mixed-KV fallback reference、CUDA graph 策略、本地显存与 benchmark 默认模型。P-1 只做最小 smoke / spike，不引入 optimizer 行为，不改变 full-only baseline。

### Dependencies / Parallelism

```text
Dependencies:
  none

Can run in parallel with:
  documentation cleanup only

Must finish before:
  P0 benchmark harness
  P1 scheduler validation config
  P3 quant pool/block-size decisions
  P4a mixed-KV fallback implementation
```

### Files to Add / Modify

```text
Modify:
  inference_systems_code_plan.md only if calibration changes assumptions
  nanovllm/config.py only if adding non-behavioral validation/logging

Add:
  tests/bench/test_capability_smoke.py
  benchmarks/capability_probe.py
```

### Calibration Questions

```text
1. Which Qwen3 model paths are locally available and loadable on target GPU?
2. What is the largest Qwen3 model / max_model_len / max_num_seqs tuple that can run B0 smoke on RTX 4070 Ti 12GB?
3. Is kvcache_block_size below 256 supported by current config + flash-attn path? If not, benchmark default remains 256.
4. Can a decode-only torch reference materialize visible FULL KV into scratch and match the existing full-only decode output?
5. Does mixed-KV fallback require enforce_eager=True until P6b fused kernel? Default answer is yes unless proven otherwise.
6. Are Llama/TinyLlama/OPT runners available? If not, they stay out of formal ablation.
```

### Key Implementation Steps

1. Run import/config smoke without loading a model.
2. Probe local model path availability and record selected Qwen3 benchmark model.
3. Probe `kvcache_block_size` support for 256 and any smaller candidate only if current config allows it.
4. Build a tiny decode-only materialization reference: FULL cache -> scratch -> attention reference, compared against current full-only decode for controlled tensors.
5. Decide CUDA graph policy: `enable_mixed_kv_fallback=True` implies `enforce_eager=True` until P6b unless a graph-safe path is explicitly validated.
6. Write `benchmarks/capability_probe.py` output JSON with chosen model, block size, max context, eager policy, and skipped assumptions.

### Feature Flags

```text
No optimizer feature flags may be enabled in P-1.
```

### Validation Commands

```bash
python -c "import nanovllm; print(nanovllm.__all__)"
python benchmarks/capability_probe.py --dry-run --output-json results/p_minus_1_capability.json
python -m pytest tests/bench/test_capability_smoke.py -v
```

### Definition of Done

- Formal benchmark model is Qwen3-first and recorded in `results/p_minus_1_capability.json`.
- Formal benchmark block size is recorded; default remains 256 unless smaller sizes are proven valid.
- Mixed-KV fallback reference has a minimal tensor-level parity test or is explicitly marked blocked before P4a.
- `enable_mixed_kv_fallback=True` defaults to eager execution until P6b graph safety is proven.
- Any non-Qwen3 model use is marked optional unless model adapter support exists.

### Failure Modes and Rollback

| Failure mode | Rollback operation | Impact on later phases |
|---|---|---|
| No local model can load | Keep dry-run benchmark only and require model path before B0 | P0 dry-run can proceed, real B0 blocked |
| Smaller block sizes fail | Freeze benchmark block size at 256 | Kernel/microbench commands use calibrated block size |
| Decode materialization reference mismatches | Keep mixed-KV fallback unimplemented and investigate before P3 runtime release | P4a blocked |
| CUDA graph cannot handle dynamic mixed visible tables | Force `enforce_eager=True` for mixed fallback | P4a can proceed, P6b owns graph recovery |
| 7B/8B does not fit 12GB | Use smaller Qwen3 + longer context/concurrency sweep | Headline remains valid if KV pressure is demonstrated |

### Estimated Days

```text
1-2 days
```

### Codex Implementation Prompt

```text
Calibrate the current Nano-vLLM repository before feature work. Produce a dry-run
capability probe that records the formal Qwen3 benchmark model, supported KV block
size, max smoke config, mixed-KV fallback reference status, and CUDA graph/eager
policy. Do not enable optimizer behavior. If smaller block sizes or non-Qwen3
models are unsupported, mark them optional rather than using them in formal
benchmark commands.
```

---
