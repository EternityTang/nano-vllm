## P0：baseline benchmark harness + feature flag scaffolding + metrics skeleton

### Objective

建立统一 benchmark/metrics 基线，不改变任何算法路径。P0 必须使用 P-1 冻结的模型、block size、eager/graph 策略产出可跑、含 dry-run 的 benchmark 脚本，并记录 B0 baseline 数据。

### Dependencies / Parallelism

```text
Dependencies:
  P-1 capability calibration

Can run in parallel with:
  P2 metadata schema design
  P7 report schema draft

Must finish before:
  P1 validation
  all ablation reporting
```

### Files to Add / Modify

```text
Modify:
  nanovllm/config.py
  nanovllm/engine/llm_engine.py
  nanovllm/engine/scheduler.py
  nanovllm/engine/block_manager.py

Add:
  benchmarks/benchmark_serving.py
  benchmarks/workloads/scheduler_stress.py
  benchmarks/workloads/long_context_pressure.py
  benchmarks/workloads/shared_prefix.py
  benchmarks/workloads/quality_passkey.py
  benchmarks/report.py
  tests/bench/test_metrics_smoke.py
```

### Public Interfaces / Function Signatures

```python
@dataclass
class RequestMetrics:
    request_id: str
    arrival_ts: float
    scheduled_ts: float | None
    first_token_ts: float | None
    finish_ts: float | None
    prompt_tokens: int
    output_tokens: int
    oom: bool
    error: str | None


@dataclass
class KVPoolMetrics:
    step: int
    free_full_blocks: int
    active_full_blocks: int
    active_quant_blocks: int
    evicted_blocks: int
    free_full_block_ratio: float
    effective_kv_memory_bytes: int
    raw_peak_vram_bytes: int


def record_request_event(
    request_id: str,
    event: Literal["arrival", "scheduled", "first_token", "finish", "oom"],
    timestamp: float,
) -> None:
    """Record request lifecycle event.
    Raises:
        MetricsStateError: if event order is invalid.
    """


def collect_kv_pool_metrics(step: int) -> KVPoolMetrics:
    """Collect current KV pool metrics from block manager.
    Raises:
        MetricsUnavailableError: if block manager state cannot be read.
    """


def run_serving_benchmark(
    workload_name: str,
    model: str,
    concurrency: int,
    max_requests: int,
    output_json: str,
    dry_run: bool = False,
) -> dict:
    """Run serving benchmark or dry-run workload generation.
    Raises:
        BenchmarkConfigError: invalid workload or model config.
        BenchmarkRuntimeError: benchmark fails after startup.
    """
```

### Key Implementation Steps

1. Add feature flag scaffolding with all optimizer flags defaulting to `False`.
2. Add metrics hooks for request arrival, scheduled time, first token, finish time, OOM, queue depth, full block stats.
3. Implement dry-run benchmark mode that generates workload and validates config without loading model.
4. Implement B0 baseline command path.
5. Emit JSON and CSV reports with fixed schema.

### Feature Flags

```text
enable_metrics_hooks=False by default
enable_memory_aware_optimizer=False by default
all optimizer flags=False
```

P0 may enable metrics explicitly in benchmark command, but must not change serving semantics.

### Validation Commands

```bash
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 1 --dry-run --output-json /tmp/b0_dryrun.json
python benchmarks/benchmark_serving.py --workload scheduler_stress --concurrency 16 --output-json results/b0_scheduler_stress.json
python benchmarks/benchmark_serving.py --workload long_context_pressure --concurrency 8 --output-json results/b0_long_context.json
python -m pytest tests/bench/test_metrics_smoke.py -v
```

### Definition of Done

- Dry-run benchmark succeeds without loading model.
- B0 baseline produces JSON and CSV with TTFT、TPOT、throughput、OOM、free full blocks、raw peak VRAM fields。
- Closing all optimizer flags preserves baseline output path.
- Metrics hooks do not mutate scheduler decisions.
- At least one B0 scheduler_stress run is saved under `results/`.

### Failure Modes and Rollback

| Failure mode | Rollback operation | Impact on later phases |
|---|---|---|
| Metrics hook changes timing too much | Disable `enable_metrics_hooks`; keep dry-run harness | P1/P2 can continue, but ablation cannot be accepted |
| JSON schema unstable | Freeze schema in `benchmarks/report.py` and regenerate | Later benchmark parsers blocked until fixed |
| Benchmark OOM before workload starts | Lower default dry-run / smoke config only; do not alter algorithm | P0 valid for dry-run, real B0 needs documented smaller config |
| Feature flags accidentally default on | Hard reset defaults to False and add regression test | Blocks all later merge gates |

### Estimated Days

```text
2-4 days
```

### Codex Implementation Prompt

```text
Implement baseline observability for Nano-vLLM without changing serving semantics.
Add request lifecycle timestamps, KV/block/memory stats, feature flag scaffolding,
and a repeatable benchmark harness that outputs TTFT/TPOT/throughput/VRAM/OOM
in JSON and CSV. Include dry-run mode. Keep all optimizer behavior disabled by
default and preserve the full-only baseline path.
```

---
