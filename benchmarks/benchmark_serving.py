#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.report import build_report, write_csv_report, write_json_report
from benchmarks.workloads import WORKLOADS, generate_workload
from nanovllm.engine.metrics import KVPoolMetrics, MetricsRecorder


CAPABILITY_JSON = REPO_ROOT / "results" / "p_minus_1_capability.json"
OPTIMIZER_FLAGS = (
    "enable_memory_aware_optimizer",
    "enable_memory_aware_scheduler",
    "enable_admission_controller",
    "enable_arkv_metadata",
    "enable_kv_q8_runtime",
    "enable_kv_q8_shadow",
    "enable_mixed_kv_fallback",
    "enable_kv_evict",
    "enable_direct_full_evict",
    "enable_triton_gather_dequant",
    "enable_mixed_kv_decode_kernel",
    "enable_quality_gate",
)


class BenchmarkConfigError(ValueError):
    pass


class BenchmarkRuntimeError(RuntimeError):
    pass


def default_model_from_capability() -> str | None:
    if not CAPABILITY_JSON.is_file():
        return None
    with CAPABILITY_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("model_support", {}).get("formal_benchmark_model", {}).get("path")


def default_block_size_from_capability() -> int:
    if not CAPABILITY_JSON.is_file():
        return 256
    with CAPABILITY_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("kv_cache", {}).get("formal_block_size", 256))


def default_bytes_per_block_from_capability() -> int:
    if not CAPABILITY_JSON.is_file():
        return 0
    with CAPABILITY_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    block_size = int(data.get("kv_cache", {}).get("formal_block_size", 256))
    formal_path = data.get("model_support", {}).get("formal_benchmark_model", {}).get("path")
    models = data.get("model_support", {}).get("available_qwen3_models", [])
    model = next((item for item in models if item.get("path") == formal_path), None)
    if model is None:
        return 0
    dtype = str(model.get("torch_dtype", "")).lower()
    dtype_bytes = 2 if dtype in {"float16", "bfloat16", "torch.float16", "torch.bfloat16"} else 4
    return (
        2
        * int(model.get("num_hidden_layers") or 0)
        * block_size
        * int(model.get("num_key_value_heads") or 0)
        * int(model.get("hidden_size") or 0)
        // max(int(model.get("num_attention_heads") or 1), 1)
        * dtype_bytes
    )


def optimizer_flags(enabled: dict[str, bool] | None = None) -> dict[str, bool]:
    enabled = enabled or {}
    return {flag: bool(enabled.get(flag, False)) for flag in OPTIMIZER_FLAGS}


def _dry_run_request_metrics(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recorder = MetricsRecorder()
    for request in requests:
        request_id = request["request_id"]
        arrival_ts = float(request["arrival_ts"])
        output_tokens = int(request["output_tokens"])
        recorder.add_request(
            request_id,
            prompt_tokens=len(request["prompt_token_ids"]),
            output_tokens=output_tokens,
            arrival_ts=arrival_ts,
        )
        scheduled_ts = arrival_ts + 0.001
        first_token_ts = scheduled_ts + 0.002
        finish_ts = first_token_ts + output_tokens * 0.001
        recorder.record_request_event(request_id, "scheduled", scheduled_ts)
        recorder.record_request_event(request_id, "first_token", first_token_ts)
        recorder.record_request_event(request_id, "finish", finish_ts)
    return recorder.request_dicts()


def _dry_run_kv_metrics(requests: list[dict[str, Any]], block_size: int, bytes_per_block: int) -> list[dict[str, Any]]:
    active_blocks = 0
    total_blocks = 1024
    metrics = []
    for step, request in enumerate(requests):
        prompt_blocks = (len(request["prompt_token_ids"]) + block_size - 1) // block_size
        active_blocks = min(total_blocks, active_blocks + prompt_blocks)
        free_blocks = max(total_blocks - active_blocks, 0)
        metrics.append(
            KVPoolMetrics(
                step=step,
                free_full_blocks=free_blocks,
                active_full_blocks=active_blocks,
                active_quant_blocks=0,
                evicted_blocks=0,
                free_full_block_ratio=free_blocks / total_blocks,
                effective_kv_memory_bytes=active_blocks * bytes_per_block,
                raw_peak_vram_bytes=0,
            ).to_dict()
        )
    return metrics


def _run_real_benchmark(
    workload_name: str,
    model: str,
    concurrency: int,
    max_requests: int,
    block_size: int,
    enabled_flags: dict[str, bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from nanovllm import LLM, SamplingParams

    requests = generate_workload(workload_name, concurrency, max_requests)
    prompts = [request["prompt_token_ids"] for request in requests]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=request["output_tokens"])
        for request in requests
    ]
    llm = LLM(
        model,
        enable_metrics_hooks=True,
        kvcache_block_size=block_size,
        enforce_eager=True,
        enable_memory_aware_scheduler=enabled_flags.get("enable_memory_aware_scheduler", False),
        enable_admission_controller=enabled_flags.get("enable_admission_controller", False),
    )
    try:
        llm.generate(prompts, sampling_params, use_tqdm=False)
        recorder = llm.metrics_recorder
        if recorder is None:
            raise BenchmarkRuntimeError("metrics recorder was not initialized")
        return recorder.request_dicts(), recorder.kv_pool_dicts(), llm.scheduler.metrics.to_dicts()
    finally:
        llm.exit()


def run_serving_benchmark(
    workload_name: str,
    model: str | None,
    concurrency: int,
    max_requests: int,
    output_json: str,
    dry_run: bool = False,
    enabled_flags: dict[str, bool] | None = None,
) -> dict:
    if workload_name not in WORKLOADS:
        raise BenchmarkConfigError(f"unknown workload {workload_name!r}")
    if concurrency < 1:
        raise BenchmarkConfigError("concurrency must be >= 1")
    if max_requests < 1:
        raise BenchmarkConfigError("max_requests must be >= 1")
    model = model or default_model_from_capability()
    if not model:
        raise BenchmarkConfigError("model was not provided and P-1 capability JSON has no formal model")

    flags = optimizer_flags(enabled_flags)
    block_size = default_block_size_from_capability()
    bytes_per_block = default_bytes_per_block_from_capability()
    started = perf_counter()
    status = "ok"
    error = None
    requests = generate_workload(workload_name, concurrency, max_requests)
    try:
        if dry_run:
            request_metrics = _dry_run_request_metrics(requests)
            kv_pool_metrics = _dry_run_kv_metrics(requests, block_size, bytes_per_block)
            scheduler_metrics = []
        else:
            request_metrics, kv_pool_metrics, scheduler_metrics = _run_real_benchmark(
                workload_name,
                model,
                concurrency,
                max_requests,
                block_size,
                flags,
            )
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        request_metrics = []
        kv_pool_metrics = []
        scheduler_metrics = []
        if not dry_run:
            raise BenchmarkRuntimeError(error) from exc
    report = build_report(
        workload_name=workload_name,
        model=model,
        concurrency=concurrency,
        max_requests=max_requests,
        dry_run=dry_run,
        request_metrics=request_metrics,
        kv_pool_metrics=kv_pool_metrics,
        scheduler_metrics=scheduler_metrics,
        optimizer_flags=flags,
        status=status,
        error=error,
    )
    report["summary"]["benchmark_wall_time_s"] = perf_counter() - started
    write_json_report(report, output_json)
    write_csv_report(report, output_json)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nano-VLLM baseline serving benchmark harness.")
    parser.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-all-optimizer-flags", action="store_true")
    parser.add_argument("--enable-memory-aware-scheduler", action="store_true")
    parser.add_argument("--enable-admission-controller", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    max_requests = args.max_requests if args.max_requests is not None else args.concurrency
    report = run_serving_benchmark(
        workload_name=args.workload,
        model=args.model,
        concurrency=args.concurrency,
        max_requests=max_requests,
        output_json=args.output_json,
        dry_run=args.dry_run,
        enabled_flags={
            "enable_memory_aware_scheduler": args.enable_memory_aware_scheduler,
            "enable_admission_controller": args.enable_admission_controller,
        },
    )
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
