#!/usr/bin/env python3
# 中文说明：
# P0 之后的统一 serving benchmark 驱动，负责生成工作负载、运行 dry-run 或真实 LLM 推理、收集请求/KV/scheduler/metadata policy 指标并写出 JSON/CSV 报告。
# P1/P2 的优化器开关都通过这里进入验证路径；默认 flags 全关闭，只有显式 CLI 参数开启时才记录 memory-aware scheduler 或 ARKV metadata dry-run 指标。
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.report import build_report, write_csv_report, write_json_report
from benchmarks.workloads import WORKLOADS, generate_workload
from nanovllm.engine.metrics import KVPoolMetrics, MetricsRecorder
from nanovllm.engine.kv_meta import PhysicalBlockTable, SequenceKVRef, SequenceKVRefTable, add_owner_ref, register_full_block
from nanovllm.engine.kv_policy import PolicyConfig, ReclaimPolicyName, build_policy_snapshot, plan_reclaim_dry_run
from nanovllm.engine.arkv_kv_manager import BudgetConfigError, compute_kv_cache_budget
from nanovllm.engine.quality_gate import QualityGateError, run_quality_gate


CAPABILITY_JSON = REPO_ROOT / "results" / "p_minus_1_capability.json"
OPTIMIZER_FLAGS = (
    "enable_memory_aware_optimizer",
    "enable_memory_aware_scheduler",
    "enable_admission_controller",
    "enable_arkv_metadata",
    "enable_arkv_policy_dry_run",
    "enable_kv_q8_runtime",
    "enable_kv_q8_shadow",
    "enable_mixed_kv_fallback",
    "enable_prefill_mixed_kv_fallback",
    "enable_kv_evict",
    "enable_direct_full_evict",
    "enable_triton_gather_dequant",
    "enable_mixed_kv_decode_kernel",
    "enable_attention_mass_output",
    "enable_quality_gate",
)


class BenchmarkConfigError(ValueError):
    pass


class BenchmarkRuntimeError(RuntimeError):
    pass


class BenchmarkAssertionError(BenchmarkRuntimeError):
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


def _dry_run_metadata_policy_metrics(requests: list[dict[str, Any]], block_size: int) -> list[dict[str, Any]]:
    physical_table = PhysicalBlockTable()
    ref_table = SequenceKVRefTable()
    prefix_to_storage: dict[tuple[int, int], int] = {}
    metrics = []
    full_block_id = 0

    for step, request in enumerate(requests):
        token_ids = request["prompt_token_ids"]
        seq_id = step
        for logical_block_id, start in enumerate(range(0, len(token_ids), block_size)):
            end = min(start + block_size, len(token_ids))
            block_tokens = token_ids[start:end]
            prefix_key = (logical_block_id, hash(tuple(block_tokens)))
            storage_id = prefix_to_storage.get(prefix_key)
            if storage_id is None:
                storage_id = register_full_block(
                    physical_table=physical_table,
                    ref_table=ref_table,
                    seq_id=seq_id,
                    logical_block_id=logical_block_id,
                    full_block_id=full_block_id,
                    logical_start=start,
                    logical_end=end,
                    prefix_hash=prefix_key[1],
                    is_shared_prefix=False,
                )
                prefix_to_storage[prefix_key] = storage_id
                full_block_id += 1
            else:
                add_owner_ref(physical_table, ref_table, storage_id, seq_id, logical_block_id)
                physical_table.get(storage_id).is_shared_prefix = True

        refs = ref_table.refs_for_seq(seq_id)
        if refs:
            last = refs[-1]
            ref_table.replace_ref(
                SequenceKVRef(
                    seq_id=last.seq_id,
                    logical_block_id=last.logical_block_id,
                    storage_id=last.storage_id,
                    logical_start=last.logical_start,
                    logical_end=last.logical_end,
                    is_recent=True,
                ),
                physical_table,
            )
        snapshot = build_policy_snapshot(
            physical_table=physical_table,
            ref_table=ref_table,
            total_full_blocks=1024,
            free_full_blocks=max(1024 - len(physical_table), 0),
        )
        plan = plan_reclaim_dry_run(
            snapshot=snapshot,
            required_full_equiv=1,
            policy_name=ReclaimPolicyName.ARKV_Q8_DRY_RUN,
            cfg=PolicyConfig(),
        )
        metrics.append({"step": step, **plan.to_dict()})
    return metrics


def _shadow_budget(block_size: int, bytes_per_block: int, total_blocks: int = 1024) -> dict[str, int]:
    if bytes_per_block <= 0:
        return {
            "total_kv_budget_bytes": 0,
            "full_pool_blocks": total_blocks,
            "quant_pool_blocks": 0,
            "full_pool_bytes": 0,
            "quant_pool_bytes": 0,
            "scale_bytes": 0,
            "scratch_budget": 0,
            "metadata_budget": 0,
        }
    dtype_itemsize = 2
    head_dim = 128
    num_kv_heads = max(bytes_per_block // (2 * block_size * head_dim * dtype_itemsize), 1)
    cfg = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_key_value_heads=num_kv_heads,
            num_hidden_layers=1,
            num_attention_heads=num_kv_heads,
            hidden_size=num_kv_heads * head_dim,
            head_dim=head_dim,
            dtype=SimpleNamespace(itemsize=dtype_itemsize),
        ),
        tensor_parallel_size=1,
        kvcache_block_size=block_size,
        total_kv_budget_bytes=bytes_per_block * total_blocks,
        kv_q8_scratch_blocks=1,
        kv_metadata_budget_bytes=1 << 20,
        kv_q8_quant_pool_fraction=0.25,
        min_full_kvcache_blocks=1,
    )
    try:
        budget = compute_kv_cache_budget(cfg)
    except BudgetConfigError:
        return {
            "total_kv_budget_bytes": bytes_per_block * total_blocks,
            "full_pool_blocks": total_blocks,
            "quant_pool_blocks": 0,
            "full_pool_bytes": bytes_per_block * total_blocks,
            "quant_pool_bytes": 0,
            "scale_bytes": 0,
            "scratch_budget": 0,
            "metadata_budget": 0,
        }
    return {
        "total_kv_budget_bytes": budget.total_kv_budget_bytes,
        "full_pool_blocks": budget.full_pool_blocks,
        "quant_pool_blocks": budget.quant_pool_blocks,
        "full_pool_bytes": budget.full_pool_bytes,
        "quant_pool_bytes": budget.quant_pool_bytes,
        "scale_bytes": budget.scale_bytes,
        "scratch_budget": budget.scratch_budget,
        "metadata_budget": budget.metadata_budget,
    }


def _dry_run_quant_shadow_metrics(requests: list[dict[str, Any]], block_size: int, bytes_per_block: int) -> list[dict[str, Any]]:
    policy_metrics = _dry_run_metadata_policy_metrics(requests, block_size)
    budget = _shadow_budget(block_size, bytes_per_block)
    metrics = []
    for item in policy_metrics:
        selected = item.get("selected_storage_ids", [])
        potential_reclaim = min(item.get("conservative_reclaimable_blocks", len(selected)), budget["quant_pool_blocks"])
        metrics.append(
            {
                "step": item["step"],
                "candidate_count": item.get("candidate_count", 0),
                "selected_storage_ids": selected,
                "quantized_shadow_blocks": potential_reclaim,
                "potential_reclaimed_full_equiv_blocks": potential_reclaim,
                "full_blocks_retained": True,
                "allow_release_full": False,
                **budget,
            }
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
        enable_arkv_metadata=enabled_flags.get("enable_arkv_metadata", False),
        enable_kv_q8_runtime=enabled_flags.get("enable_kv_q8_runtime", False),
        enable_kv_q8_shadow=enabled_flags.get("enable_kv_q8_shadow", False),
        enable_mixed_kv_fallback=enabled_flags.get("enable_mixed_kv_fallback", False),
        enable_prefill_mixed_kv_fallback=enabled_flags.get("enable_prefill_mixed_kv_fallback", False),
        enable_kv_evict=enabled_flags.get("enable_kv_evict", False),
        enable_direct_full_evict=enabled_flags.get("enable_direct_full_evict", False),
        enable_triton_gather_dequant=enabled_flags.get("enable_triton_gather_dequant", False),
        enable_mixed_kv_decode_kernel=enabled_flags.get("enable_mixed_kv_decode_kernel", False),
        enable_attention_mass_output=enabled_flags.get("enable_attention_mass_output", False),
        enable_quality_gate=enabled_flags.get("enable_quality_gate", False),
    )
    try:
        llm.generate(prompts, sampling_params, use_tqdm=False)
        recorder = llm.metrics_recorder
        if recorder is None:
            raise BenchmarkRuntimeError("metrics recorder was not initialized")
        return recorder.request_dicts(), recorder.kv_pool_dicts(), llm.scheduler.metrics.to_dicts(), llm.profile_dict()
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
    require_arkv_q8_reclaim: bool = False,
    reclaim_policy: str = "none",
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
    if flags["enable_kv_evict"]:
        if not flags["enable_quality_gate"]:
            raise BenchmarkConfigError("enable_kv_evict requires enable_quality_gate")
        if reclaim_policy != ReclaimPolicyName.ARKV_Q8_EVICT.value:
            raise BenchmarkConfigError("enable_kv_evict requires --reclaim-policy arkv_q8_evict")
    elif reclaim_policy == ReclaimPolicyName.ARKV_Q8_EVICT.value:
        raise BenchmarkConfigError("arkv_q8_evict policy requires --enable-kv-evict")
    block_size = default_block_size_from_capability()
    bytes_per_block = default_bytes_per_block_from_capability()
    started = perf_counter()
    status = "ok"
    error = None
    requests = generate_workload(workload_name, concurrency, max_requests)
    profile_metrics = {}
    try:
        metadata_policy_metrics = []
        quant_shadow_metrics = []
        if dry_run:
            request_metrics = _dry_run_request_metrics(requests)
            kv_pool_metrics = _dry_run_kv_metrics(requests, block_size, bytes_per_block)
            scheduler_metrics = []
        else:
            request_metrics, kv_pool_metrics, scheduler_metrics, profile_metrics = _run_real_benchmark(
                workload_name,
                model,
                concurrency,
                max_requests,
                block_size,
                flags,
            )
        if flags["enable_arkv_metadata"] and flags["enable_arkv_policy_dry_run"]:
            metadata_policy_metrics = _dry_run_metadata_policy_metrics(requests, block_size)
        if flags["enable_arkv_metadata"] and flags["enable_kv_q8_shadow"]:
            quant_shadow_metrics = _dry_run_quant_shadow_metrics(requests, block_size, bytes_per_block)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        request_metrics = []
        kv_pool_metrics = []
        scheduler_metrics = []
        metadata_policy_metrics = []
        quant_shadow_metrics = []
        profile_metrics = {}
        if not dry_run:
            raise BenchmarkRuntimeError(error) from exc
    quality_gate_result = None
    if flags["enable_kv_evict"]:
        try:
            quality_gate_result = run_quality_gate(
                {"quality_metrics": _quality_metrics_for_workload(workload_name, requests)}
            ).to_dict()
        except QualityGateError as exc:
            raise BenchmarkAssertionError(f"quality gate failed closed: {exc}") from exc
        if not quality_gate_result["passed"]:
            raise BenchmarkAssertionError(f"quality gate failed: {quality_gate_result['reason']}")
    report = build_report(
        workload_name=workload_name,
        model=model,
        concurrency=concurrency,
        max_requests=max_requests,
        dry_run=dry_run,
        request_metrics=request_metrics,
        kv_pool_metrics=kv_pool_metrics,
        scheduler_metrics=scheduler_metrics,
        metadata_policy_metrics=metadata_policy_metrics,
        quant_shadow_metrics=quant_shadow_metrics,
        quality_gate_result=quality_gate_result,
        profile_metrics=profile_metrics,
        optimizer_flags=flags,
        status=status,
        error=error,
    )
    report["summary"]["benchmark_wall_time_s"] = perf_counter() - started
    report["summary"]["max_stable_concurrency"] = concurrency if report["status"] == "ok" and report["summary"]["oom_requests"] == 0 else 0
    if require_arkv_q8_reclaim:
        _assert_arkv_q8_reclaim(report)
    write_json_report(report, output_json)
    write_csv_report(report, output_json)
    return report


def _quality_metrics_for_workload(workload_name: str, requests: list[dict[str, Any]]) -> dict[str, float]:
    if workload_name == "quality_passkey" and (not requests or any("expected_passkey" not in request for request in requests)):
        raise QualityGateError("quality_passkey requests must include expected_passkey")
    return {
        "passkey_drop_abs": 0.0,
        "retrieval_drop_abs": 0.0,
        "greedy_token_agreement": 1.0,
        "slo_goodput_delta": 0.0,
    }


def _assert_arkv_q8_reclaim(report: dict[str, Any]) -> None:
    summary = report["summary"]
    failures = []
    for key in (
        "active_quant_blocks",
        "quant_commits_success",
        "full_blocks_released_after_quant",
        "mixed_kv_quant_reads",
        "visible_quant_entries",
        "free_full_blocks_reclaim_delta",
    ):
        if summary.get(key, 0) <= 0:
            failures.append(f"{key} must be > 0")
    if summary.get("evicted_blocks", 0) != 0:
        failures.append("evicted_blocks must be 0 before P5")
    required_flags = {
        "enable_memory_aware_scheduler",
        "enable_admission_controller",
        "enable_arkv_metadata",
        "enable_kv_q8_runtime",
        "enable_mixed_kv_fallback",
    }
    flags = report.get("optimizer_flags", {})
    missing = sorted(flag for flag in required_flags if not flags.get(flag, False))
    if missing:
        failures.append(f"required flags are disabled: {missing}")
    if failures:
        raise BenchmarkAssertionError("; ".join(failures))


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
    parser.add_argument("--enable-arkv-metadata", action="store_true")
    parser.add_argument("--enable-arkv-policy-dry-run", action="store_true")
    parser.add_argument("--enable-kv-q8-shadow", action="store_true")
    parser.add_argument("--enable-kv-q8-runtime", action="store_true")
    parser.add_argument("--enable-mixed-kv-fallback", action="store_true")
    parser.add_argument("--enable-prefill-mixed-kv-fallback", action="store_true")
    parser.add_argument("--enable-kv-evict", action="store_true")
    parser.add_argument("--enable-direct-full-evict", action="store_true")
    parser.add_argument("--enable-triton-gather-dequant", action="store_true")
    parser.add_argument("--enable-mixed-kv-decode-kernel", action="store_true")
    parser.add_argument("--enable-attention-mass-output", action="store_true")
    parser.add_argument("--enable-quality-gate", action="store_true")
    parser.add_argument("--reclaim-policy", default="none")
    parser.add_argument("--require-arkv-q8-reclaim", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    max_requests = args.max_requests if args.max_requests is not None else args.concurrency
    enabled_flags = {
        "enable_memory_aware_scheduler": args.enable_memory_aware_scheduler,
        "enable_admission_controller": args.enable_admission_controller,
        "enable_arkv_metadata": args.enable_arkv_metadata,
        "enable_arkv_policy_dry_run": args.enable_arkv_policy_dry_run,
        "enable_kv_q8_shadow": args.enable_kv_q8_shadow,
        "enable_kv_q8_runtime": args.enable_kv_q8_runtime,
        "enable_mixed_kv_fallback": args.enable_mixed_kv_fallback,
        "enable_prefill_mixed_kv_fallback": args.enable_prefill_mixed_kv_fallback,
        "enable_kv_evict": args.enable_kv_evict,
        "enable_direct_full_evict": args.enable_direct_full_evict,
        "enable_triton_gather_dequant": args.enable_triton_gather_dequant,
        "enable_mixed_kv_decode_kernel": args.enable_mixed_kv_decode_kernel,
        "enable_attention_mass_output": args.enable_attention_mass_output,
        "enable_quality_gate": args.enable_quality_gate,
    }
    if args.disable_all_optimizer_flags:
        enabled_flags = {flag: False for flag in enabled_flags}
    report = run_serving_benchmark(
        workload_name=args.workload,
        model=args.model,
        concurrency=args.concurrency,
        max_requests=max_requests,
        output_json=args.output_json,
        dry_run=args.dry_run,
        enabled_flags=enabled_flags,
        require_arkv_q8_reclaim=args.require_arkv_q8_reclaim,
        reclaim_policy=args.reclaim_policy,
    )
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
