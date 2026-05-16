from __future__ import annotations

# 中文说明：
# P0 benchmark 报告汇总器，将 request、KV pool、scheduler admission 和 P2 metadata policy 明细聚合成稳定的 summary 字段，并写出 JSON/CSV。
# 后续消融实验依赖这里的字段名进行对比，因此本文件负责维持报告 schema 的兼容性和默认空指标的安全降级。

import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


REPORT_SCHEMA_VERSION = 1


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def summarize_requests(request_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    ttft = [
        metric["first_token_ts"] - metric["arrival_ts"]
        for metric in request_metrics
        if metric.get("first_token_ts") is not None and metric.get("arrival_ts") is not None
    ]
    finished = [
        metric
        for metric in request_metrics
        if metric.get("finish_ts") is not None and metric.get("arrival_ts") is not None
    ]
    durations = [metric["finish_ts"] - metric["arrival_ts"] for metric in finished]
    output_tokens = sum(metric.get("output_tokens", 0) for metric in request_metrics)
    generated_tokens = sum(
        max(metric.get("output_tokens", 0), 0)
        for metric in request_metrics
        if metric.get("finish_ts") is not None
    )
    tpot_values = []
    for metric in finished:
        output = metric.get("output_tokens", 0)
        first_token_ts = metric.get("first_token_ts")
        finish_ts = metric.get("finish_ts")
        if output > 1 and first_token_ts is not None and finish_ts is not None:
            tpot_values.append((finish_ts - first_token_ts) / (output - 1))
    wall_time = max(durations) if durations else 0.0
    return {
        "request_count": len(request_metrics),
        "finished_requests": len(finished),
        "oom_requests": sum(1 for metric in request_metrics if metric.get("oom")),
        "prompt_tokens": sum(metric.get("prompt_tokens", 0) for metric in request_metrics),
        "output_tokens": output_tokens,
        "generated_tokens": generated_tokens,
        "ttft_s": {
            "mean": mean(ttft) if ttft else None,
            "p50": median(ttft) if ttft else None,
            "p95": percentile(ttft, 0.95),
        },
        "tpot_s": {
            "mean": mean(tpot_values) if tpot_values else None,
            "p50": median(tpot_values) if tpot_values else None,
            "p95": percentile(tpot_values, 0.95),
        },
        "throughput_tokens_per_s": generated_tokens / wall_time if wall_time > 0 else 0.0,
        "wall_time_s": wall_time,
    }


def summarize_kv_pool(kv_pool_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not kv_pool_metrics:
        return {
            "free_full_blocks": None,
            "active_full_blocks": None,
            "active_quant_blocks": 0,
            "evicted_blocks": 0,
            "free_full_block_ratio": None,
            "effective_kv_memory_bytes": 0,
            "raw_peak_vram_bytes": 0,
        }
    latest = kv_pool_metrics[-1]
    return {
        "free_full_blocks": latest["free_full_blocks"],
        "active_full_blocks": latest["active_full_blocks"],
        "active_quant_blocks": latest["active_quant_blocks"],
        "evicted_blocks": latest["evicted_blocks"],
        "free_full_block_ratio": latest["free_full_block_ratio"],
        "effective_kv_memory_bytes": max(metric["effective_kv_memory_bytes"] for metric in kv_pool_metrics),
        "raw_peak_vram_bytes": max(metric["raw_peak_vram_bytes"] for metric in kv_pool_metrics),
    }


def build_report(
    *,
    workload_name: str,
    model: str,
    concurrency: int,
    max_requests: int,
    dry_run: bool,
    request_metrics: list[dict[str, Any]],
    kv_pool_metrics: list[dict[str, Any]],
    scheduler_metrics: list[dict[str, Any]] | None = None,
    metadata_policy_metrics: list[dict[str, Any]] | None = None,
    quant_shadow_metrics: list[dict[str, Any]] | None = None,
    optimizer_flags: dict[str, bool],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    scheduler_metrics = scheduler_metrics or []
    metadata_policy_metrics = metadata_policy_metrics or []
    quant_shadow_metrics = quant_shadow_metrics or []
    admission_summary = {
        "admitted": sum(metric.get("admitted", 0) for metric in scheduler_metrics),
        "admit_after_reclaim": sum(metric.get("admit_after_reclaim", 0) for metric in scheduler_metrics),
        "shrunk": sum(metric.get("shrunk", 0) for metric in scheduler_metrics),
        "deferred": sum(metric.get("deferred", 0) for metric in scheduler_metrics),
        "rejected_temp": sum(metric.get("rejected_temp", 0) for metric in scheduler_metrics),
        "starvation_forced": sum(metric.get("starvation_forced", 0) for metric in scheduler_metrics),
    }
    metadata_policy_summary = {
        "candidate_count": sum(metric.get("candidate_count", 0) for metric in metadata_policy_metrics),
        "conservative_reclaimable_blocks": sum(
            metric.get("conservative_reclaimable_blocks", 0) for metric in metadata_policy_metrics
        ),
        "protected_ratio_max": max(
            (metric.get("protected_ratio", 0.0) for metric in metadata_policy_metrics),
            default=0.0,
        ),
    }
    quant_shadow_summary = {
        "candidate_count": sum(metric.get("candidate_count", 0) for metric in quant_shadow_metrics),
        "potential_reclaimed_full_equiv_blocks": sum(
            metric.get("potential_reclaimed_full_equiv_blocks", 0) for metric in quant_shadow_metrics
        ),
        "quantized_shadow_blocks": sum(metric.get("quantized_shadow_blocks", 0) for metric in quant_shadow_metrics),
        "full_blocks_retained": all(metric.get("full_blocks_retained", True) for metric in quant_shadow_metrics),
        "quant_pool_blocks": max((metric.get("quant_pool_blocks", 0) for metric in quant_shadow_metrics), default=0),
        "full_pool_blocks": max((metric.get("full_pool_blocks", 0) for metric in quant_shadow_metrics), default=0),
        "total_kv_budget_bytes": max((metric.get("total_kv_budget_bytes", 0) for metric in quant_shadow_metrics), default=0),
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "error": error,
        "dry_run": dry_run,
        "workload": {
            "name": workload_name,
            "concurrency": concurrency,
            "max_requests": max_requests,
        },
        "model": model,
        "optimizer_flags": optimizer_flags,
        "request_metrics": request_metrics,
        "kv_pool_metrics": kv_pool_metrics,
        "scheduler_metrics": scheduler_metrics,
        "metadata_policy_metrics": metadata_policy_metrics,
        "quant_shadow_metrics": quant_shadow_metrics,
        "summary": {
            **summarize_requests(request_metrics),
            **summarize_kv_pool(kv_pool_metrics),
            "admission": admission_summary,
            "metadata_policy": metadata_policy_summary,
            "quant_shadow": quant_shadow_summary,
        },
    }


def write_json_report(report: dict[str, Any], output_json: str | Path) -> Path:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_csv_report(report: dict[str, Any], output_json: str | Path) -> Path:
    output_path = Path(output_json).with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    row = {
        "schema_version": report["schema_version"],
        "status": report["status"],
        "dry_run": report["dry_run"],
        "workload": report["workload"]["name"],
        "concurrency": report["workload"]["concurrency"],
        "max_requests": report["workload"]["max_requests"],
        "model": report["model"],
        "ttft_p50_s": summary["ttft_s"]["p50"],
        "tpot_p50_s": summary["tpot_s"]["p50"],
        "throughput_tokens_per_s": summary["throughput_tokens_per_s"],
        "oom_requests": summary["oom_requests"],
        "free_full_blocks": summary["free_full_blocks"],
        "active_full_blocks": summary["active_full_blocks"],
        "active_quant_blocks": summary["active_quant_blocks"],
        "evicted_blocks": summary["evicted_blocks"],
        "free_full_block_ratio": summary["free_full_block_ratio"],
        "effective_kv_memory_bytes": summary["effective_kv_memory_bytes"],
        "raw_peak_vram_bytes": summary["raw_peak_vram_bytes"],
        "admission_deferred": summary["admission"]["deferred"],
        "admission_shrunk": summary["admission"]["shrunk"],
        "admission_rejected_temp": summary["admission"]["rejected_temp"],
        "metadata_policy_candidate_count": summary["metadata_policy"]["candidate_count"],
        "metadata_policy_reclaimable_blocks": summary["metadata_policy"]["conservative_reclaimable_blocks"],
        "metadata_policy_protected_ratio_max": summary["metadata_policy"]["protected_ratio_max"],
        "quant_shadow_reclaimed_full_equiv_blocks": summary["quant_shadow"]["potential_reclaimed_full_equiv_blocks"],
        "quant_shadow_quantized_blocks": summary["quant_shadow"]["quantized_shadow_blocks"],
        "quant_shadow_full_blocks_retained": summary["quant_shadow"]["full_blocks_retained"],
    }
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return output_path
