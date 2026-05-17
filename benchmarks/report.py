from __future__ import annotations

# 中文说明：
# P0 benchmark 报告汇总器，将 request、KV pool、scheduler admission 和 P2 metadata policy 明细聚合成稳定的 summary 字段，并写出 JSON/CSV。
# 后续消融实验依赖这里的字段名进行对比，因此本文件负责维持报告 schema 的兼容性和默认空指标的安全降级。

import csv
import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


REPORT_SCHEMA_VERSION = 1

ABLATION_FILES = (
    ("B0", "Baseline", ("final_b0.json", "b0_scheduler_stress.json")),
    ("B1", "Scheduler only", ("b1_scheduler_only.json",)),
    ("B2a", "Naive Q8", ("b2a_naive_q8.json",)),
    ("B2b", "ARKV Q8", ("b2b_arkv_q8.json",)),
    ("B2c", "Optional EVICT", ("b2c_optional_evict.json",)),
    ("B3", "Scheduler + ARKV Q8", ("b3_profile.json", "b3_scheduler_arkv_q8.json")),
    ("B4", "B3 + Triton gather/dequant", ("b4_profile.json", "b4_triton_gather_dequant.json")),
    ("B5", "B3 + fused mixed-KV decode", ("b5_profile.json", "b5_mixed_kv_decode_kernel.json")),
)


class ReportSchemaError(ValueError):
    pass


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
        "slo_goodput_tokens_per_s": generated_tokens / wall_time if wall_time > 0 else 0.0,
        "max_stable_concurrency": None,
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
            "quantized_block_ratio": 0.0,
            "reclaim_trigger_count": 0,
            "quant_commits_success": 0,
            "quant_commits_rollback": 0,
            "full_blocks_released_after_quant": 0,
            "mixed_kv_quant_reads": 0,
            "visible_quant_entries": 0,
            "free_full_blocks_before_reclaim": 0,
            "free_full_blocks_after_reclaim": 0,
            "free_full_blocks_reclaim_delta": 0,
        }
    latest = kv_pool_metrics[-1]
    active_quant_max = max(metric.get("active_quant_blocks", 0) for metric in kv_pool_metrics)
    return {
        "free_full_blocks": latest["free_full_blocks"],
        "active_full_blocks": latest["active_full_blocks"],
        "active_quant_blocks": active_quant_max,
        "evicted_blocks": latest["evicted_blocks"],
        "free_full_block_ratio": latest["free_full_block_ratio"],
        "effective_kv_memory_bytes": max(metric["effective_kv_memory_bytes"] for metric in kv_pool_metrics),
        "raw_peak_vram_bytes": max(metric["raw_peak_vram_bytes"] for metric in kv_pool_metrics),
        "quantized_block_ratio": max(metric.get("quantized_block_ratio", 0.0) for metric in kv_pool_metrics),
        "reclaim_trigger_count": max(metric.get("reclaim_trigger_count", 0) for metric in kv_pool_metrics),
        "quant_commits_success": max(metric.get("quant_commits_success", 0) for metric in kv_pool_metrics),
        "quant_commits_rollback": max(metric.get("quant_commits_rollback", 0) for metric in kv_pool_metrics),
        "full_blocks_released_after_quant": max(
            metric.get("full_blocks_released_after_quant", 0) for metric in kv_pool_metrics
        ),
        "mixed_kv_quant_reads": max(metric.get("mixed_kv_quant_reads", 0) for metric in kv_pool_metrics),
        "visible_quant_entries": max(metric.get("visible_quant_entries", 0) for metric in kv_pool_metrics),
        "free_full_blocks_before_reclaim": max(
            metric.get("free_full_blocks_before_reclaim", 0) for metric in kv_pool_metrics
        ),
        "free_full_blocks_after_reclaim": max(
            metric.get("free_full_blocks_after_reclaim", 0) for metric in kv_pool_metrics
        ),
        "free_full_blocks_reclaim_delta": max(
            metric.get("free_full_blocks_reclaim_delta", 0) for metric in kv_pool_metrics
        ),
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
    quality_gate_result: dict[str, Any] | None = None,
    profile_metrics: dict[str, Any] | None = None,
    optimizer_flags: dict[str, bool],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    scheduler_metrics = scheduler_metrics or []
    metadata_policy_metrics = metadata_policy_metrics or []
    quant_shadow_metrics = quant_shadow_metrics or []
    profile_metrics = profile_metrics or {}
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
        "quality_gate": quality_gate_result,
        "profile_metrics": profile_metrics,
        "summary": {
            **summarize_requests(request_metrics),
            **summarize_kv_pool(kv_pool_metrics),
            **_profile_summary(profile_metrics),
            "admission": admission_summary,
            "metadata_policy": metadata_policy_summary,
            "quant_shadow": quant_shadow_summary,
            "quality_gate_passed": bool(quality_gate_result.get("passed", False)) if quality_gate_result else False,
            "quality_gate_reason": quality_gate_result.get("reason") if quality_gate_result else None,
        },
    }


def _profile_summary(profile_metrics: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "scheduler_ms",
        "admission_ms",
        "reclaim_planning_ms",
        "quantize_from_full_ms",
        "visible_table_build_ms",
        "visible_table_tensor_pack_ms",
        "workspace_planning_ms",
        "gather_dequant_ms",
        "mixed_kv_decode_kernel_ms",
        "model_forward_non_attention_ms",
        "cuda_sync_ms",
        "fused_kernel_calls",
        "fused_kernel_fallbacks",
        "fallback_count",
        "parity_check_calls",
        "avg_step_ms",
    )
    return {field: profile_metrics.get(field, 0) for field in fields}


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
        "slo_goodput_tokens_per_s": summary["slo_goodput_tokens_per_s"],
        "max_stable_concurrency": summary["max_stable_concurrency"],
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
        "quantized_block_ratio": summary["quantized_block_ratio"],
        "reclaim_trigger_count": summary["reclaim_trigger_count"],
        "quant_commits_success": summary["quant_commits_success"],
        "quant_commits_rollback": summary["quant_commits_rollback"],
        "full_blocks_released_after_quant": summary["full_blocks_released_after_quant"],
        "mixed_kv_quant_reads": summary["mixed_kv_quant_reads"],
        "visible_quant_entries": summary["visible_quant_entries"],
        "free_full_blocks_before_reclaim": summary["free_full_blocks_before_reclaim"],
        "free_full_blocks_after_reclaim": summary["free_full_blocks_after_reclaim"],
        "free_full_blocks_reclaim_delta": summary["free_full_blocks_reclaim_delta"],
        "quality_gate_passed": summary["quality_gate_passed"],
        "quality_gate_reason": summary["quality_gate_reason"],
        "scheduler_ms": summary["scheduler_ms"],
        "admission_ms": summary["admission_ms"],
        "reclaim_planning_ms": summary["reclaim_planning_ms"],
        "quantize_from_full_ms": summary["quantize_from_full_ms"],
        "visible_table_build_ms": summary["visible_table_build_ms"],
        "visible_table_tensor_pack_ms": summary["visible_table_tensor_pack_ms"],
        "workspace_planning_ms": summary["workspace_planning_ms"],
        "gather_dequant_ms": summary["gather_dequant_ms"],
        "mixed_kv_decode_kernel_ms": summary["mixed_kv_decode_kernel_ms"],
        "model_forward_non_attention_ms": summary["model_forward_non_attention_ms"],
        "cuda_sync_ms": summary["cuda_sync_ms"],
        "fused_kernel_calls": summary["fused_kernel_calls"],
        "fused_kernel_fallbacks": summary["fused_kernel_fallbacks"],
        "fallback_count": summary["fallback_count"],
        "parity_check_calls": summary["parity_check_calls"],
        "avg_step_ms": summary["avg_step_ms"],
    }
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return output_path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "summary" not in data:
        raise ReportSchemaError(f"{path} missing summary")
    return data


def _find_ablation_report(results_dir: Path, candidates: tuple[str, ...]) -> tuple[Path, dict[str, Any]] | None:
    for name in candidates:
        path = results_dir / name
        if path.is_file():
            return path, _load_json(path)
    return None


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ablation_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, name, candidates in ABLATION_FILES:
        found = _find_ablation_report(results_dir, candidates)
        if found is None:
            rows.append({"label": label, "name": name, "status": "missing", "file": "-"})
            continue
        path, report = found
        summary = report["summary"]
        rows.append(
            {
                "label": label,
                "name": name,
                "file": str(path),
                "status": report.get("status"),
                "workload": report.get("workload", {}).get("name"),
                "concurrency": report.get("workload", {}).get("concurrency"),
                "throughput_tokens_per_s": summary.get("throughput_tokens_per_s"),
                "slo_goodput_tokens_per_s": summary.get("slo_goodput_tokens_per_s"),
                "oom_requests": summary.get("oom_requests"),
                "max_stable_concurrency": summary.get("max_stable_concurrency"),
                "raw_peak_vram_bytes": summary.get("raw_peak_vram_bytes"),
                "active_quant_blocks": summary.get("active_quant_blocks"),
                "quantized_block_ratio": summary.get("quantized_block_ratio"),
                "quant_commits_success": summary.get("quant_commits_success"),
                "full_blocks_released_after_quant": summary.get("full_blocks_released_after_quant"),
                "mixed_kv_quant_reads": summary.get("mixed_kv_quant_reads"),
                "visible_quant_entries": summary.get("visible_quant_entries"),
                "evicted_blocks": summary.get("evicted_blocks"),
                "quality_gate_passed": summary.get("quality_gate_passed"),
                "fused_kernel_calls": summary.get("fused_kernel_calls"),
                "fused_kernel_fallbacks": summary.get("fused_kernel_fallbacks"),
                "fallback_count": summary.get("fallback_count"),
                "avg_step_ms": summary.get("avg_step_ms"),
            }
        )
    return rows


def generate_ablation_report(results_dir: str | Path, output_markdown: str | Path, output_csv: str | Path) -> None:
    results_path = Path(results_dir)
    rows = _ablation_rows(results_path)
    output_markdown = Path(output_markdown)
    output_csv = Path(output_csv)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    csv_fields = [
        "label",
        "name",
        "file",
        "status",
        "workload",
        "concurrency",
        "throughput_tokens_per_s",
        "slo_goodput_tokens_per_s",
        "oom_requests",
        "max_stable_concurrency",
        "raw_peak_vram_bytes",
        "active_quant_blocks",
        "quantized_block_ratio",
        "quant_commits_success",
        "full_blocks_released_after_quant",
        "mixed_kv_quant_reads",
        "visible_quant_entries",
        "evicted_blocks",
        "quality_gate_passed",
        "fused_kernel_calls",
        "fused_kernel_fallbacks",
        "fallback_count",
        "avg_step_ms",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Memory-Aware Optimizer 消融报告",
        "",
        "本报告由 benchmark JSON artifacts 生成。B2c 是 optional 且必须通过 quality gate；B4/B5 默认基于 B3 QUANT-only，必须保持 `evicted_blocks=0`。",
        "",
        "## 汇总",
        "",
        "| 组别 | 名称 | 状态 | Workload | Throughput tok/s | SLO-goodput tok/s | OOM | Active QUANT | EVICT | Fused calls | Fused fallbacks | Fallback count | Avg step ms |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {name} | {status} | {workload} | {throughput} | {slo} | {oom} | {active_quant} | {evicted} | {fused_calls} | {fused_fallbacks} | {fallback_count} | {avg_step} |".format(
                label=row["label"],
                name=row["name"],
                status=row.get("status", "-"),
                workload=row.get("workload", "-"),
                throughput=_fmt(row.get("throughput_tokens_per_s")),
                slo=_fmt(row.get("slo_goodput_tokens_per_s")),
                oom=_fmt(row.get("oom_requests"), 0),
                active_quant=_fmt(row.get("active_quant_blocks"), 0),
                evicted=_fmt(row.get("evicted_blocks"), 0),
                fused_calls=_fmt(row.get("fused_kernel_calls"), 0),
                fused_fallbacks=_fmt(row.get("fused_kernel_fallbacks"), 0),
                fallback_count=_fmt(row.get("fallback_count"), 0),
                avg_step=_fmt(row.get("avg_step_ms")),
            )
        )
    lines.extend(
        [
            "",
            "## Release Gates",
            "",
            "1. All optimizer feature flags default off.",
            "2. Full-only fallback path always available.",
            "3. Logical / physical / visible table invariants preserved.",
            "4. FULL->QUANT uses rollback-safe two-phase commit.",
            "5. KV budget accounting uses fixed `total_kv_budget_bytes` split.",
            "6. EVICT is locked to P5 optional quality gate; B4/B5 default to QUANT-only.",
            "7. Kernel paths require torch reference parity and automatic fallback.",
            "",
            "## 当前限制",
            "",
            "- 当前 ARKV policy 是 block-level / rule-driven，还不是训练出的动态策略。",
            "- `attention_mass_ema` 和 `layer_sensitivity` 仍是后续工作，不属于当前 release headline。",
            "- P6c 后 mixed-KV path 明显优于 fallback，但仍未超过原始 full-cache fast path；当前收益重点是容量、reclaim 激活和 QUANT-only kernel path 的恢复，而不是全面替代 full-cache 快路径。",
            "",
            "## P6c Profile 解释",
            "",
            "B5 profile 中 `fused_kernel_calls > 0` 且 `fused_kernel_fallbacks == 0` 证明 fused kernel 已真实 dispatch。剩余 serving gap 应结合 `model_forward_non_attention_ms`、`avg_step_ms` 和小 kernel launch overhead 解释，而不是归因于 unsupported shape 或 parity check。",
            "",
        ]
    )
    output_markdown.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Nano-VLLM ablation reports from benchmark JSON files.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generate_ablation_report(args.results_dir, args.output_markdown, args.output_csv)
    print(json.dumps({"status": "ok", "output_markdown": args.output_markdown, "output_csv": args.output_csv}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
