#!/usr/bin/env python3
"""Run the P7 B0-B5 ablation command suite.

This script is intentionally thin: it preserves the benchmark_serving.py command
surface as the source of truth and only sequences reproducible release runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "benchmarks" / "benchmark_serving.py"


def _commands(model: str | None, output_dir: Path, concurrency: int, include_optional_evict: bool) -> list[tuple[str, list[str]]]:
    base = [sys.executable, str(BENCHMARK)]
    model_args = ["--model", model] if model else []
    specs = [
        (
            "b0",
            [
                "--workload",
                "scheduler_stress",
                "--concurrency",
                str(concurrency),
                "--output-json",
                str(output_dir / f"b0_c{concurrency}.json"),
                "--disable-all-optimizer-flags",
            ],
        ),
        (
            "b1",
            [
                "--workload",
                "scheduler_stress",
                "--concurrency",
                str(concurrency),
                "--output-json",
                str(output_dir / f"b1_c{concurrency}.json"),
                "--enable-memory-aware-scheduler",
                "--enable-admission-controller",
            ],
        ),
        (
            "b2a",
            [
                "--workload",
                "long_context_pressure",
                "--concurrency",
                str(concurrency),
                "--output-json",
                str(output_dir / f"b2a_c{concurrency}.json"),
                "--enable-arkv-metadata",
                "--enable-kv-q8-runtime",
                "--enable-mixed-kv-fallback",
                "--reclaim-policy",
                "naive_age_q8",
            ],
        ),
        (
            "b2b",
            [
                "--workload",
                "long_context_pressure",
                "--concurrency",
                str(concurrency),
                "--output-json",
                str(output_dir / f"b2b_c{concurrency}.json"),
                "--enable-arkv-metadata",
                "--enable-kv-q8-runtime",
                "--enable-mixed-kv-fallback",
                "--reclaim-policy",
                "arkv_q8",
            ],
        ),
        (
            "b3",
            [
                "--workload",
                "long_context_pressure",
                "--concurrency",
                str(concurrency),
                "--output-json",
                str(output_dir / f"b3_c{concurrency}.json"),
                "--enable-memory-aware-scheduler",
                "--enable-admission-controller",
                "--enable-arkv-metadata",
                "--enable-kv-q8-runtime",
                "--enable-mixed-kv-fallback",
                "--reclaim-policy",
                "arkv_q8",
            ],
        ),
        (
            "b4",
            [
                "--workload",
                "long_context_pressure",
                "--concurrency",
                str(concurrency),
                "--output-json",
                str(output_dir / f"b4_c{concurrency}.json"),
                "--enable-memory-aware-scheduler",
                "--enable-admission-controller",
                "--enable-arkv-metadata",
                "--enable-kv-q8-runtime",
                "--enable-mixed-kv-fallback",
                "--enable-triton-gather-dequant",
                "--reclaim-policy",
                "arkv_q8",
            ],
        ),
        (
            "b5",
            [
                "--workload",
                "long_context_pressure",
                "--concurrency",
                str(concurrency),
                "--output-json",
                str(output_dir / f"b5_c{concurrency}.json"),
                "--enable-memory-aware-scheduler",
                "--enable-admission-controller",
                "--enable-arkv-metadata",
                "--enable-kv-q8-runtime",
                "--enable-mixed-kv-fallback",
                "--enable-mixed-kv-decode-kernel",
                "--reclaim-policy",
                "arkv_q8",
            ],
        ),
    ]
    if include_optional_evict:
        specs.append(
            (
                "b2c",
                [
                    "--workload",
                    "long_context_pressure",
                    "--concurrency",
                    str(concurrency),
                    "--output-json",
                    str(output_dir / f"b2c_c{concurrency}.json"),
                    "--enable-arkv-metadata",
                    "--enable-kv-q8-runtime",
                    "--enable-mixed-kv-fallback",
                    "--enable-kv-evict",
                    "--enable-quality-gate",
                    "--reclaim-policy",
                    "arkv_q8_evict",
                ],
            )
        )
    return [(label, base + spec + model_args) for label, spec in specs]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P7 B0-B5 ablation suite.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency-sweep", default="16")
    parser.add_argument("--include-optional-evict", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    concurrencies = [int(item) for item in args.concurrency_sweep.split(",") if item]
    commands = [
        {"label": label, "command": command}
        for concurrency in concurrencies
        for label, command in _commands(args.model, output_dir, concurrency, args.include_optional_evict)
    ]
    if args.plan_only:
        print(json.dumps({"status": "planned", "commands": commands}, indent=2))
        return 0
    for item in commands:
        subprocess.run(item["command"], cwd=REPO_ROOT, check=True)
    print(json.dumps({"status": "ok", "commands_run": len(commands), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
