from __future__ import annotations

from importlib import import_module
from typing import Any


WORKLOADS = {
    "scheduler_stress": "benchmarks.workloads.scheduler_stress",
    "long_context_pressure": "benchmarks.workloads.long_context_pressure",
    "shared_prefix": "benchmarks.workloads.shared_prefix",
    "quality_passkey": "benchmarks.workloads.quality_passkey",
}


def load_workload(name: str):
    module_name = WORKLOADS.get(name)
    if module_name is None:
        raise ValueError(f"unknown workload {name!r}; expected one of {sorted(WORKLOADS)}")
    return import_module(module_name)


def generate_workload(name: str, concurrency: int, max_requests: int, seed: int = 0) -> list[dict[str, Any]]:
    module = load_workload(name)
    return module.generate(concurrency=concurrency, max_requests=max_requests, seed=seed)
