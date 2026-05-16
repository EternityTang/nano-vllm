from __future__ import annotations

# 中文说明：
# benchmark workload 注册表和动态加载入口，把 workload 名称映射到具体生成器模块，供 benchmark_serving.py 统一调用。
# P0/P1/P2 的 dry-run 与真实 benchmark 都通过该入口获得确定性请求集合，避免各阶段直接依赖单个 workload 文件路径。

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
