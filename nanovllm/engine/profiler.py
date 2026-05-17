"""Lightweight runtime profiler for optimizer benchmark paths."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


PROFILE_FIELDS = (
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
    "avg_step_ms",
)

COUNTER_FIELDS = (
    "fused_kernel_calls",
    "fused_kernel_fallbacks",
    "fallback_count",
    "parity_check_calls",
)


class RuntimeProfiler:
    def __init__(self):
        self.ms = {field: 0.0 for field in PROFILE_FIELDS if field != "avg_step_ms"}
        self.counts = {field: 0 for field in COUNTER_FIELDS}
        self.step_ms_total = 0.0
        self.step_count = 0

    def add_ms(self, field: str, value_ms: float) -> None:
        if field == "avg_step_ms":
            self.step_ms_total += float(value_ms)
            self.step_count += 1
            return
        if field in self.ms:
            self.ms[field] += float(value_ms)

    def inc(self, field: str, amount: int = 1) -> None:
        if field in self.counts:
            self.counts[field] += int(amount)

    def to_dict(self) -> dict[str, float | int]:
        data: dict[str, float | int] = {field: self.ms.get(field, 0.0) for field in PROFILE_FIELDS if field != "avg_step_ms"}
        data["avg_step_ms"] = self.step_ms_total / self.step_count if self.step_count else 0.0
        data.update(self.counts)
        return data


_CURRENT: RuntimeProfiler | None = None


def set_profiler(profiler: RuntimeProfiler | None) -> None:
    global _CURRENT
    _CURRENT = profiler


def get_profiler() -> RuntimeProfiler | None:
    return _CURRENT


def record_ms(field: str, value_ms: float) -> None:
    profiler = _CURRENT
    if profiler is not None:
        profiler.add_ms(field, value_ms)


def increment(field: str, amount: int = 1) -> None:
    profiler = _CURRENT
    if profiler is not None:
        profiler.inc(field, amount)


@contextmanager
def timed(field: str):
    if _CURRENT is None:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        record_ms(field, (perf_counter() - started) * 1000.0)
