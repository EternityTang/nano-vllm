from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


RequestEvent = Literal["arrival", "scheduled", "first_token", "finish", "oom"]


class MetricsStateError(RuntimeError):
    pass


class MetricsUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class RequestMetrics:
    request_id: str
    arrival_ts: float
    scheduled_ts: float | None = None
    first_token_ts: float | None = None
    finish_ts: float | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    oom: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class KVPoolMetrics:
    step: int
    free_full_blocks: int
    active_full_blocks: int
    active_quant_blocks: int
    evicted_blocks: int
    free_full_block_ratio: float
    effective_kv_memory_bytes: int
    raw_peak_vram_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


class MetricsRecorder:
    def __init__(self):
        self.requests: dict[str, RequestMetrics] = {}
        self.kv_pool: list[KVPoolMetrics] = []

    def add_request(self, request_id: str, prompt_tokens: int, output_tokens: int = 0, arrival_ts: float = 0.0):
        if request_id in self.requests:
            raise MetricsStateError(f"duplicate request_id: {request_id}")
        self.requests[request_id] = RequestMetrics(
            request_id=request_id,
            arrival_ts=arrival_ts,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

    def record_request_event(self, request_id: str, event: RequestEvent, timestamp: float) -> None:
        metric = self.requests.get(request_id)
        if metric is None:
            if event != "arrival":
                raise MetricsStateError(f"request {request_id!r} has no arrival event")
            self.requests[request_id] = RequestMetrics(request_id=request_id, arrival_ts=timestamp)
            return
        if event == "arrival":
            if metric.arrival_ts:
                raise MetricsStateError(f"request {request_id!r} arrival recorded twice")
            metric.arrival_ts = timestamp
        elif event == "scheduled":
            if timestamp < metric.arrival_ts:
                raise MetricsStateError("scheduled timestamp precedes arrival")
            if metric.scheduled_ts is None:
                metric.scheduled_ts = timestamp
        elif event == "first_token":
            if metric.scheduled_ts is None:
                raise MetricsStateError("first_token recorded before scheduled")
            if metric.first_token_ts is None:
                metric.first_token_ts = timestamp
        elif event == "finish":
            if metric.first_token_ts is None:
                raise MetricsStateError("finish recorded before first_token")
            if metric.finish_ts is None:
                metric.finish_ts = timestamp
        elif event == "oom":
            metric.oom = True
        else:
            raise MetricsStateError(f"unknown event: {event}")

    def collect_kv_pool_metrics(self, step: int, block_manager) -> KVPoolMetrics:
        if block_manager is None or not hasattr(block_manager, "collect_metrics"):
            raise MetricsUnavailableError("block manager metrics are unavailable")
        metric = block_manager.collect_metrics(step)
        self.kv_pool.append(metric)
        return metric

    def request_dicts(self) -> list[dict]:
        return [metric.to_dict() for metric in self.requests.values()]

    def kv_pool_dicts(self) -> list[dict]:
        return [metric.to_dict() for metric in self.kv_pool]


_DEFAULT_RECORDER = MetricsRecorder()


def record_request_event(request_id: str, event: RequestEvent, timestamp: float) -> None:
    _DEFAULT_RECORDER.record_request_event(request_id, event, timestamp)


def collect_kv_pool_metrics(step: int, block_manager=None) -> KVPoolMetrics:
    return _DEFAULT_RECORDER.collect_kv_pool_metrics(step, block_manager)
