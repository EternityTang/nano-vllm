from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class SchedulerStepMetrics:
    step: int
    batch_kind: str
    scheduled_sequences: int
    scheduled_tokens: int
    admitted: int = 0
    admit_after_reclaim: int = 0
    shrunk: int = 0
    deferred: int = 0
    rejected_temp: int = 0
    starvation_forced: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class SchedulerMetricsRecorder:
    def __init__(self):
        self.steps: list[SchedulerStepMetrics] = []

    def append(self, metric: SchedulerStepMetrics) -> None:
        self.steps.append(metric)

    def to_dicts(self) -> list[dict]:
        return [metric.to_dict() for metric in self.steps]
