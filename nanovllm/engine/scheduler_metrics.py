from __future__ import annotations

# 中文说明：
# P1 scheduler/admission 计数记录器，按 step 保存 batch 类型、调度序列数、token 数以及 admission 各类决策数量。
# benchmark_serving.py 读取这些 step metrics 生成 B1 及后续阶段的 admission summary，用于解释调度策略收益或退化。

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
