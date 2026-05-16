from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class TaskKind(Enum):
    DECODE = "decode"
    PREFILL = "prefill"


class SchedulerInvariantError(RuntimeError):
    pass


@dataclass(slots=True)
class DecodeTask:
    seq_id: int
    request_id: str
    num_tokens: int = 1


@dataclass(slots=True)
class PrefillTask:
    seq_id: int
    request_id: str
    start_pos: int
    chunk_tokens: int
    is_long_prefill: bool
    skip_count: int


@dataclass(slots=True)
class BatchPlan:
    batch_id: int
    kind: Literal["decode", "prefill"]
    decode_tasks: list[DecodeTask]
    prefill_tasks: list[PrefillTask]
    token_budget: int
    slot_mapping: Any | None = None
    visible_block_tables: dict[int, list[Any]] | None = None
    workspace_plan: Any | None = None

    def __post_init__(self):
        if self.kind == TaskKind.DECODE.value and self.prefill_tasks:
            raise SchedulerInvariantError("decode BatchPlan cannot contain prefill tasks")
        if self.kind == TaskKind.PREFILL.value and self.decode_tasks:
            raise SchedulerInvariantError("prefill BatchPlan cannot contain decode tasks")
        if self.decode_tasks and self.prefill_tasks:
            raise SchedulerInvariantError("P1 BatchPlan must be homogeneous")


def build_batch_plan(
    waiting,
    running,
    sched_snapshot,
    kv_snapshot,
    cfg,
) -> BatchPlan:
    batch_id = getattr(sched_snapshot, "step", 0)
    if running:
        max_tasks = min(len(running), getattr(cfg, "max_num_seqs", len(running)))
        tasks = [
            DecodeTask(seq_id=seq.seq_id, request_id=getattr(seq, "request_id", str(seq.seq_id)))
            for seq in list(running)[:max_tasks]
        ]
        return BatchPlan(
            batch_id=batch_id,
            kind=TaskKind.DECODE.value,
            decode_tasks=tasks,
            prefill_tasks=[],
            token_budget=len(tasks),
        )

    budget = getattr(cfg, "max_num_batched_tokens", 0)
    tasks: list[PrefillTask] = []
    used_tokens = 0
    for seq in waiting:
        if len(tasks) >= getattr(cfg, "max_num_seqs", len(waiting)):
            break
        remaining = seq.num_tokens - seq.num_cached_tokens
        if remaining <= 0:
            continue
        chunk = min(remaining, budget - used_tokens) if budget else remaining
        if chunk <= 0:
            break
        tasks.append(
            PrefillTask(
                seq_id=seq.seq_id,
                request_id=getattr(seq, "request_id", str(seq.seq_id)),
                start_pos=seq.num_cached_tokens,
                chunk_tokens=chunk,
                is_long_prefill=remaining > getattr(cfg, "long_prefill_token_threshold", 2048),
                skip_count=getattr(seq, "scheduler_skip_count", 0),
            )
        )
        used_tokens += chunk
    return BatchPlan(
        batch_id=batch_id,
        kind=TaskKind.PREFILL.value,
        decode_tasks=[],
        prefill_tasks=tasks,
        token_budget=budget,
    )
