from __future__ import annotations

# 中文说明：
# P1 reclaim-aware admission controller 的纯策略模块，负责估算未来 decode reserve、选择 prefill chunk，并给出 admit/shrink/defer/reject 决策。
# 该模块只读取 Sequence、KVSnapshot 和 SchedulerSnapshot，不直接修改队列或 KV cache；实际调度动作由 scheduler.py 根据决策执行。

from dataclasses import dataclass
from enum import Enum

from nanovllm.engine.sequence import Sequence


class AdmissionStateError(RuntimeError):
    pass


class AdmissionError(RuntimeError):
    pass


class AdmitAction(Enum):
    ADMIT = "admit"
    ADMIT_AFTER_RECLAIM = "admit_after_reclaim"
    SHRINK = "shrink"
    DEFER = "defer"
    REJECT_TEMP = "reject_temp"


@dataclass(slots=True)
class SchedulerConfig:
    block_size: int = 256
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    prefill_chunk_min_tokens: int = 256
    prefill_chunk_max_tokens: int = 2048
    long_prefill_token_threshold: int = 2048
    starvation_threshold: int = 4
    decode_reserve_blocks_per_seq: int = 1


@dataclass(slots=True)
class SchedulerSnapshot:
    waiting: int
    running: int
    step: int = 0


@dataclass(slots=True)
class KVSnapshot:
    free_full_blocks: int
    active_full_blocks: int
    total_full_blocks: int
    reclaimable_full_blocks: int = 0


@dataclass(slots=True)
class AdmitDecision:
    action: AdmitAction
    chunk_tokens: int
    reason: str

    @property
    def admitted(self) -> bool:
        return self.action in {
            AdmitAction.ADMIT,
            AdmitAction.ADMIT_AFTER_RECLAIM,
            AdmitAction.SHRINK,
        }


def estimate_future_decode_reserve(req: Sequence, cfg: SchedulerConfig) -> int:
    if req.max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    if cfg.block_size <= 0:
        raise ValueError("block_size must be positive")
    decode_blocks = (req.max_tokens + cfg.block_size - 1) // cfg.block_size
    return max(cfg.decode_reserve_blocks_per_seq, decode_blocks)


def _required_prefill_blocks(req: Sequence, cfg: SchedulerConfig) -> int:
    remaining = max(req.num_tokens - req.num_cached_tokens, 0)
    return (remaining + cfg.block_size - 1) // cfg.block_size


def choose_prefill_chunk(
    req: Sequence,
    sched_snapshot: SchedulerSnapshot,
    kv_snapshot: KVSnapshot,
    cfg: SchedulerConfig,
) -> int:
    remaining = req.num_tokens - req.num_cached_tokens
    if remaining <= 0:
        raise AdmissionError("request has no prefill tokens remaining")
    max_chunk = min(cfg.prefill_chunk_max_tokens, cfg.max_num_batched_tokens, remaining)
    if remaining > cfg.long_prefill_token_threshold and sched_snapshot.running:
        max_chunk = min(max_chunk, cfg.long_prefill_token_threshold)
    if max_chunk < cfg.prefill_chunk_min_tokens and remaining > cfg.prefill_chunk_min_tokens:
        raise AdmissionError("prefill chunk is below minimum")
    return max(1, max_chunk)


def decide_admission(
    req: Sequence,
    sched_snapshot: SchedulerSnapshot,
    kv_snapshot: KVSnapshot,
    cfg: SchedulerConfig,
) -> AdmitDecision:
    if req.num_cached_tokens > req.num_tokens:
        raise AdmissionStateError("num_cached_tokens exceeds prompt length")
    if kv_snapshot.free_full_blocks < 0 or kv_snapshot.total_full_blocks < 0:
        raise AdmissionStateError("invalid KV snapshot")

    required_blocks = _required_prefill_blocks(req, cfg) + estimate_future_decode_reserve(req, cfg)
    if required_blocks <= kv_snapshot.free_full_blocks:
        return AdmitDecision(AdmitAction.ADMIT, choose_prefill_chunk(req, sched_snapshot, kv_snapshot, cfg), "fits")
    if required_blocks <= kv_snapshot.free_full_blocks + kv_snapshot.reclaimable_full_blocks:
        return AdmitDecision(
            AdmitAction.ADMIT_AFTER_RECLAIM,
            choose_prefill_chunk(req, sched_snapshot, kv_snapshot, cfg),
            "fits after reclaim",
        )

    prefill_only_blocks = _required_prefill_blocks(req, cfg)
    if prefill_only_blocks <= kv_snapshot.free_full_blocks:
        return AdmitDecision(
            AdmitAction.SHRINK,
            choose_prefill_chunk(req, sched_snapshot, kv_snapshot, cfg),
            "prefill fits but future decode reserve does not",
        )
    if getattr(req, "scheduler_skip_count", 0) >= cfg.starvation_threshold and kv_snapshot.free_full_blocks > 0:
        chunk = min(cfg.block_size * kv_snapshot.free_full_blocks, cfg.max_num_batched_tokens)
        return AdmitDecision(AdmitAction.SHRINK, max(1, chunk), "starvation guard")
    if kv_snapshot.free_full_blocks == 0 and sched_snapshot.running:
        return AdmitDecision(AdmitAction.DEFER, 0, "waiting for decode progress")
    return AdmitDecision(AdmitAction.REJECT_TEMP, 0, "insufficient KV blocks")
