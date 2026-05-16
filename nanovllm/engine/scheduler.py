from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.admission import (
    AdmitAction,
    KVSnapshot,
    SchedulerConfig,
    SchedulerSnapshot,
    decide_admission,
)
from nanovllm.engine.scheduler_metrics import SchedulerMetricsRecorder, SchedulerStepMetrics


class Scheduler:

    def __init__(self, config: Config):
        self.config = config
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.enable_memory_aware_scheduler = config.enable_memory_aware_scheduler
        self.enable_admission_controller = config.enable_admission_controller
        self.scheduler_cfg = SchedulerConfig(
            block_size=config.kvcache_block_size,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_seqs=config.max_num_seqs,
            prefill_chunk_min_tokens=config.prefill_chunk_min_tokens,
            prefill_chunk_max_tokens=config.prefill_chunk_max_tokens,
            long_prefill_token_threshold=config.long_prefill_token_threshold,
            starvation_threshold=config.scheduler_starvation_threshold,
        )
        self.metrics = SchedulerMetricsRecorder()
        self.step_count = 0
        bytes_per_block = 0
        if config.hf_config is not None:
            num_kv_heads = config.hf_config.num_key_value_heads // config.tensor_parallel_size
            head_dim = getattr(config.hf_config, "head_dim", config.hf_config.hidden_size // config.hf_config.num_attention_heads)
            bytes_per_block = (
                2
                * config.hf_config.num_hidden_layers
                * config.kvcache_block_size
                * num_kv_heads
                * head_dim
                * config.hf_config.dtype.itemsize
            )
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size, bytes_per_block)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def queue_depths(self) -> dict[str, int]:
        return {"waiting": len(self.waiting), "running": len(self.running)}

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.enable_memory_aware_scheduler:
            return self._schedule_memory_aware()
        return self._schedule_legacy()

    def _schedule_legacy(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def _kv_snapshot(self) -> KVSnapshot:
        return KVSnapshot(
            free_full_blocks=len(self.block_manager.free_block_ids),
            active_full_blocks=len(self.block_manager.used_block_ids),
            total_full_blocks=len(self.block_manager.blocks),
        )

    def _sched_snapshot(self) -> SchedulerSnapshot:
        return SchedulerSnapshot(waiting=len(self.waiting), running=len(self.running), step=self.step_count)

    def _record_step(
        self,
        batch_kind: str,
        scheduled_sequences: int,
        scheduled_tokens: int,
        decisions: list[AdmitAction] | None = None,
        starvation_forced: int = 0,
    ) -> None:
        decisions = decisions or []
        self.metrics.append(
            SchedulerStepMetrics(
                step=self.step_count,
                batch_kind=batch_kind,
                scheduled_sequences=scheduled_sequences,
                scheduled_tokens=scheduled_tokens,
                admitted=sum(action == AdmitAction.ADMIT for action in decisions),
                admit_after_reclaim=sum(action == AdmitAction.ADMIT_AFTER_RECLAIM for action in decisions),
                shrunk=sum(action == AdmitAction.SHRINK for action in decisions),
                deferred=sum(action == AdmitAction.DEFER for action in decisions),
                rejected_temp=sum(action == AdmitAction.REJECT_TEMP for action in decisions),
                starvation_forced=starvation_forced,
            )
        )
        self.step_count += 1

    def _schedule_memory_aware(self) -> tuple[list[Sequence], bool]:
        decode_batch = self._schedule_decode_first()
        if decode_batch:
            for seq in self.waiting:
                seq.scheduler_skip_count += 1
            self._record_step("decode", len(decode_batch), len(decode_batch))
            return decode_batch, False

        prefill_batch, decisions, forced = self._schedule_prefill_memory_aware()
        if prefill_batch:
            self._record_step(
                "prefill",
                len(prefill_batch),
                sum(seq.num_scheduled_tokens for seq in prefill_batch),
                decisions,
                forced,
            )
            return prefill_batch, True
        assert prefill_batch

    def _schedule_decode_first(self) -> list[Sequence]:
        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def _select_prefill_index(self) -> int:
        for i, seq in enumerate(self.waiting):
            if seq.scheduler_skip_count >= self.scheduler_cfg.starvation_threshold:
                return i
        for i, seq in enumerate(self.waiting):
            remaining = seq.num_tokens - seq.num_cached_tokens
            if remaining <= self.scheduler_cfg.long_prefill_token_threshold:
                return i
        return 0

    def _schedule_prefill_memory_aware(self) -> tuple[list[Sequence], list[AdmitAction], int]:
        scheduled_seqs = []
        decisions: list[AdmitAction] = []
        starvation_forced = 0
        num_batched_tokens = 0
        attempts = len(self.waiting)

        while self.waiting and attempts > 0 and len(scheduled_seqs) < self.max_num_seqs:
            attempts -= 1
            index = self._select_prefill_index()
            self.waiting.rotate(-index)
            seq = self.waiting[0]
            remaining_budget = self.max_num_batched_tokens - num_batched_tokens
            if remaining_budget <= 0:
                break

            decision = None
            if self.enable_admission_controller:
                decision = decide_admission(seq, self._sched_snapshot(), self._kv_snapshot(), self.scheduler_cfg)
                decisions.append(decision.action)
                if not decision.admitted:
                    seq.scheduler_skip_count += 1
                    self.waiting.rotate(-1)
                    continue

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    seq.scheduler_skip_count += 1
                    decisions.append(AdmitAction.DEFER)
                    self.waiting.rotate(-1)
                    continue
                self.block_manager.allocate(seq, num_cached_blocks)
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            chunk_tokens = decision.chunk_tokens if decision is not None else num_tokens
            chunk_tokens = max(1, min(num_tokens, chunk_tokens, remaining_budget))
            if remaining_budget < chunk_tokens and scheduled_seqs:
                break

            if seq.scheduler_skip_count >= self.scheduler_cfg.starvation_threshold:
                starvation_forced += 1
            seq.num_scheduled_tokens = chunk_tokens
            seq.scheduler_skip_count = 0
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        return scheduled_seqs, decisions, starvation_forced

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
