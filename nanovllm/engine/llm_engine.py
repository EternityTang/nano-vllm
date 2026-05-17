import atexit
from dataclasses import fields
from types import SimpleNamespace
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.arkv_kv_manager import ARKVKVManager
from nanovllm.engine.kv_meta import (
    KVBlockState,
    SequenceKVRef,
    SequenceKVState,
    PhysicalBlockTable,
    SequenceKVRefTable,
    register_full_block,
)
from nanovllm.engine.kv_policy import (
    PolicyConfig,
    ReclaimPolicyName,
    build_policy_snapshot,
    plan_reclaim_dry_run,
    select_blocks_to_evict,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.metrics import MetricsRecorder
from nanovllm.engine import profiler
from nanovllm.engine.tasks import PrefillTask
from nanovllm.engine.visible_tables import VisibleBlockTable, VisibleTableConfig, build_visible_block_table
from nanovllm.layers.mixed_kv_fallback import MixedKVWorkspacePlan, WorkspacePlanningError, split_prefill_for_workspace


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.metrics_recorder = MetricsRecorder() if config.enable_metrics_hooks else None
        self.runtime_profiler = profiler.RuntimeProfiler() if config.enable_metrics_hooks else None
        profiler.set_profiler(self.runtime_profiler)
        self.step_count = 0
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self._init_arkv_runtime()
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        if self.metrics_recorder is not None:
            self.metrics_recorder.add_request(str(seq.seq_id), len(prompt), sampling_params.max_tokens, perf_counter())
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        step_started = perf_counter()
        with profiler.timed("scheduler_ms"):
            seqs, is_prefill = self.scheduler.schedule()
        scheduled_ts = perf_counter()
        if self.metrics_recorder is not None:
            for seq in seqs:
                self.metrics_recorder.record_request_event(str(seq.seq_id), "scheduled", scheduled_ts)
        before_completion_tokens = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        if self.arkv_runtime_enabled:
            if is_prefill and self.config.enable_prefill_mixed_kv_fallback:
                self._prepare_prefill_visible_entries(seqs)
            elif not is_prefill:
                self._prepare_decode_visible_entries(seqs)
        try:
            token_ids = self.model_runner.call("run", seqs, is_prefill)
        except WorkspacePlanningError:
            if not (
                is_prefill
                and self.config.enable_prefill_mixed_kv_fallback
                and self._split_prefill_chunks_after_workspace_error(seqs)
            ):
                raise
            if self.arkv_runtime_enabled:
                self._prepare_prefill_visible_entries(seqs)
            token_ids = self.model_runner.call("run", seqs, is_prefill)
        for seq in seqs:
            if hasattr(seq, "visible_entries"):
                delattr(seq, "visible_entries")
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        if self.arkv_runtime_enabled:
            if is_prefill:
                self._sync_arkv_metadata(
                    (seq for seq in seqs if not seq.is_finished),
                    context_len_by_seq={seq.seq_id: seq.num_cached_tokens for seq in seqs},
                )
            else:
                self._sync_arkv_metadata(seq for seq in seqs if not seq.is_finished)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        if self.metrics_recorder is not None:
            event_ts = perf_counter()
            for seq in seqs:
                if before_completion_tokens[seq.seq_id] == 0 and seq.num_completion_tokens > 0:
                    self.metrics_recorder.record_request_event(str(seq.seq_id), "first_token", event_ts)
            for seq_id, _ in outputs:
                self.metrics_recorder.record_request_event(str(seq_id), "finish", event_ts)
            raw_peak_vram_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            self._publish_arkv_metrics()
            self.metrics_recorder.kv_pool.append(
                self.scheduler.block_manager.collect_metrics(self.step_count, raw_peak_vram_bytes)
            )
            self.step_count += 1
        profiler.record_ms("avg_step_ms", (perf_counter() - step_started) * 1000.0)
        return outputs, num_tokens

    def _init_arkv_runtime(self) -> None:
        cfg = self.config
        self.arkv_runtime_enabled = bool(
            cfg.enable_arkv_metadata
            and cfg.enable_kv_q8_runtime
            and cfg.enable_mixed_kv_fallback
            and self.model_runner.quant_cache is not None
        )
        self.physical_table = PhysicalBlockTable()
        self.ref_table = SequenceKVRefTable()
        self.visible_table = VisibleBlockTable()
        self._seq_logical_to_storage: dict[tuple[int, int], int] = {}
        self._full_block_to_storage: dict[int, int] = {}
        self.arkv_metrics = {
            "reclaim_trigger_count": 0,
            "quant_commits_success": 0,
            "quant_commits_rollback": 0,
            "evict_commits_success": 0,
            "evict_commits_rollback": 0,
            "full_blocks_released_after_quant": 0,
            "free_full_blocks_before_reclaim": 0,
            "free_full_blocks_after_reclaim": 0,
            "free_full_blocks_reclaim_delta": 0,
        }
        self.arkv_manager = None
        if not self.arkv_runtime_enabled:
            return
        self.arkv_manager = ARKVKVManager(
            self.model_runner.kv_cache,
            self.model_runner.quant_cache,
            self.physical_table,
            self.ref_table,
            self.visible_table,
            mixed_kv_read_available=True,
            enable_kv_evict=cfg.enable_kv_evict,
            allow_direct_full_evict=cfg.enable_direct_full_evict,
            quality_gate_passed=cfg.enable_quality_gate,
            release_full_callback=self._release_full_block_after_quant,
        )

    def _release_full_block_after_quant(self, full_block_id: int) -> None:
        self.scheduler.block_manager.release_full_block_after_quant(full_block_id)
        self._full_block_to_storage.pop(full_block_id, None)
        self.arkv_metrics["full_blocks_released_after_quant"] += 1

    def _sync_arkv_metadata(
        self,
        seqs,
        context_len_by_seq: dict[int, int] | None = None,
        inflight_blocks_by_seq: dict[int, set[int]] | None = None,
    ) -> None:
        if not self.arkv_runtime_enabled:
            return
        context_len_by_seq = context_len_by_seq or {}
        inflight_blocks_by_seq = inflight_blocks_by_seq or {}
        for seq in seqs:
            if not seq.block_table:
                continue
            logical_context_len = int(context_len_by_seq.get(seq.seq_id, len(seq)))
            if logical_context_len <= 0:
                continue
            last_logical_block = (logical_context_len - 1) // self.config.kvcache_block_size
            inflight_blocks = inflight_blocks_by_seq.get(seq.seq_id)
            for logical_block_id, full_block_id in enumerate(seq.block_table):
                logical_start = logical_block_id * self.config.kvcache_block_size
                if logical_start >= logical_context_len:
                    continue
                logical_end = min((logical_block_id + 1) * self.config.kvcache_block_size, logical_context_len)
                if logical_end <= logical_start:
                    continue
                key = (seq.seq_id, logical_block_id)
                storage_id = self._seq_logical_to_storage.get(key)
                is_recent = logical_block_id == last_logical_block
                is_inflight = logical_block_id in inflight_blocks if inflight_blocks is not None else is_recent
                if storage_id is None:
                    if full_block_id == -1:
                        continue
                    storage_id = register_full_block(
                        self.physical_table,
                        self.ref_table,
                        seq.seq_id,
                        logical_block_id,
                        full_block_id,
                        logical_start,
                        logical_end,
                        prefix_hash=None,
                        is_shared_prefix=False,
                    )
                    self._seq_logical_to_storage[key] = storage_id
                    self._full_block_to_storage[full_block_id] = storage_id
                else:
                    meta = self.physical_table.get(storage_id)
                    meta.logical_end = logical_end
                    if meta.state == KVBlockState.FULL and full_block_id != -1:
                        meta.full_block_id = full_block_id
                        self._full_block_to_storage[full_block_id] = storage_id
                    self.ref_table.replace_ref(
                        SequenceKVRef(
                            seq_id=seq.seq_id,
                            logical_block_id=logical_block_id,
                            storage_id=storage_id,
                            logical_start=logical_start,
                            logical_end=logical_end,
                            is_sink=logical_block_id == 0,
                            is_recent=is_recent,
                            is_inflight_write=is_inflight,
                        ),
                        self.physical_table,
                    )
                    continue
                self.ref_table.replace_ref(
                    SequenceKVRef(
                        seq_id=seq.seq_id,
                        logical_block_id=logical_block_id,
                        storage_id=storage_id,
                        logical_start=logical_start,
                        logical_end=logical_end,
                        is_sink=logical_block_id == 0,
                        is_recent=is_recent,
                        is_inflight_write=is_inflight,
                    ),
                    self.physical_table,
                )
            self._refresh_visible_entries(seq)

    def _refresh_visible_entries(self, seq: Sequence) -> None:
        with profiler.timed("visible_table_build_ms"):
            entries = build_visible_block_table(
                seq.seq_id,
                self.ref_table.refs_for_seq(seq.seq_id),
                self.physical_table,
                VisibleTableConfig(include_quant=True),
            )
        self.visible_table.add_entries(seq.seq_id, entries)

    def _prepare_decode_visible_entries(self, seqs: list[Sequence]) -> None:
        self._sync_arkv_metadata(seqs)
        self._maybe_quant_reclaim(allow_evict=True)
        for seq in seqs:
            self._refresh_visible_entries(seq)
            seq.visible_entries = self.visible_table.entries_for_seq(seq.seq_id)

    def _prepare_prefill_visible_entries(self, seqs: list[Sequence]) -> None:
        context_len_by_seq: dict[int, int] = {}
        inflight_blocks_by_seq: dict[int, set[int]] = {}
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            context_len_by_seq[seq.seq_id] = end
            if end <= start:
                inflight_blocks_by_seq[seq.seq_id] = set()
                continue
            start_block = start // self.config.kvcache_block_size
            end_block = (end + self.config.kvcache_block_size - 1) // self.config.kvcache_block_size
            inflight_blocks_by_seq[seq.seq_id] = set(range(start_block, end_block))
        self._sync_arkv_metadata(
            seqs,
            context_len_by_seq=context_len_by_seq,
            inflight_blocks_by_seq=inflight_blocks_by_seq,
        )
        self._maybe_quant_reclaim(allow_evict=False)
        for seq in seqs:
            self._refresh_visible_entries(seq)
            seq.visible_entries = self.visible_table.entries_for_seq(seq.seq_id)

    def _split_prefill_chunks_after_workspace_error(self, seqs: list[Sequence]) -> bool:
        scratch = getattr(self.model_runner, "q8_scratch", None)
        if scratch is None:
            return False
        scratch_budget = scratch.numel() * scratch.element_size()
        if scratch_budget <= 0:
            return False
        cfg = SimpleNamespace(
            scratch_kv_budget_bytes=scratch_budget,
            prefill_chunk_min_tokens=self.config.prefill_chunk_min_tokens,
        )
        split_any = False
        for seq in seqs:
            old_tokens = int(seq.num_scheduled_tokens)
            if old_tokens <= self.config.prefill_chunk_min_tokens:
                continue
            task = PrefillTask(
                seq_id=seq.seq_id,
                request_id=getattr(seq, "request_id", str(seq.seq_id)),
                start_pos=seq.num_cached_tokens,
                chunk_tokens=old_tokens,
                is_long_prefill=old_tokens > self.config.long_prefill_token_threshold,
                skip_count=getattr(seq, "scheduler_skip_count", 0),
            )
            overflow_plan = MixedKVWorkspacePlan(
                batch_size=1,
                max_quant_blocks_per_seq=max(int(getattr(self.config, "kv_q8_scratch_blocks", 1)) + 1, 1),
                total_quant_blocks=max(int(getattr(self.config, "kv_q8_scratch_blocks", 1)) + 1, 1),
                scratch_shape=tuple(scratch.shape),
                scratch_numel=scratch.numel(),
                scratch_bytes=scratch_budget * 2,
            )
            chunks = split_prefill_for_workspace(task, overflow_plan, cfg)
            new_tokens = chunks[0].chunk_tokens
            if new_tokens < old_tokens:
                seq.num_scheduled_tokens = new_tokens
                split_any = True
        return split_any

    def _maybe_quant_reclaim(self, allow_evict: bool = True) -> None:
        if self.arkv_manager is None:
            return
        if allow_evict:
            self._maybe_evict_reclaim()
        free_full_before = len(self.scheduler.block_manager.free_block_ids)
        with profiler.timed("reclaim_planning_ms"):
            snapshot = build_policy_snapshot(
                self.physical_table,
                self.ref_table,
                total_full_blocks=len(self.scheduler.block_manager.blocks),
                free_full_blocks=free_full_before,
            )
            plan = plan_reclaim_dry_run(
                snapshot,
                required_full_equiv=free_full_before + 1,
                policy_name=ReclaimPolicyName.ARKV_Q8_DRY_RUN,
                cfg=PolicyConfig(),
            )
        if not plan.selected_storage_ids:
            return
        attempted = False
        for storage_id in plan.selected_storage_ids:
            if self._would_exceed_seq_scratch_limit(storage_id):
                continue
            if not attempted:
                self.arkv_metrics["reclaim_trigger_count"] += 1
                attempted = True
            try:
                with profiler.timed("quantize_from_full_ms"):
                    result = self.arkv_manager.quantize_from_full(
                        storage_id,
                        reason="p4a_runtime_pressure",
                        step=self.step_count,
                        allow_release_full=True,
                    )
            except Exception:
                self.arkv_metrics["quant_commits_rollback"] += 1
                raise
            else:
                if result.released_full_block_id is not None:
                    self._mark_quantized_logical_refs(storage_id, result.released_full_block_id)
                    free_after = len(self.scheduler.block_manager.free_block_ids)
                    self.arkv_metrics["free_full_blocks_before_reclaim"] = max(
                        self.arkv_metrics.get("free_full_blocks_before_reclaim", 0),
                        free_full_before,
                    )
                    self.arkv_metrics["free_full_blocks_after_reclaim"] = max(
                        self.arkv_metrics.get("free_full_blocks_after_reclaim", 0),
                        free_after,
                    )
                    self.arkv_metrics["free_full_blocks_reclaim_delta"] = max(
                        self.arkv_metrics.get("free_full_blocks_reclaim_delta", 0),
                        free_after - free_full_before,
                    )
                self.arkv_metrics["quant_commits_success"] += 1
        if allow_evict:
            self._maybe_evict_reclaim()

    def _maybe_evict_reclaim(self) -> None:
        if not (
            getattr(self.config, "enable_kv_evict", False)
            and getattr(self.config, "enable_quality_gate", False)
        ):
            return
        free_full_before = len(self.scheduler.block_manager.free_block_ids)
        cfg = PolicyConfig(
            allow_evict=True,
            allow_direct_full_evict=getattr(self.config, "enable_direct_full_evict", False),
            quality_gate_passed=True,
        )
        with profiler.timed("reclaim_planning_ms"):
            snapshot = build_policy_snapshot(
                self.physical_table,
                self.ref_table,
                total_full_blocks=len(self.scheduler.block_manager.blocks),
                free_full_blocks=free_full_before,
            )
            plan = plan_reclaim_dry_run(
                snapshot,
                required_full_equiv=free_full_before + 1,
                policy_name=ReclaimPolicyName.ARKV_Q8_EVICT,
                cfg=cfg,
            )
        if not plan.selected_storage_ids:
            return
        seq_states = {
            seq.seq_id: SequenceKVState.ACTIVE for seq in list(self.scheduler.running) + list(self.scheduler.waiting)
        }
        selected, _ = select_blocks_to_evict(list(plan.candidates), 1, seq_states, cfg)
        if not selected:
            return
        self.arkv_metrics["reclaim_trigger_count"] += 1
        for candidate in selected:
            try:
                self.arkv_manager.apply_evict_transition(
                    candidate.storage_id,
                    reason="p5_quality_gated_evict",
                    step=self.step_count,
                )
            except Exception:
                self.arkv_metrics["evict_commits_rollback"] += 1
                raise
            else:
                self.arkv_metrics["evict_commits_success"] += 1

    def _would_exceed_seq_scratch_limit(self, storage_id: int) -> bool:
        scratch_blocks = max(int(getattr(self.config, "kv_q8_scratch_blocks", 1)), 1)
        meta = self.physical_table.get(storage_id)
        for owner in meta.copy_owner_refs():
            quant_entries = 0
            for ref in self.ref_table.refs_for_seq(owner.seq_id):
                if self.physical_table.get(ref.storage_id).state == KVBlockState.QUANT:
                    quant_entries += 1
            if quant_entries >= scratch_blocks:
                return True
        return False

    def _mark_quantized_logical_refs(self, storage_id: int, released_full_block_id: int) -> None:
        meta = self.physical_table.get(storage_id)
        seqs_by_id = {seq.seq_id: seq for seq in list(self.scheduler.running) + list(self.scheduler.waiting)}
        for owner in meta.copy_owner_refs():
            seq = seqs_by_id.get(owner.seq_id)
            if seq is None or owner.logical_block_id >= len(seq.block_table):
                continue
            if seq.block_table[owner.logical_block_id] == released_full_block_id:
                seq.block_table[owner.logical_block_id] = -1

    def _publish_arkv_metrics(self) -> None:
        if not self.arkv_runtime_enabled or self.model_runner.quant_cache is None:
            return
        visible_quant_entries = sum(
            1
            for seq_id in self.visible_table.seq_ids()
            for entry in self.visible_table.entries_for_seq(seq_id)
            if entry.state == KVBlockState.QUANT
        )
        self.scheduler.block_manager.arkv_metrics = {
            "active_quant_blocks": len(self.model_runner.quant_cache.used_quant_block_ids),
            "evicted_blocks": sum(1 for meta in self.physical_table.values() if meta.state == KVBlockState.EVICT),
            "visible_quant_entries": visible_quant_entries,
            "mixed_kv_quant_reads": self.model_runner.runtime_metrics.get("mixed_kv_quant_reads", 0),
            **self.arkv_metrics,
        }

    def profile_dict(self) -> dict[str, float | int]:
        if self.runtime_profiler is None:
            return profiler.RuntimeProfiler().to_dict()
        return self.runtime_profiler.to_dict()

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
