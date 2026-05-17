"""Mixed FULL/Q8 KV fallback used before fused kernels exist."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import os
from typing import Any

import torch

from nanovllm.engine.kv_meta import KVBlockState
from nanovllm.engine import profiler
from nanovllm.engine.quant_cache import QuantCache, ScratchOverflowError
from nanovllm.engine.tasks import BatchPlan, PrefillTask
from nanovllm.engine.visible_tables import VisibleBlockEntry, VisibleBlockTable
from nanovllm.kernels.mixed_kv_decode_attention import (
    mixed_kv_decode_attention,
    mixed_kv_decode_attention_supported,
    visible_entries_to_tensor,
)
from nanovllm.kernels.triton_gather_dequant import (
    KernelNotSupportedError,
    KernelRuntimeError,
    gather_dequant_reference,
    gather_dequant_triton,
    triton_gather_dequant_supported,
)


class MixedKVReadError(RuntimeError):
    pass


class WorkspacePlanningError(RuntimeError):
    pass


class FullReuseSafetyError(RuntimeError):
    pass


class InvalidWriteTargetError(MixedKVReadError):
    pass


class PolicyInvariantError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FullKVCache:
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    layer_id: int = 0


@dataclass(frozen=True, slots=True)
class MaterializedKV:
    k: torch.Tensor
    v: torch.Tensor
    context_len: int


@dataclass(frozen=True, slots=True)
class AttentionMetadata:
    layer_id: int = 0
    softmax_scale: float = 1.0
    query_lengths: tuple[int, ...] | None = None
    query_start_positions: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class MixedKVWorkspacePlan:
    batch_size: int
    max_quant_blocks_per_seq: int
    total_quant_blocks: int
    scratch_shape: tuple[int, int, int, int, int]
    scratch_numel: int
    scratch_bytes: int


def plan_mixed_kv_workspace(
    batch_plan: BatchPlan,
    visible_tables: VisibleBlockTable,
    cache_cfg: Any,
) -> MixedKVWorkspacePlan:
    """Estimate scratch needed for decode-only mixed-KV fallback."""
    if batch_plan.kind != "decode":
        raise WorkspacePlanningError("mixed-KV fallback workspace is decode-only")

    block_size = _cfg_int(cache_cfg, "block_size", "kvcache_block_size")
    num_kv_heads = _cfg_int(cache_cfg, "num_kv_heads")
    head_dim = _cfg_int(cache_cfg, "head_dim")
    dtype = getattr(cache_cfg, "dtype", torch.float16)
    itemsize = _dtype_itemsize(dtype)

    max_quant_blocks = 0
    total_quant_blocks = 0
    for task in batch_plan.decode_tasks:
        entries = list(visible_tables.entries_for_seq(task.seq_id))
        if not entries:
            raise WorkspacePlanningError(f"missing visible entries for decode seq_id {task.seq_id}")
        try:
            _validate_visible_entries(entries)
        except MixedKVReadError as exc:
            raise WorkspacePlanningError(str(exc)) from exc
        quant_blocks = sum(1 for entry in entries if entry.state == KVBlockState.QUANT)
        max_quant_blocks = max(max_quant_blocks, quant_blocks)
        total_quant_blocks += quant_blocks

    scratch_shape = (max_quant_blocks, 2, block_size, num_kv_heads, head_dim)
    scratch_numel = prod(scratch_shape)
    scratch_bytes = scratch_numel * itemsize
    scratch_budget = int(getattr(cache_cfg, "scratch_kv_budget_bytes", 0) or 0)
    if scratch_budget and scratch_bytes > scratch_budget:
        raise WorkspacePlanningError(f"mixed-KV scratch needs {scratch_bytes} bytes but budget is {scratch_budget}")
    return MixedKVWorkspacePlan(
        batch_size=len(batch_plan.decode_tasks),
        max_quant_blocks_per_seq=max_quant_blocks,
        total_quant_blocks=total_quant_blocks,
        scratch_shape=scratch_shape,
        scratch_numel=scratch_numel,
        scratch_bytes=scratch_bytes,
    )


def plan_prefill_mixed_kv_workspace(
    batch_plan: BatchPlan,
    visible_tables: VisibleBlockTable,
    cache_cfg: Any,
) -> MixedKVWorkspacePlan:
    """Estimate scratch needed for prefill mixed-KV fallback."""
    if batch_plan.kind != "prefill":
        raise WorkspacePlanningError("prefill mixed-KV fallback workspace requires a prefill batch")

    block_size = _cfg_int(cache_cfg, "block_size", "kvcache_block_size")
    num_kv_heads = _cfg_int(cache_cfg, "num_kv_heads")
    head_dim = _cfg_int(cache_cfg, "head_dim")
    dtype = getattr(cache_cfg, "dtype", torch.float16)
    itemsize = _dtype_itemsize(dtype)

    max_quant_blocks = 0
    total_quant_blocks = 0
    for task in batch_plan.prefill_tasks:
        entries = list(visible_tables.entries_for_seq(task.seq_id))
        if not entries:
            raise WorkspacePlanningError(f"missing visible entries for prefill seq_id {task.seq_id}")
        try:
            _validate_visible_entries(entries)
        except MixedKVReadError as exc:
            raise WorkspacePlanningError(str(exc)) from exc
        if entries[-1].logical_end < task.start_pos + task.chunk_tokens:
            raise WorkspacePlanningError("visible entries do not cover the prefill chunk")
        if any(entry.state == KVBlockState.EVICT for entry in entries):
            raise WorkspacePlanningError("prefill mixed-KV fallback cannot read EVICT entries before P5")
        quant_blocks = sum(1 for entry in entries if entry.state == KVBlockState.QUANT)
        max_quant_blocks = max(max_quant_blocks, quant_blocks)
        total_quant_blocks += quant_blocks

    scratch_shape = (max_quant_blocks, 2, block_size, num_kv_heads, head_dim)
    scratch_numel = prod(scratch_shape)
    scratch_bytes = scratch_numel * itemsize
    scratch_budget = int(getattr(cache_cfg, "scratch_kv_budget_bytes", 0) or 0)
    if scratch_budget and scratch_bytes > scratch_budget:
        raise WorkspacePlanningError(f"prefill mixed-KV scratch needs {scratch_bytes} bytes but budget is {scratch_budget}")
    return MixedKVWorkspacePlan(
        batch_size=len(batch_plan.prefill_tasks),
        max_quant_blocks_per_seq=max_quant_blocks,
        total_quant_blocks=total_quant_blocks,
        scratch_shape=scratch_shape,
        scratch_numel=scratch_numel,
        scratch_bytes=scratch_bytes,
    )


def split_prefill_for_workspace(
    prefill_task: PrefillTask,
    workspace_plan: MixedKVWorkspacePlan,
    cfg: Any,
) -> list[PrefillTask]:
    """Split a prefill task when the planned mixed-KV scratch exceeds budget."""
    scratch_budget = int(getattr(cfg, "scratch_kv_budget_bytes", 0) or getattr(cfg, "mixed_kv_scratch_budget_bytes", 0) or 0)
    if scratch_budget <= 0 or workspace_plan.scratch_bytes <= scratch_budget:
        return [prefill_task]

    min_chunk = max(int(getattr(cfg, "prefill_chunk_min_tokens", 1) or 1), 1)
    if prefill_task.chunk_tokens <= min_chunk:
        raise WorkspacePlanningError("cannot split prefill chunk below minimum chunk size")

    ratio = max(min(scratch_budget / max(workspace_plan.scratch_bytes, 1), 0.5), 0.01)
    chunk = max(min_chunk, int(prefill_task.chunk_tokens * ratio))
    if chunk >= prefill_task.chunk_tokens:
        chunk = max(min_chunk, prefill_task.chunk_tokens // 2)
    if chunk < min_chunk:
        raise WorkspacePlanningError("cannot split prefill chunk below minimum chunk size")

    tasks = []
    remaining = prefill_task.chunk_tokens
    offset = 0
    while remaining > 0:
        current = min(chunk, remaining)
        if current < min_chunk and tasks:
            previous = tasks.pop()
            current += previous.chunk_tokens
            offset = previous.start_pos - prefill_task.start_pos
        tasks.append(
            PrefillTask(
                seq_id=prefill_task.seq_id,
                request_id=prefill_task.request_id,
                start_pos=prefill_task.start_pos + offset,
                chunk_tokens=current,
                is_long_prefill=prefill_task.is_long_prefill,
                skip_count=prefill_task.skip_count,
            )
        )
        offset += current
        remaining -= current
    return tasks


def materialize_visible_kv_for_decode(
    visible_entries: list[VisibleBlockEntry] | tuple[VisibleBlockEntry, ...],
    full_cache: FullKVCache,
    quant_cache: QuantCache,
    workspace: torch.Tensor,
    use_triton_gather_dequant: bool = False,
) -> MaterializedKV:
    """Build contiguous K/V tensors for one decode sequence from visible entries."""
    entries = list(visible_entries)
    _validate_visible_entries(entries)
    if not entries:
        raise MixedKVReadError("decode mixed-KV fallback requires at least one visible entry")

    block_size = full_cache.k_cache.shape[1]
    device = full_cache.k_cache.device
    dtype = full_cache.k_cache.dtype
    num_kv_heads = full_cache.k_cache.shape[2]
    head_dim = full_cache.k_cache.shape[3]
    context_len = sum(entry.logical_end - entry.logical_start for entry in entries if entry.state != KVBlockState.EVICT)
    if context_len <= 0:
        raise MixedKVReadError("visible entries contain no readable KV blocks")

    k_out = torch.empty(context_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_out = torch.empty_like(k_out)

    quant_entries = [entry for entry in entries if entry.state == KVBlockState.QUANT]
    dequantized = None
    quant_index_by_storage: dict[int, int] = {}
    if quant_entries:
        quant_block_ids = []
        for entry in quant_entries:
            if entry.quant_block_id is None:
                raise MixedKVReadError(f"QUANT entry {entry.storage_id} is missing quant_block_id")
            quant_index_by_storage[entry.storage_id] = len(quant_block_ids)
            quant_block_ids.append(entry.quant_block_id)
        dequantized = _dequantize_quant_entries(
            quant_cache,
            quant_entries,
            quant_block_ids,
            full_cache.layer_id,
            workspace,
            use_triton_gather_dequant=use_triton_gather_dequant,
        )

    cursor = 0
    for entry in entries:
        length = entry.logical_end - entry.logical_start
        if entry.state == KVBlockState.EVICT:
            continue
        if length < 1 or length > block_size:
            raise MixedKVReadError(f"invalid visible entry length: {length}")
        offset = entry.logical_start % block_size
        end = offset + length
        if end > block_size:
            raise MixedKVReadError("visible entry crosses a physical block boundary")

        if entry.state == KVBlockState.FULL:
            if entry.full_block_id is None:
                raise MixedKVReadError(f"FULL entry {entry.storage_id} is missing full_block_id")
            k_block = full_cache.k_cache[entry.full_block_id]
            v_block = full_cache.v_cache[entry.full_block_id]
        elif entry.state == KVBlockState.QUANT:
            if dequantized is None:
                raise MixedKVReadError("QUANT entry has no dequantized scratch")
            quant_index = quant_index_by_storage[entry.storage_id]
            k_block = dequantized[quant_index, 0]
            v_block = dequantized[quant_index, 1]
        else:
            raise MixedKVReadError(f"unsupported visible entry state: {entry.state}")

        k_out[cursor : cursor + length].copy_(k_block[offset:end])
        v_out[cursor : cursor + length].copy_(v_block[offset:end])
        cursor += length

    if cursor != context_len:
        raise MixedKVReadError("materialized context length mismatch")
    return MaterializedKV(k=k_out, v=v_out, context_len=context_len)


def run_prefill_mixed_kv_fallback(
    q: torch.Tensor,
    visible_entries: list[list[VisibleBlockEntry]] | list[tuple[VisibleBlockEntry, ...]],
    slot_mapping: torch.Tensor,
    full_k_cache: torch.Tensor,
    full_v_cache: torch.Tensor,
    quant_cache: QuantCache,
    workspace: torch.Tensor,
    attn_metadata: AttentionMetadata,
    use_triton_gather_dequant: bool = False,
) -> torch.Tensor:
    """Prefill attention over FULL/QUANT visible entries.

    The write path is intentionally not part of the visible table: callers must
    store K/V into FULL cache slots through slot_mapping before invoking this.
    """
    if q.ndim != 3:
        raise MixedKVReadError(f"prefill q must have shape [tokens, heads, dim], got {tuple(q.shape)}")
    if full_k_cache.shape != full_v_cache.shape:
        raise MixedKVReadError("full K/V cache shapes differ")
    if attn_metadata.query_lengths is None:
        raise MixedKVReadError("prefill fallback requires query_lengths metadata")
    query_lengths = tuple(int(length) for length in attn_metadata.query_lengths)
    if len(query_lengths) != len(visible_entries):
        raise MixedKVReadError("visible entry batch size does not match prefill metadata")
    if sum(query_lengths) != q.shape[0]:
        raise MixedKVReadError("prefill query length metadata does not match q tokens")

    query_starts = attn_metadata.query_start_positions
    if query_starts is not None and len(query_starts) != len(query_lengths):
        raise MixedKVReadError("query_start_positions size does not match query_lengths")

    outputs = []
    cursor = 0
    full_cache = FullKVCache(full_k_cache, full_v_cache, layer_id=attn_metadata.layer_id)
    for batch_idx, entries in enumerate(visible_entries):
        entries = list(entries)
        if any(entry.state == KVBlockState.EVICT for entry in entries):
            raise MixedKVReadError("prefill mixed-KV fallback cannot read EVICT entries before P5")
        query_len = query_lengths[batch_idx]
        q_seq = q[cursor : cursor + query_len]
        slot_seq = slot_mapping[cursor : cursor + query_len]
        _validate_prefill_write_targets(slot_seq, entries, full_k_cache.shape[1])
        materialized = materialize_visible_kv_for_decode(
            entries,
            full_cache,
            quant_cache,
            workspace,
            use_triton_gather_dequant=use_triton_gather_dequant,
        )
        query_start = int(query_starts[batch_idx]) if query_starts is not None else materialized.context_len - query_len
        if query_start < 0 or query_start + query_len > materialized.context_len:
            raise MixedKVReadError("prefill query span is outside visible context")
        outputs.append(
            _prefill_attention(
                q_seq,
                materialized.k,
                materialized.v,
                query_start=query_start,
                scale=attn_metadata.softmax_scale,
            )
        )
        cursor += query_len
    return torch.cat(outputs, dim=0)


def run_decode_mixed_kv_fallback(
    q: torch.Tensor,
    visible_entries: list[list[VisibleBlockEntry]] | list[tuple[VisibleBlockEntry, ...]],
    full_k_cache: torch.Tensor,
    full_v_cache: torch.Tensor,
    quant_cache: QuantCache,
    workspace: torch.Tensor,
    attn_metadata: AttentionMetadata,
    use_triton_gather_dequant: bool = False,
    use_mixed_kv_decode_kernel: bool = False,
    enable_attention_mass_output: bool = False,
    packed_visible_entries: torch.Tensor | None = None,
    packed_visible_entry_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode-only attention over FULL/QUANT visible entries."""
    if q.ndim != 3:
        raise MixedKVReadError(f"decode q must have shape [batch, heads, dim], got {tuple(q.shape)}")
    if len(visible_entries) != q.shape[0]:
        raise MixedKVReadError("visible entry batch size does not match q batch")
    if full_k_cache.shape != full_v_cache.shape:
        raise MixedKVReadError("full K/V cache shapes differ")

    if use_mixed_kv_decode_kernel and _can_try_mixed_kv_decode_kernel(q, full_k_cache, quant_cache, attn_metadata.layer_id):
        try:
            output = _run_decode_mixed_kv_kernel(
                q,
                visible_entries,
                full_k_cache,
                full_v_cache,
                quant_cache,
                attn_metadata,
                enable_attention_mass_output=enable_attention_mass_output,
                packed_visible_entries=packed_visible_entries,
                packed_visible_entry_counts=packed_visible_entry_counts,
            )
            if _validate_mixed_kv_decode_kernel():
                profiler.increment("parity_check_calls")
                reference = _run_decode_materialized(
                    q,
                    visible_entries,
                    full_k_cache,
                    full_v_cache,
                    quant_cache,
                    workspace,
                    attn_metadata,
                    use_triton_gather_dequant=use_triton_gather_dequant,
                )
                if not torch.allclose(output, reference, atol=5e-2, rtol=5e-2):
                    profiler.increment("fused_kernel_fallbacks")
                    return reference
            return output
        except (KernelNotSupportedError, KernelRuntimeError, RuntimeError):
            profiler.increment("fused_kernel_fallbacks")
            pass
    elif use_mixed_kv_decode_kernel:
        profiler.increment("fused_kernel_fallbacks")

    return _run_decode_materialized(
        q,
        visible_entries,
        full_k_cache,
        full_v_cache,
        quant_cache,
        workspace,
        attn_metadata,
        use_triton_gather_dequant=use_triton_gather_dequant,
    )


def _run_decode_materialized(
    q: torch.Tensor,
    visible_entries: list[list[VisibleBlockEntry]] | list[tuple[VisibleBlockEntry, ...]],
    full_k_cache: torch.Tensor,
    full_v_cache: torch.Tensor,
    quant_cache: QuantCache,
    workspace: torch.Tensor,
    attn_metadata: AttentionMetadata,
    *,
    use_triton_gather_dequant: bool,
) -> torch.Tensor:
    profiler.increment("fallback_count")
    outputs = []
    full_cache = FullKVCache(full_k_cache, full_v_cache, layer_id=attn_metadata.layer_id)
    for batch_idx, entries in enumerate(visible_entries):
        materialized = materialize_visible_kv_for_decode(
            entries,
            full_cache,
            quant_cache,
            workspace,
            use_triton_gather_dequant=use_triton_gather_dequant,
        )
        outputs.append(_decode_attention(q[batch_idx], materialized.k, materialized.v, attn_metadata.softmax_scale))
    return torch.stack(outputs, dim=0)


def _run_decode_mixed_kv_kernel(
    q: torch.Tensor,
    visible_entries: list[list[VisibleBlockEntry]] | list[tuple[VisibleBlockEntry, ...]],
    full_k_cache: torch.Tensor,
    full_v_cache: torch.Tensor,
    quant_cache: QuantCache,
    attn_metadata: AttentionMetadata,
    *,
    enable_attention_mass_output: bool,
    packed_visible_entries: torch.Tensor | None,
    packed_visible_entry_counts: torch.Tensor | None,
) -> torch.Tensor:
    entries_tensor = packed_visible_entries
    entry_counts = packed_visible_entry_counts
    if entries_tensor is None or entry_counts is None:
        started = _now()
        entries_tensor, entry_counts = visible_entries_to_tensor(visible_entries, device=q.device)
        profiler.record_ms("visible_table_tensor_pack_ms", _elapsed_ms(started))
    block_attn_mass = None
    if enable_attention_mass_output:
        block_attn_mass = torch.empty(
            q.shape[0],
            q.shape[1],
            entries_tensor.shape[1],
            dtype=torch.float32,
            device=q.device,
        )
    profiler.increment("fused_kernel_calls")
    started = _now()
    try:
        return mixed_kv_decode_attention(
            q,
            full_k_cache,
            full_v_cache,
            quant_cache.q_cache[:, 0, attn_metadata.layer_id],
            quant_cache.q_cache[:, 1, attn_metadata.layer_id],
            quant_cache.scales[:, 0, attn_metadata.layer_id],
            quant_cache.scales[:, 1, attn_metadata.layer_id],
            entries_tensor,
            entry_counts,
            softmax_scale=attn_metadata.softmax_scale,
            block_attn_mass=block_attn_mass,
        )
    finally:
        profiler.record_ms("mixed_kv_decode_kernel_ms", _elapsed_ms(started))


def _can_try_mixed_kv_decode_kernel(
    q: torch.Tensor,
    full_k_cache: torch.Tensor,
    quant_cache: QuantCache,
    layer_id: int,
) -> bool:
    if layer_id < 0 or layer_id >= quant_cache.spec.num_layers:
        return False
    return mixed_kv_decode_attention_supported(
        head_dim=q.shape[-1],
        dtype=q.dtype,
        block_size=full_k_cache.shape[1],
        device=q.device,
        num_q_heads=q.shape[1],
        num_kv_heads=full_k_cache.shape[2],
    )


def _dequantize_quant_entries(
    quant_cache: QuantCache,
    quant_entries: list[VisibleBlockEntry],
    quant_block_ids: list[int],
    layer_id: int,
    workspace: torch.Tensor,
    *,
    use_triton_gather_dequant: bool,
) -> torch.Tensor:
    if not quant_entries:
        return workspace[:0]
    needed_shape = (
        len(quant_entries),
        2,
        quant_cache.spec.block_size,
        quant_cache.spec.num_kv_heads,
        quant_cache.spec.head_dim,
    )
    needed = prod(needed_shape)
    if workspace.numel() < needed:
        raise ScratchOverflowError(f"scratch has {workspace.numel()} elements but needs {needed}")
    output = workspace.reshape(-1)[:needed].view(needed_shape)
    if use_triton_gather_dequant and _can_try_triton_gather_dequant(quant_cache, output, layer_id):
        try:
            ids = torch.tensor(quant_block_ids, dtype=torch.int64, device=output.device)
            started = _now()
            try:
                gather_dequant_triton(
                    quant_cache.q_cache[:, 0, layer_id],
                    quant_cache.q_cache[:, 1, layer_id],
                    quant_cache.scales[:, 0, layer_id],
                    quant_cache.scales[:, 1, layer_id],
                    ids,
                    output[:, 0],
                    output[:, 1],
                    block_size=quant_cache.spec.block_size,
                    head_dim=quant_cache.spec.head_dim,
                )
            finally:
                profiler.record_ms("gather_dequant_ms", _elapsed_ms(started))
            if _validate_triton_gather_dequant():
                profiler.increment("parity_check_calls")
                reference = gather_dequant_reference(quant_cache, quant_entries, torch.empty_like(output), layer_id=layer_id)
                if not torch.allclose(output, reference, atol=3e-2, rtol=3e-2):
                    output.copy_(reference)
            return output
        except (KernelNotSupportedError, KernelRuntimeError, RuntimeError):
            pass
    try:
        started = _now()
        try:
            return quant_cache.dequantize_to_scratch(quant_block_ids, layer_id, output)
        finally:
            profiler.record_ms("gather_dequant_ms", _elapsed_ms(started))
    except ScratchOverflowError:
        raise
    except Exception as exc:
        raise MixedKVReadError(str(exc)) from exc


def _can_try_triton_gather_dequant(quant_cache: QuantCache, output: torch.Tensor, layer_id: int) -> bool:
    if layer_id < 0 or layer_id >= quant_cache.spec.num_layers:
        return False
    return triton_gather_dequant_supported(
        head_dim=quant_cache.spec.head_dim,
        dtype=output.dtype,
        block_size=quant_cache.spec.block_size,
        device=output.device,
    )


def _validate_triton_gather_dequant() -> bool:
    return os.environ.get("NANOVLLM_VALIDATE_TRITON_GATHER_DEQUANT", "").lower() in {"1", "true", "yes"}


def _validate_mixed_kv_decode_kernel() -> bool:
    return os.environ.get("NANOVLLM_VALIDATE_MIXED_KV_DECODE_KERNEL", "").lower() in {"1", "true", "yes"}


def _now() -> float:
    from time import perf_counter

    return perf_counter()


def _elapsed_ms(started: float) -> float:
    return (_now() - started) * 1000.0


def enable_full_reuse_after_quant(storage_id: int, commit_result) -> None:
    """Validate that a quant commit can safely release its old FULL block."""
    if commit_result.storage_id != storage_id:
        raise FullReuseSafetyError("commit result storage_id does not match requested storage_id")
    if commit_result.quant_block_id is None:
        raise FullReuseSafetyError("commit result has no quant_block_id")
    if commit_result.released_full_block_id is None:
        raise FullReuseSafetyError("commit did not release a full block")
    if commit_result.reclaimed_full_equiv_blocks != 1:
        raise FullReuseSafetyError("commit result did not reclaim exactly one full-equivalent block")


def _decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    num_q_heads = q.shape[0]
    num_kv_heads = k.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise MixedKVReadError("q heads must be divisible by kv heads for fallback GQA")
    if num_q_heads != num_kv_heads:
        repeat = num_q_heads // num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    scores = torch.einsum("hd,thd->ht", q.to(torch.float32), k.to(torch.float32)) * float(scale)
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("ht,thd->hd", weights, v.to(torch.float32))
    return out.to(q.dtype)


def _prefill_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_start: int, scale: float) -> torch.Tensor:
    outputs = []
    for token_offset in range(q.shape[0]):
        context_end = query_start + token_offset + 1
        outputs.append(_decode_attention(q[token_offset], k[:context_end], v[:context_end], scale))
    return torch.stack(outputs, dim=0)


def _validate_prefill_write_targets(
    slot_mapping: torch.Tensor,
    entries: list[VisibleBlockEntry],
    block_size: int,
) -> None:
    if slot_mapping.numel() == 0:
        return
    if torch.any(slot_mapping < 0):
        raise InvalidWriteTargetError("prefill slot_mapping contains an invalid write slot")
    full_block_ids = {entry.full_block_id for entry in entries if entry.state == KVBlockState.FULL}
    write_block_ids = torch.div(slot_mapping.detach().to("cpu"), block_size, rounding_mode="floor").unique().tolist()
    missing = sorted(int(block_id) for block_id in write_block_ids if int(block_id) not in full_block_ids)
    if missing:
        raise InvalidWriteTargetError(f"prefill slot_mapping points to non-FULL blocks: {missing}")


def validate_unfinished_prefill_policy(seq_state: Any, reclaim_plan: Any) -> None:
    """Reject EVICT candidates for unfinished-prefill sequences."""
    is_prefill = bool(getattr(seq_state, "is_prefill", False))
    is_finished = bool(getattr(seq_state, "is_finished", False))
    cached = int(getattr(seq_state, "num_cached_tokens", 0) or 0)
    total = int(getattr(seq_state, "num_tokens", cached) or cached)
    if not is_prefill or is_finished or cached >= total:
        return
    candidates = getattr(reclaim_plan, "candidates", ())
    for candidate in candidates:
        if getattr(candidate, "state", None) == KVBlockState.EVICT:
            raise PolicyInvariantError("unfinished-prefill sequences cannot have EVICT candidates before P5")


def _validate_visible_entries(entries: list[VisibleBlockEntry]) -> None:
    previous_logical_end = 0
    previous_visible_end = 0
    for entry in entries:
        if entry.logical_start < previous_logical_end:
            raise MixedKVReadError("visible entries are not in monotonic logical order")
        if entry.visible_start != previous_visible_end:
            raise MixedKVReadError("visible entries are not in contiguous visible order")
        if entry.logical_end <= entry.logical_start:
            raise MixedKVReadError("visible entry has an invalid logical span")
        if entry.visible_end <= entry.visible_start:
            raise MixedKVReadError("visible entry has an invalid visible span")
        previous_logical_end = entry.logical_end
        previous_visible_end = entry.visible_end


def _cfg_int(cfg: Any, *names: str) -> int:
    for name in names:
        if hasattr(cfg, name):
            value = int(getattr(cfg, name))
            if value < 1:
                raise WorkspacePlanningError(f"{name} must be positive")
            return value
    raise WorkspacePlanningError(f"cache config missing one of: {', '.join(names)}")


def _dtype_itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()
