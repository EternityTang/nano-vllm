"""Decode-only mixed FULL/Q8 KV attention kernel."""

from __future__ import annotations

import torch

from nanovllm.engine.kv_meta import KVBlockState
from nanovllm.engine.visible_tables import VisibleBlockEntry
from nanovllm.kernels.triton_gather_dequant import KernelNotSupportedError, KernelRuntimeError

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - import availability is environment-specific.
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


FULL_STATE = 0
QUANT_STATE = 1
EVICT_STATE = 2
VISIBLE_ENTRY_COLUMNS = 5


def mixed_kv_decode_attention_supported(
    *,
    head_dim: int,
    dtype: torch.dtype,
    block_size: int,
    device: torch.device | str,
    num_q_heads: int,
    num_kv_heads: int,
) -> bool:
    if not _TRITON_AVAILABLE:
        return False
    device = torch.device(device)
    if device.type != "cuda":
        return False
    if dtype not in {torch.float16, torch.bfloat16}:
        return False
    if int(block_size) < 1 or int(block_size) > 256:
        return False
    if int(head_dim) not in {8, 16, 32, 64, 128, 256}:
        return False
    if int(num_q_heads) < 1 or int(num_kv_heads) < 1:
        return False
    return int(num_q_heads) % int(num_kv_heads) == 0


def visible_entries_to_tensor(
    visible_entries: list[list[VisibleBlockEntry]] | list[tuple[VisibleBlockEntry, ...]],
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(visible_entries)
    max_entries = max((len(entries) for entries in visible_entries), default=0)
    rows = [[[-1] * VISIBLE_ENTRY_COLUMNS for _ in range(max_entries)] for _ in range(batch_size)]
    counts = [0] * batch_size
    for batch_idx, entries in enumerate(visible_entries):
        counts[batch_idx] = len(entries)
        for entry_idx, entry in enumerate(entries):
            rows[batch_idx][entry_idx] = [
                _state_code(entry.state),
                -1 if entry.full_block_id is None else int(entry.full_block_id),
                -1 if entry.quant_block_id is None else int(entry.quant_block_id),
                int(entry.logical_start),
                int(entry.logical_end),
            ]
    entries_tensor = torch.tensor(rows, dtype=torch.int64).to(device=device, non_blocking=True)
    entry_counts = torch.tensor(counts, dtype=torch.int32).to(device=device, non_blocking=True)
    return entries_tensor, entry_counts


def mixed_kv_decode_attention_reference(
    q: torch.Tensor,
    full_k_cache: torch.Tensor,
    full_v_cache: torch.Tensor,
    quant_k_cache: torch.Tensor,
    quant_v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    visible_entries: torch.Tensor,
    entry_counts: torch.Tensor,
    *,
    softmax_scale: float,
    head_map: torch.Tensor | None = None,
    block_attn_mass: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_tensor_inputs(
        q,
        full_k_cache,
        full_v_cache,
        quant_k_cache,
        quant_v_cache,
        k_scale,
        v_scale,
        visible_entries,
        entry_counts,
    )
    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads = full_k_cache.shape[2]
    block_size = full_k_cache.shape[1]
    if head_map is None:
        if num_q_heads % num_kv_heads != 0:
            raise KernelNotSupportedError("q heads must be divisible by kv heads")
        repeat = num_q_heads // num_kv_heads
        head_map = torch.div(torch.arange(num_q_heads, device=q.device), repeat, rounding_mode="floor")
    output = torch.empty_like(q)
    if block_attn_mass is not None:
        block_attn_mass.zero_()

    for batch_idx in range(batch_size):
        entries = visible_entries[batch_idx, : int(entry_counts[batch_idx])]
        keys = []
        values = []
        block_lengths = []
        for entry in entries:
            state = int(entry[0].item())
            start = int(entry[3].item())
            end = int(entry[4].item())
            length = end - start
            if state == EVICT_STATE or length <= 0:
                block_lengths.append(0)
                continue
            offset = start % block_size
            if state == FULL_STATE:
                block_id = int(entry[1].item())
                k_block = full_k_cache[block_id, offset : offset + length]
                v_block = full_v_cache[block_id, offset : offset + length]
            elif state == QUANT_STATE:
                block_id = int(entry[2].item())
                k_block = quant_k_cache[block_id, offset : offset + length].to(torch.float32)
                v_block = quant_v_cache[block_id, offset : offset + length].to(torch.float32)
                k_block = k_block * k_scale[block_id, offset : offset + length]
                v_block = v_block * v_scale[block_id, offset : offset + length]
            else:
                raise KernelRuntimeError(f"unsupported visible state: {state}")
            keys.append(k_block.to(q.dtype))
            values.append(v_block.to(q.dtype))
            block_lengths.append(length)
        if not keys:
            raise KernelRuntimeError("visible entries contain no readable tokens")
        k_seq = torch.cat(keys, dim=0)
        v_seq = torch.cat(values, dim=0)
        for q_head in range(num_q_heads):
            kv_head = int(head_map[q_head].item())
            scores = torch.einsum(
                "d,td->t",
                q[batch_idx, q_head].to(torch.float32),
                k_seq[:, kv_head].to(torch.float32),
            ) * float(softmax_scale)
            weights = torch.softmax(scores, dim=-1)
            output[batch_idx, q_head] = torch.einsum("t,td->d", weights, v_seq[:, kv_head].to(torch.float32)).to(q.dtype)
            if block_attn_mass is not None:
                cursor = 0
                for entry_idx, length in enumerate(block_lengths):
                    if length > 0:
                        block_attn_mass[batch_idx, q_head, entry_idx] = weights[cursor : cursor + length].sum().to(
                            block_attn_mass.dtype
                        )
                        cursor += length
    return output


def mixed_kv_decode_attention(
    q: torch.Tensor,
    full_k_cache: torch.Tensor,
    full_v_cache: torch.Tensor,
    quant_k_cache: torch.Tensor,
    quant_v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    visible_entries: torch.Tensor,
    entry_counts: torch.Tensor,
    *,
    softmax_scale: float,
    head_map: torch.Tensor | None = None,
    block_attn_mass: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_tensor_inputs(
        q,
        full_k_cache,
        full_v_cache,
        quant_k_cache,
        quant_v_cache,
        k_scale,
        v_scale,
        visible_entries,
        entry_counts,
    )
    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads = full_k_cache.shape[2]
    block_size = full_k_cache.shape[1]
    if not mixed_kv_decode_attention_supported(
        head_dim=head_dim,
        dtype=q.dtype,
        block_size=block_size,
        device=q.device,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    ):
        raise KernelNotSupportedError("mixed-KV decode attention kernel does not support this shape or device")
    if not all(
        tensor.is_cuda
        for tensor in (q, full_k_cache, full_v_cache, quant_k_cache, quant_v_cache, k_scale, v_scale, visible_entries)
    ):
        raise KernelNotSupportedError("mixed-KV decode attention kernel requires CUDA tensors")
    if head_map is None:
        repeat = num_q_heads // num_kv_heads
        head_map = torch.div(torch.arange(num_q_heads, device=q.device, dtype=torch.int64), repeat, rounding_mode="floor")
    if head_map.device != q.device:
        raise KernelNotSupportedError("head_map must be on the same device as q")
    if block_attn_mass is not None and tuple(block_attn_mass.shape) != (batch_size, num_q_heads, visible_entries.shape[1]):
        raise KernelNotSupportedError("block_attn_mass must have shape [batch, q_heads, max_entries]")

    output = torch.empty_like(q)
    mass = block_attn_mass if block_attn_mass is not None else output
    try:
        grid = (batch_size, num_q_heads)
        _mixed_kv_decode_attention_kernel[grid](
            q,
            full_k_cache,
            full_v_cache,
            quant_k_cache,
            quant_v_cache,
            k_scale,
            v_scale,
            visible_entries,
            entry_counts,
            head_map,
            output,
            mass,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            full_k_cache.stride(0),
            full_k_cache.stride(1),
            full_k_cache.stride(2),
            full_k_cache.stride(3),
            quant_k_cache.stride(0),
            quant_k_cache.stride(1),
            quant_k_cache.stride(2),
            quant_k_cache.stride(3),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            mass.stride(0),
            mass.stride(1),
            mass.stride(2) if mass.ndim == 3 else 0,
            float(softmax_scale),
            MAX_ENTRIES=visible_entries.shape[1],
            ENTRY_COLUMNS=VISIBLE_ENTRY_COLUMNS,
            FULL_CODE=FULL_STATE,
            QUANT_CODE=QUANT_STATE,
            EVICT_CODE=EVICT_STATE,
            BLOCK_SIZE=block_size,
            D=head_dim,
            BLOCK_D=triton.next_power_of_2(head_dim),
            HAS_MASS=block_attn_mass is not None,
        )
    except Exception as exc:
        raise KernelRuntimeError(str(exc)) from exc
    return output


def _validate_tensor_inputs(
    q: torch.Tensor,
    full_k_cache: torch.Tensor,
    full_v_cache: torch.Tensor,
    quant_k_cache: torch.Tensor,
    quant_v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    visible_entries: torch.Tensor,
    entry_counts: torch.Tensor,
) -> None:
    if q.ndim != 3:
        raise KernelNotSupportedError("q must have shape [batch, q_heads, head_dim]")
    if full_k_cache.shape != full_v_cache.shape or quant_k_cache.shape != quant_v_cache.shape:
        raise KernelNotSupportedError("K/V cache shapes must match")
    if full_k_cache.ndim != 4 or quant_k_cache.ndim != 4:
        raise KernelNotSupportedError("cache tensors must have shape [blocks, block, kv_heads, head_dim]")
    if k_scale.shape != v_scale.shape or k_scale.ndim != 4 or k_scale.shape[-1] != 1:
        raise KernelNotSupportedError("scale tensors must have shape [blocks, block, kv_heads, 1]")
    if visible_entries.ndim != 3 or visible_entries.shape[-1] != VISIBLE_ENTRY_COLUMNS:
        raise KernelNotSupportedError("visible_entries must have shape [batch, max_entries, 5]")
    if entry_counts.ndim != 1 or entry_counts.shape[0] != q.shape[0]:
        raise KernelNotSupportedError("entry_counts must have shape [batch]")
    if q.shape[2] != full_k_cache.shape[3] or q.shape[2] != quant_k_cache.shape[3]:
        raise KernelNotSupportedError("head dimensions differ")
    if full_k_cache.shape[1:3] != quant_k_cache.shape[1:3]:
        raise KernelNotSupportedError("FULL and QUANT cache block/head shapes differ")
    if k_scale.shape[:3] != quant_k_cache.shape[:3]:
        raise KernelNotSupportedError("scale shape does not match quant cache")
    if visible_entries.shape[0] != q.shape[0]:
        raise KernelNotSupportedError("visible entry batch size differs from q")
    if quant_k_cache.dtype != torch.int8 or quant_v_cache.dtype != torch.int8:
        raise KernelNotSupportedError("quant cache tensors must be int8")


def _state_code(state: KVBlockState) -> int:
    if state == KVBlockState.FULL:
        return FULL_STATE
    if state == KVBlockState.QUANT:
        return QUANT_STATE
    if state == KVBlockState.EVICT:
        return EVICT_STATE
    raise KernelRuntimeError(f"unsupported visible state: {state}")


if _TRITON_AVAILABLE:

    @triton.jit
    def _mixed_kv_decode_attention_kernel(
        q,
        full_k,
        full_v,
        quant_k,
        quant_v,
        k_scale,
        v_scale,
        visible_entries,
        entry_counts,
        head_map,
        output,
        block_attn_mass,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_d: tl.constexpr,
        full_stride_b: tl.constexpr,
        full_stride_t: tl.constexpr,
        full_stride_h: tl.constexpr,
        full_stride_d: tl.constexpr,
        quant_stride_b: tl.constexpr,
        quant_stride_t: tl.constexpr,
        quant_stride_h: tl.constexpr,
        quant_stride_d: tl.constexpr,
        scale_stride_b: tl.constexpr,
        scale_stride_t: tl.constexpr,
        scale_stride_h: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_d: tl.constexpr,
        mass_stride_b: tl.constexpr,
        mass_stride_h: tl.constexpr,
        mass_stride_e: tl.constexpr,
        softmax_scale: tl.constexpr,
        MAX_ENTRIES: tl.constexpr,
        ENTRY_COLUMNS: tl.constexpr,
        FULL_CODE: tl.constexpr,
        QUANT_CODE: tl.constexpr,
        EVICT_CODE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_MASS: tl.constexpr,
    ):
        batch = tl.program_id(0)
        q_head = tl.program_id(1)
        kv_head = tl.load(head_map + q_head)
        entry_count = tl.load(entry_counts + batch)
        offsets = tl.arange(0, BLOCK_D)
        d_mask = offsets < D
        q_vec = tl.load(q + batch * q_stride_b + q_head * q_stride_h + offsets * q_stride_d, mask=d_mask, other=0).to(
            tl.float32
        )

        max_score = -3.4028234663852886e38
        for entry_idx in range(0, MAX_ENTRIES):
            entry_base = (batch * MAX_ENTRIES + entry_idx) * ENTRY_COLUMNS
            state = tl.load(visible_entries + entry_base + 0)
            full_block = tl.load(visible_entries + entry_base + 1)
            quant_block = tl.load(visible_entries + entry_base + 2)
            logical_start = tl.load(visible_entries + entry_base + 3)
            logical_end = tl.load(visible_entries + entry_base + 4)
            length = logical_end - logical_start
            token_base = logical_start % BLOCK_SIZE
            entry_valid = (entry_idx < entry_count) & (state != EVICT_CODE) & (length > 0)
            for token in range(0, BLOCK_SIZE):
                token_valid = entry_valid & (token < length)
                cache_token = token_base + token
                full_offsets = full_block * full_stride_b + cache_token * full_stride_t + kv_head * full_stride_h + offsets * full_stride_d
                quant_offsets = quant_block * quant_stride_b + cache_token * quant_stride_t + kv_head * quant_stride_h + offsets * quant_stride_d
                scale_offset = quant_block * scale_stride_b + cache_token * scale_stride_t + kv_head * scale_stride_h
                k_full = tl.load(full_k + full_offsets, mask=d_mask & token_valid & (state == FULL_CODE), other=0).to(tl.float32)
                k_q = tl.load(quant_k + quant_offsets, mask=d_mask & token_valid & (state == QUANT_CODE), other=0).to(tl.float32)
                k_s = tl.load(k_scale + scale_offset, mask=token_valid & (state == QUANT_CODE), other=0).to(tl.float32)
                k_vec = tl.where(state == FULL_CODE, k_full, k_q * k_s)
                score = tl.sum(q_vec * k_vec, axis=0) * softmax_scale
                score = tl.where(token_valid, score, -3.4028234663852886e38)
                max_score = tl.maximum(max_score, score)

        denom = 0.0
        acc = tl.zeros((BLOCK_D,), tl.float32)
        for entry_idx in range(0, MAX_ENTRIES):
            entry_base = (batch * MAX_ENTRIES + entry_idx) * ENTRY_COLUMNS
            state = tl.load(visible_entries + entry_base + 0)
            full_block = tl.load(visible_entries + entry_base + 1)
            quant_block = tl.load(visible_entries + entry_base + 2)
            logical_start = tl.load(visible_entries + entry_base + 3)
            logical_end = tl.load(visible_entries + entry_base + 4)
            length = logical_end - logical_start
            token_base = logical_start % BLOCK_SIZE
            entry_valid = (entry_idx < entry_count) & (state != EVICT_CODE) & (length > 0)
            for token in range(0, BLOCK_SIZE):
                token_valid = entry_valid & (token < length)
                cache_token = token_base + token
                full_offsets = full_block * full_stride_b + cache_token * full_stride_t + kv_head * full_stride_h + offsets * full_stride_d
                quant_offsets = quant_block * quant_stride_b + cache_token * quant_stride_t + kv_head * quant_stride_h + offsets * quant_stride_d
                scale_offset = quant_block * scale_stride_b + cache_token * scale_stride_t + kv_head * scale_stride_h
                k_full = tl.load(full_k + full_offsets, mask=d_mask & token_valid & (state == FULL_CODE), other=0).to(tl.float32)
                v_full = tl.load(full_v + full_offsets, mask=d_mask & token_valid & (state == FULL_CODE), other=0).to(tl.float32)
                k_q = tl.load(quant_k + quant_offsets, mask=d_mask & token_valid & (state == QUANT_CODE), other=0).to(tl.float32)
                v_q = tl.load(quant_v + quant_offsets, mask=d_mask & token_valid & (state == QUANT_CODE), other=0).to(tl.float32)
                k_s = tl.load(k_scale + scale_offset, mask=token_valid & (state == QUANT_CODE), other=0).to(tl.float32)
                v_s = tl.load(v_scale + scale_offset, mask=token_valid & (state == QUANT_CODE), other=0).to(tl.float32)
                k_vec = tl.where(state == FULL_CODE, k_full, k_q * k_s)
                v_vec = tl.where(state == FULL_CODE, v_full, v_q * v_s)
                score = tl.sum(q_vec * k_vec, axis=0) * softmax_scale
                exp_score = tl.exp(score - max_score)
                exp_score = tl.where(token_valid, exp_score, 0.0)
                denom += exp_score
                acc += exp_score * v_vec

        if HAS_MASS:
            for entry_idx in range(0, MAX_ENTRIES):
                entry_base = (batch * MAX_ENTRIES + entry_idx) * ENTRY_COLUMNS
                state = tl.load(visible_entries + entry_base + 0)
                full_block = tl.load(visible_entries + entry_base + 1)
                quant_block = tl.load(visible_entries + entry_base + 2)
                logical_start = tl.load(visible_entries + entry_base + 3)
                logical_end = tl.load(visible_entries + entry_base + 4)
                length = logical_end - logical_start
                token_base = logical_start % BLOCK_SIZE
                entry_valid = (entry_idx < entry_count) & (state != EVICT_CODE) & (length > 0)
                mass = 0.0
                for token in range(0, BLOCK_SIZE):
                    token_valid = entry_valid & (token < length)
                    cache_token = token_base + token
                    full_offsets = full_block * full_stride_b + cache_token * full_stride_t + kv_head * full_stride_h + offsets * full_stride_d
                    quant_offsets = quant_block * quant_stride_b + cache_token * quant_stride_t + kv_head * quant_stride_h + offsets * quant_stride_d
                    scale_offset = quant_block * scale_stride_b + cache_token * scale_stride_t + kv_head * scale_stride_h
                    k_full = tl.load(full_k + full_offsets, mask=d_mask & token_valid & (state == FULL_CODE), other=0).to(tl.float32)
                    k_q = tl.load(quant_k + quant_offsets, mask=d_mask & token_valid & (state == QUANT_CODE), other=0).to(tl.float32)
                    k_s = tl.load(k_scale + scale_offset, mask=token_valid & (state == QUANT_CODE), other=0).to(tl.float32)
                    k_vec = tl.where(state == FULL_CODE, k_full, k_q * k_s)
                    score = tl.sum(q_vec * k_vec, axis=0) * softmax_scale
                    exp_score = tl.exp(score - max_score)
                    exp_score = tl.where(token_valid, exp_score, 0.0)
                    mass += exp_score
                mass_value = tl.where(denom > 0.0, mass / denom, 0.0)
                tl.store(block_attn_mass + batch * mass_stride_b + q_head * mass_stride_h + entry_idx * mass_stride_e, mass_value)

        out_vec = acc / denom
        tl.store(output + batch * out_stride_b + q_head * out_stride_h + offsets * out_stride_d, out_vec, mask=d_mask)
else:
    _mixed_kv_decode_attention_kernel = None
