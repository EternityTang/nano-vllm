"""Triton gather+dequant for Q8 KV materialization."""

from __future__ import annotations

import torch

from nanovllm.engine.kv_meta import KVBlockState
from nanovllm.engine.quant_cache import QuantCache
from nanovllm.engine.visible_tables import VisibleBlockEntry

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - import availability is environment-specific.
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


class KernelNotSupportedError(RuntimeError):
    pass


class KernelRuntimeError(RuntimeError):
    pass


def triton_gather_dequant_supported(
    head_dim: int,
    dtype: torch.dtype,
    block_size: int,
    device: torch.device | str,
) -> bool:
    if not _TRITON_AVAILABLE:
        return False
    device = torch.device(device)
    if device.type != "cuda":
        return False
    if dtype not in {torch.float16, torch.bfloat16}:
        return False
    return int(block_size) > 0 and int(head_dim) in {8, 16, 32, 64, 128, 256}


def gather_dequant_reference(
    quant_cache: QuantCache,
    entries: list[VisibleBlockEntry] | tuple[VisibleBlockEntry, ...],
    output: torch.Tensor,
    *,
    layer_id: int = 0,
) -> torch.Tensor:
    quant_entries = [entry for entry in entries if entry.state == KVBlockState.QUANT]
    if not quant_entries:
        return output[:0]
    quant_block_ids = []
    for entry in quant_entries:
        if entry.quant_block_id is None:
            raise KernelRuntimeError(f"QUANT entry {entry.storage_id} is missing quant_block_id")
        quant_block_ids.append(entry.quant_block_id)
    needed_shape = (
        len(quant_entries),
        2,
        quant_cache.spec.block_size,
        quant_cache.spec.num_kv_heads,
        quant_cache.spec.head_dim,
    )
    needed = 1
    for dim in needed_shape:
        needed *= dim
    if output.numel() < needed:
        raise KernelRuntimeError(f"output has {output.numel()} elements but needs {needed}")
    out = output.reshape(-1)[:needed].view(needed_shape)
    return quant_cache.dequantize_to_scratch(quant_block_ids, layer_id, out)


def gather_dequant_triton(
    quant_k: torch.Tensor,
    quant_v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    visible_entries: torch.Tensor,
    output_k: torch.Tensor,
    output_v: torch.Tensor,
    *,
    block_size: int,
    head_dim: int,
) -> None:
    if not triton_gather_dequant_supported(head_dim, output_k.dtype, block_size, output_k.device):
        raise KernelNotSupportedError("Triton gather/dequant does not support this shape or device")
    if visible_entries.numel() == 0:
        return
    if not (quant_k.is_cuda and quant_v.is_cuda and k_scale.is_cuda and v_scale.is_cuda and visible_entries.is_cuda):
        raise KernelNotSupportedError("Triton gather/dequant requires CUDA tensors")
    if quant_k.dtype != torch.int8 or quant_v.dtype != torch.int8:
        raise KernelNotSupportedError("Triton gather/dequant expects int8 quant tensors")
    if output_k.shape != output_v.shape:
        raise KernelNotSupportedError("output K/V shapes must match")
    if output_k.ndim != 4:
        raise KernelNotSupportedError("output K/V must have shape [blocks, block, heads, dim]")
    if output_k.shape[0] != visible_entries.numel():
        raise KernelNotSupportedError("visible entry ids must match output block count")
    if output_k.shape[1] != block_size or output_k.shape[-1] != head_dim:
        raise KernelNotSupportedError("output shape does not match block_size/head_dim")

    try:
        grid = (visible_entries.numel(), block_size, output_k.shape[2])
        _gather_dequant_kernel[grid](
            quant_k,
            quant_v,
            k_scale,
            v_scale,
            visible_entries,
            output_k,
            output_v,
            output_k.shape[2],
            quant_k.stride(0),
            quant_k.stride(1),
            quant_k.stride(2),
            quant_k.stride(3),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            output_k.stride(0),
            output_k.stride(1),
            output_k.stride(2),
            output_k.stride(3),
            D=head_dim,
            BLOCK_D=triton.next_power_of_2(head_dim),
        )
    except Exception as exc:
        raise KernelRuntimeError(str(exc)) from exc


if _TRITON_AVAILABLE:

    @triton.jit
    def _gather_dequant_kernel(
        quant_k,
        quant_v,
        k_scale,
        v_scale,
        block_ids,
        output_k,
        output_v,
        num_heads: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_t: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_d: tl.constexpr,
        s_stride_b: tl.constexpr,
        s_stride_t: tl.constexpr,
        s_stride_h: tl.constexpr,
        o_stride_b: tl.constexpr,
        o_stride_t: tl.constexpr,
        o_stride_h: tl.constexpr,
        o_stride_d: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        out_block = tl.program_id(0)
        token = tl.program_id(1)
        head = tl.program_id(2)
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < D
        quant_block = tl.load(block_ids + out_block)

        q_offsets = quant_block * q_stride_b + token * q_stride_t + head * q_stride_h + offsets * q_stride_d
        s_offset = quant_block * s_stride_b + token * s_stride_t + head * s_stride_h
        o_offsets = out_block * o_stride_b + token * o_stride_t + head * o_stride_h + offsets * o_stride_d

        k_q = tl.load(quant_k + q_offsets, mask=mask, other=0).to(tl.float32)
        v_q = tl.load(quant_v + q_offsets, mask=mask, other=0).to(tl.float32)
        k_s = tl.load(k_scale + s_offset).to(tl.float32)
        v_s = tl.load(v_scale + s_offset).to(tl.float32)
        tl.store(output_k + o_offsets, k_q * k_s, mask=mask)
        tl.store(output_v + o_offsets, v_q * v_s, mask=mask)
else:
    _gather_dequant_kernel = None
