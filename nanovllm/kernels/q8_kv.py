"""Torch reference Q8 KV quantization helpers."""

from __future__ import annotations

import torch


class QuantizationKernelError(RuntimeError):
    pass


def quantize_q8_reference(tensor: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 quantization with one scale per last-dim vector."""
    if not tensor.is_floating_point():
        raise QuantizationKernelError("Q8 reference quantization expects a floating-point tensor")
    if tensor.numel() == 0:
        raise QuantizationKernelError("Q8 reference quantization expects a non-empty tensor")
    if tensor.shape[-1] < 1:
        raise QuantizationKernelError("Q8 reference quantization expects a non-empty head dimension")

    source = tensor.to(torch.float32)
    max_abs = source.abs().amax(dim=-1, keepdim=True)
    scales = torch.clamp(max_abs / 127.0, min=eps).to(torch.float32)
    quantized = torch.clamp(torch.round(source / scales), min=-127, max=127).to(torch.int8)
    return quantized, scales


def dequantize_q8_reference(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if quantized.dtype != torch.int8:
        raise QuantizationKernelError("Q8 reference dequantization expects int8 input")
    if scales.shape != quantized.shape[:-1] + (1,):
        raise QuantizationKernelError(f"scale shape {tuple(scales.shape)} does not match quantized shape {tuple(quantized.shape)}")

    result = quantized.to(torch.float32) * scales.to(torch.float32)
    if dtype is not None:
        result = result.to(dtype)
    if out is not None:
        if out.shape != quantized.shape:
            raise QuantizationKernelError(f"scratch shape {tuple(out.shape)} does not match result shape {tuple(quantized.shape)}")
        out.copy_(result)
        return out
    return result


def q8_scale_numel(block_shape: tuple[int, ...]) -> int:
    if len(block_shape) < 1 or block_shape[-1] < 1:
        raise QuantizationKernelError(f"invalid Q8 block shape: {block_shape}")
    numel = 1
    for dim in block_shape[:-1]:
        numel *= dim
    return numel


def q8_block_nbytes(block_shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in block_shape:
        if dim < 1:
            raise QuantizationKernelError(f"invalid Q8 block shape: {block_shape}")
        numel *= dim
    return numel


def q8_scale_nbytes(block_shape: tuple[int, ...], scale_dtype_bytes: int = 4) -> int:
    return q8_scale_numel(block_shape) * scale_dtype_bytes
