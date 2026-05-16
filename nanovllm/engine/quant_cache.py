"""Q8 quantized KV cache storage used by the P3 shadow path."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from nanovllm.kernels.q8_kv import dequantize_q8_reference, quantize_q8_reference


class QuantCacheError(RuntimeError):
    pass


class QuantPoolExhaustedError(QuantCacheError):
    pass


class ScratchOverflowError(QuantCacheError):
    pass


@dataclass(frozen=True, slots=True)
class QuantCacheSpec:
    num_quant_blocks: int
    num_layers: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device | str = "cpu"

    @property
    def block_shape(self) -> tuple[int, int, int, int, int]:
        return (2, self.num_layers, self.block_size, self.num_kv_heads, self.head_dim)

    @property
    def scale_shape(self) -> tuple[int, int, int, int, int]:
        return (2, self.num_layers, self.block_size, self.num_kv_heads, 1)


class QuantCache:
    def __init__(self, spec: QuantCacheSpec):
        if spec.num_quant_blocks < 0:
            raise QuantCacheError("num_quant_blocks must be non-negative")
        self.spec = spec
        shape = (spec.num_quant_blocks, *spec.block_shape)
        scale_shape = (spec.num_quant_blocks, *spec.scale_shape)
        self.q_cache = torch.empty(shape, dtype=torch.int8, device=spec.device)
        self.scales = torch.empty(scale_shape, dtype=torch.float32, device=spec.device)
        self.free_quant_block_ids: deque[int] = deque(range(spec.num_quant_blocks))
        self.used_quant_block_ids: set[int] = set()

    @property
    def num_blocks(self) -> int:
        return self.spec.num_quant_blocks

    def allocate(self) -> int:
        if not self.free_quant_block_ids:
            raise QuantPoolExhaustedError("Q8 quant pool is exhausted")
        quant_block_id = self.free_quant_block_ids.popleft()
        self.used_quant_block_ids.add(quant_block_id)
        return quant_block_id

    def free(self, quant_block_id: int) -> None:
        self._check_quant_block_id(quant_block_id)
        if quant_block_id not in self.used_quant_block_ids:
            raise QuantCacheError(f"quant block {quant_block_id} is not allocated")
        self.used_quant_block_ids.remove(quant_block_id)
        self.free_quant_block_ids.append(quant_block_id)

    def write_from_full(self, quant_block_id: int, full_block: torch.Tensor) -> None:
        self._check_quant_block_id(quant_block_id)
        if quant_block_id not in self.used_quant_block_ids:
            raise QuantCacheError(f"quant block {quant_block_id} is not allocated")
        if tuple(full_block.shape) != self.spec.block_shape:
            raise QuantCacheError(f"full block shape {tuple(full_block.shape)} does not match {self.spec.block_shape}")
        quantized, scales = quantize_q8_reference(full_block)
        self.q_cache[quant_block_id].copy_(quantized)
        self.scales[quant_block_id].copy_(scales)

    def dequantize_block(self, quant_block_id: int, dtype: torch.dtype | None = None) -> torch.Tensor:
        self._check_quant_block_id(quant_block_id)
        if quant_block_id not in self.used_quant_block_ids:
            raise QuantCacheError(f"quant block {quant_block_id} is not allocated")
        return dequantize_q8_reference(self.q_cache[quant_block_id], self.scales[quant_block_id], dtype or self.spec.dtype)

    def dequantize_to_scratch(
        self,
        quant_block_ids: list[int],
        layer_id: int,
        scratch: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        if layer_id < 0 or layer_id >= self.spec.num_layers:
            raise QuantCacheError(f"invalid layer_id: {layer_id}")
        for quant_block_id in quant_block_ids:
            self._check_quant_block_id(quant_block_id)
            if quant_block_id not in self.used_quant_block_ids:
                raise QuantCacheError(f"quant block {quant_block_id} is not allocated")

        output_shape = (len(quant_block_ids), 2, self.spec.block_size, self.spec.num_kv_heads, self.spec.head_dim)
        needed = 1
        for dim in output_shape:
            needed *= dim
        if scratch.numel() < needed:
            raise ScratchOverflowError(f"scratch has {scratch.numel()} elements but needs {needed}")

        q = self.q_cache[quant_block_ids, :, layer_id]
        scales = self.scales[quant_block_ids, :, layer_id]
        out = scratch.reshape(-1)[:needed].view(output_shape)
        if stream is not None and q.is_cuda:
            with torch.cuda.stream(stream):
                return dequantize_q8_reference(q, scales, self.spec.dtype, out=out)
        return dequantize_q8_reference(q, scales, self.spec.dtype, out=out)

    def _check_quant_block_id(self, quant_block_id: int) -> None:
        if quant_block_id < 0 or quant_block_id >= self.spec.num_quant_blocks:
            raise QuantCacheError(f"invalid quant block id: {quant_block_id}")
