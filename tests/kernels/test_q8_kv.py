"""Regression tests for the P3 torch reference Q8 KV path."""

from __future__ import annotations

import unittest

import torch

from nanovllm.engine.quant_cache import QuantCache, QuantCacheSpec, ScratchOverflowError
from nanovllm.kernels.q8_kv import dequantize_q8_reference, quantize_q8_reference


class Q8KVKernelTest(unittest.TestCase):
    def test_q8_round_trip_uses_calibrated_256_block_size(self):
        torch.manual_seed(0)
        full = torch.randn(2, 2, 256, 3, 16, dtype=torch.float16)

        quantized, scales = quantize_q8_reference(full)
        restored = dequantize_q8_reference(quantized, scales, torch.float16)

        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(scales.shape, full.shape[:-1] + (1,))
        self.assertLessEqual((full.float() - restored.float()).abs().max().item(), scales.max().item() + 1e-3)

    def test_quant_cache_dequantizes_layer_to_scratch(self):
        torch.manual_seed(1)
        spec = QuantCacheSpec(num_quant_blocks=2, num_layers=2, block_size=256, num_kv_heads=2, head_dim=8, dtype=torch.float16)
        cache = QuantCache(spec)
        full = torch.randn(spec.block_shape, dtype=torch.float16)

        quant_id = cache.allocate()
        cache.write_from_full(quant_id, full)
        scratch = torch.empty(1, 2, spec.block_size, spec.num_kv_heads, spec.head_dim, dtype=torch.float16)
        restored = cache.dequantize_to_scratch([quant_id], layer_id=1, scratch=scratch)

        self.assertEqual(restored.data_ptr(), scratch.data_ptr())
        self.assertEqual(restored.shape, scratch.shape)
        self.assertLessEqual((full[:, 1].float().unsqueeze(0) - restored.float()).abs().max().item(), 0.03)

    def test_scratch_overflow_is_rejected(self):
        spec = QuantCacheSpec(num_quant_blocks=1, num_layers=1, block_size=256, num_kv_heads=1, head_dim=8, dtype=torch.float16)
        cache = QuantCache(spec)
        quant_id = cache.allocate()
        cache.write_from_full(quant_id, torch.ones(spec.block_shape, dtype=torch.float16))

        with self.assertRaises(ScratchOverflowError):
            cache.dequantize_to_scratch([quant_id], layer_id=0, scratch=torch.empty(1, dtype=torch.float16))


if __name__ == "__main__":
    unittest.main()
