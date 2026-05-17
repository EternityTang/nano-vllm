"""P6a Triton gather/dequant parity and fallback tests."""

from __future__ import annotations

import unittest

import torch

from nanovllm.engine.kv_meta import KVBlockState
from nanovllm.engine.quant_cache import QuantCache, QuantCacheSpec
from nanovllm.engine.visible_tables import VisibleBlockEntry
from nanovllm.kernels.triton_gather_dequant import (
    gather_dequant_reference,
    gather_dequant_triton,
    triton_gather_dequant_supported,
)
from nanovllm.layers.mixed_kv_fallback import FullKVCache, materialize_visible_kv_for_decode


def quant_entry(logical_block_id: int, quant_block_id: int, block_size: int) -> VisibleBlockEntry:
    return VisibleBlockEntry(
        seq_id=1,
        logical_block_id=logical_block_id,
        storage_id=100 + logical_block_id,
        state=KVBlockState.QUANT,
        full_block_id=None,
        quant_block_id=quant_block_id,
        logical_start=logical_block_id * block_size,
        logical_end=(logical_block_id + 1) * block_size,
        visible_start=logical_block_id * block_size,
        visible_end=(logical_block_id + 1) * block_size,
    )


def make_quant_cache(device: str = "cpu", block_size: int = 16, head_dim: int = 8) -> tuple[QuantCache, list[VisibleBlockEntry]]:
    torch.manual_seed(31)
    spec = QuantCacheSpec(
        num_quant_blocks=4,
        num_layers=2,
        block_size=block_size,
        num_kv_heads=2,
        head_dim=head_dim,
        dtype=torch.float16,
        device=device,
    )
    cache = QuantCache(spec)
    entries = []
    for logical_block_id in range(2):
        quant_id = cache.allocate()
        full = torch.randn(spec.block_shape, dtype=torch.float16, device=device)
        cache.write_from_full(quant_id, full)
        entries.append(quant_entry(logical_block_id, quant_id, block_size))
    return cache, entries


class GatherDequantTest(unittest.TestCase):
    def test_reference_gathers_visible_quant_entries_to_output(self):
        cache, entries = make_quant_cache()
        output = torch.empty(len(entries), 2, cache.spec.block_size, cache.spec.num_kv_heads, cache.spec.head_dim)

        actual = gather_dequant_reference(cache, entries, output, layer_id=1)
        expected = cache.dequantize_to_scratch(
            [entry.quant_block_id for entry in entries],
            layer_id=1,
            scratch=torch.empty_like(output),
        )

        self.assertEqual(actual.data_ptr(), output.data_ptr())
        torch.testing.assert_close(actual, expected)

    def test_unsupported_cpu_shape_reports_false_and_runtime_falls_back(self):
        cache, entries = make_quant_cache()
        full_cache = torch.zeros(2, 16, 2, 8, dtype=torch.float16)

        self.assertFalse(triton_gather_dequant_supported(8, torch.float16, 16, torch.device("cpu")))
        expected = materialize_visible_kv_for_decode(
            entries,
            FullKVCache(full_cache, full_cache, layer_id=0),
            cache,
            torch.empty(2, 2, 16, 2, 8, dtype=torch.float16),
            use_triton_gather_dequant=False,
        )
        actual = materialize_visible_kv_for_decode(
            entries,
            FullKVCache(full_cache, full_cache, layer_id=0),
            cache,
            torch.empty(2, 2, 16, 2, 8, dtype=torch.float16),
            use_triton_gather_dequant=True,
        )

        torch.testing.assert_close(actual.k, expected.k)
        torch.testing.assert_close(actual.v, expected.v)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton gather/dequant parity")
    def test_triton_gather_dequant_matches_reference_on_cuda(self):
        cache, entries = make_quant_cache(device="cuda", block_size=16, head_dim=16)
        output = torch.empty(len(entries), 2, cache.spec.block_size, cache.spec.num_kv_heads, cache.spec.head_dim, device="cuda", dtype=torch.float16)
        reference = gather_dequant_reference(cache, entries, torch.empty_like(output), layer_id=1)
        ids = torch.tensor([entry.quant_block_id for entry in entries], dtype=torch.int64, device="cuda")

        gather_dequant_triton(
            cache.q_cache[:, 0, 1],
            cache.q_cache[:, 1, 1],
            cache.scales[:, 0, 1],
            cache.scales[:, 1, 1],
            ids,
            output[:, 0],
            output[:, 1],
            block_size=cache.spec.block_size,
            head_dim=cache.spec.head_dim,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
