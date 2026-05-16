"""P4a decode mixed-KV fallback parity tests."""

from __future__ import annotations

import unittest

import torch

from nanovllm.engine.kv_meta import KVBlockState
from nanovllm.engine.quant_cache import QuantCache, QuantCacheSpec
from nanovllm.engine.visible_tables import VisibleBlockEntry
from nanovllm.layers.mixed_kv_fallback import AttentionMetadata, run_decode_mixed_kv_fallback


def entry(logical_block_id, state, full_block_id, quant_block_id, start, end):
    return VisibleBlockEntry(
        seq_id=1,
        logical_block_id=logical_block_id,
        storage_id=logical_block_id,
        state=state,
        full_block_id=full_block_id,
        quant_block_id=quant_block_id,
        logical_start=start,
        logical_end=end,
        visible_start=start,
        visible_end=end,
    )


class DecodeMixedKVFallbackTest(unittest.TestCase):
    def test_mixed_full_quant_decode_matches_full_only_reference(self):
        torch.manual_seed(4)
        block_size = 4
        num_blocks = 3
        num_kv_heads = 2
        head_dim = 8
        dtype = torch.float16
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        quant_cache = QuantCache(
            QuantCacheSpec(
                num_quant_blocks=1,
                num_layers=1,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
            )
        )
        quant_block_id = quant_cache.allocate()
        quant_cache.write_from_full(quant_block_id, torch.stack((k_cache[1], v_cache[1])).unsqueeze(1))
        workspace = torch.empty(1, 2, block_size, num_kv_heads, head_dim, dtype=dtype)
        q = torch.randn(1, num_kv_heads, head_dim, dtype=dtype)

        full_entries = [
            entry(0, KVBlockState.FULL, 0, None, 0, 4),
            entry(1, KVBlockState.FULL, 1, None, 4, 8),
            entry(2, KVBlockState.FULL, 2, None, 8, 10),
        ]
        mixed_entries = [
            entry(0, KVBlockState.FULL, 0, None, 0, 4),
            entry(1, KVBlockState.QUANT, None, quant_block_id, 4, 8),
            entry(2, KVBlockState.FULL, 2, None, 8, 10),
        ]

        expected = run_decode_mixed_kv_fallback(
            q,
            [full_entries],
            k_cache,
            v_cache,
            quant_cache,
            workspace,
            AttentionMetadata(layer_id=0, softmax_scale=head_dim**-0.5),
        )
        actual = run_decode_mixed_kv_fallback(
            q,
            [mixed_entries],
            k_cache,
            v_cache,
            quant_cache,
            workspace,
            AttentionMetadata(layer_id=0, softmax_scale=head_dim**-0.5),
        )

        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    def test_scratch_overflow_is_rejected(self):
        block_size = 4
        k_cache = torch.zeros(1, block_size, 1, 2, dtype=torch.float16)
        v_cache = torch.zeros_like(k_cache)
        quant_cache = QuantCache(QuantCacheSpec(1, 1, block_size, 1, 2, torch.float16))
        quant_block_id = quant_cache.allocate()
        quant_cache.write_from_full(quant_block_id, torch.stack((k_cache[0], v_cache[0])).unsqueeze(1))
        q = torch.zeros(1, 1, 2, dtype=torch.float16)

        with self.assertRaises(Exception):
            run_decode_mixed_kv_fallback(
                q,
                [[entry(0, KVBlockState.QUANT, None, quant_block_id, 0, 4)]],
                k_cache,
                v_cache,
                quant_cache,
                torch.empty(1, dtype=torch.float16),
                AttentionMetadata(),
            )


if __name__ == "__main__":
    unittest.main()
