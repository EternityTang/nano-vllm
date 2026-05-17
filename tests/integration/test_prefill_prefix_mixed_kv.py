"""P4b prefill-prefix mixed-KV fallback tests."""

from __future__ import annotations

import unittest

import torch

from nanovllm.engine.kv_meta import KVBlockState
from nanovllm.engine.quant_cache import QuantCache, QuantCacheSpec
from nanovllm.engine.visible_tables import VisibleBlockEntry
from nanovllm.layers.mixed_kv_fallback import AttentionMetadata, run_prefill_mixed_kv_fallback


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


class PrefillPrefixMixedKVTest(unittest.TestCase):
    def test_prefill_prefix_reads_quant_and_matches_full_reference(self):
        torch.manual_seed(21)
        block_size = 4
        num_blocks = 3
        num_kv_heads = 2
        head_dim = 8
        dtype = torch.float16
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        quant_cache = QuantCache(QuantCacheSpec(1, 1, block_size, num_kv_heads, head_dim, dtype))
        quant_block_id = quant_cache.allocate()
        quant_cache.write_from_full(quant_block_id, torch.stack((k_cache[1], v_cache[1])).unsqueeze(1))
        workspace = torch.empty(1, 2, block_size, num_kv_heads, head_dim, dtype=dtype)
        q = torch.randn(2, num_kv_heads, head_dim, dtype=dtype)
        slot_mapping = torch.tensor([2 * block_size, 2 * block_size + 1], dtype=torch.int32)

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
        metadata = AttentionMetadata(
            layer_id=0,
            softmax_scale=head_dim**-0.5,
            query_lengths=(2,),
            query_start_positions=(8,),
        )

        expected = run_prefill_mixed_kv_fallback(
            q,
            [full_entries],
            slot_mapping,
            k_cache,
            v_cache,
            quant_cache,
            workspace,
            metadata,
        )
        actual = run_prefill_mixed_kv_fallback(
            q,
            [mixed_entries],
            slot_mapping,
            k_cache,
            v_cache,
            quant_cache,
            workspace,
            metadata,
        )

        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    unittest.main()
