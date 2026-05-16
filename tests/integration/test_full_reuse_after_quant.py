"""P4a full-block reuse safety after successful Q8 commit."""

from __future__ import annotations

import unittest

import torch

from nanovllm.engine.arkv_kv_manager import ARKVKVManager
from nanovllm.engine.kv_meta import KVBlockState, PhysicalBlockTable, SequenceKVRefTable, register_full_block
from nanovllm.engine.quant_cache import QuantCache, QuantCacheSpec
from nanovllm.engine.visible_tables import VisibleBlockTable, VisibleTableConfig, build_visible_block_table
from nanovllm.layers.mixed_kv_fallback import FullKVCache, materialize_visible_kv_for_decode


class FullReuseAfterQuantTest(unittest.TestCase):
    def test_reused_full_block_does_not_affect_quant_visible_read(self):
        torch.manual_seed(8)
        block_size = 4
        full_cache = torch.randn(2, 1, 2, block_size, 1, 4, dtype=torch.float16)
        original_k = full_cache[0, 0, 0].clone()
        original_v = full_cache[1, 0, 0].clone()
        quant_cache = QuantCache(QuantCacheSpec(1, 1, block_size, 1, 4, torch.float16))
        physical = PhysicalBlockTable()
        refs = SequenceKVRefTable()
        storage_id = register_full_block(physical, refs, 1, 0, 0, 0, block_size, None, False)
        visible = VisibleBlockTable()
        visible.add_entries(1, build_visible_block_table(1, refs.refs_for_seq(1), physical, VisibleTableConfig()))
        released = []
        manager = ARKVKVManager(
            full_cache,
            quant_cache,
            physical,
            refs,
            visible,
            mixed_kv_read_available=True,
            release_full_callback=released.append,
        )

        result = manager.quantize_from_full(storage_id, reason="p4a-test", step=1, allow_release_full=True)

        self.assertEqual(result.released_full_block_id, 0)
        self.assertEqual(released, [0])
        self.assertEqual(physical.get(storage_id).state, KVBlockState.QUANT)
        self.assertIsNone(physical.get(storage_id).full_block_id)
        full_cache[:, :, 0].fill_(1234)

        materialized = materialize_visible_kv_for_decode(
            visible.entries_for_seq(1),
            FullKVCache(full_cache[0, 0], full_cache[1, 0], layer_id=0),
            quant_cache,
            torch.empty(1, 2, block_size, 1, 4, dtype=torch.float16),
        )

        torch.testing.assert_close(materialized.k, original_k, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(materialized.v, original_v, atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    unittest.main()
