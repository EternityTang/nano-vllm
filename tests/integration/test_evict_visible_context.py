"""P5 integration tests for EVICT metadata and visible read views."""

from __future__ import annotations

import unittest

import torch

from nanovllm.engine.arkv_kv_manager import ARKVKVManager, EvictNotAllowedError
from nanovllm.engine.kv_meta import KVBlockState, PhysicalBlockTable, SequenceKVRefTable, register_full_block
from nanovllm.engine.quant_cache import QuantCache, QuantCacheSpec
from nanovllm.engine.visible_tables import VisibleBlockTable, VisibleTableConfig, build_visible_block_table


class EvictVisibleContextTest(unittest.TestCase):
    def test_visible_table_skips_evicted_span_but_keeps_logical_refs(self):
        physical = PhysicalBlockTable()
        refs = SequenceKVRefTable()
        register_full_block(physical, refs, 1, 0, 10, 0, 256, None, False)
        evicted_storage = register_full_block(physical, refs, 1, 1, 11, 256, 512, None, False)
        register_full_block(physical, refs, 1, 2, 12, 512, 768, None, False)
        physical.get(evicted_storage).state = KVBlockState.EVICT
        physical.get(evicted_storage).full_block_id = None

        entries = build_visible_block_table(1, refs.refs_for_seq(1), physical, VisibleTableConfig())

        self.assertEqual([entry.logical_block_id for entry in entries], [0, 2])
        self.assertEqual(entries[0].visible_start, 0)
        self.assertEqual(entries[0].visible_end, 256)
        self.assertEqual(entries[1].visible_start, 256)
        self.assertEqual(entries[1].visible_end, 512)
        self.assertEqual(refs.get(1, 1).logical_end, 512)
        self.assertTrue(all(entry.state != KVBlockState.EVICT for entry in entries))

    def test_quant_to_evict_commit_releases_quant_and_refreshes_visible_table(self):
        torch.manual_seed(19)
        block_size = 4
        full_cache = torch.randn(2, 1, 2, block_size, 1, 4, dtype=torch.float16)
        quant_cache = QuantCache(QuantCacheSpec(1, 1, block_size, 1, 4, torch.float16))
        physical = PhysicalBlockTable()
        refs = SequenceKVRefTable()
        storage_id = register_full_block(physical, refs, 1, 0, 0, 0, block_size, None, False)
        visible = VisibleBlockTable()
        visible.add_entries(1, build_visible_block_table(1, refs.refs_for_seq(1), physical, VisibleTableConfig()))
        manager = ARKVKVManager(
            full_cache,
            quant_cache,
            physical,
            refs,
            visible,
            mixed_kv_read_available=True,
            enable_kv_evict=True,
            quality_gate_passed=True,
        )
        manager.quantize_from_full(storage_id, reason="test", step=1, allow_release_full=True)

        result = manager.apply_evict_transition(storage_id, step=2, reason="p5-test")

        self.assertEqual(result.released_quant_block_id, 0)
        self.assertEqual(len(quant_cache.used_quant_block_ids), 0)
        self.assertEqual(physical.get(storage_id).state, KVBlockState.EVICT)
        self.assertEqual(visible.entries_for_seq(1), ())
        self.assertEqual(refs.get(1, 0).logical_end, block_size)

    def test_evict_commit_fails_closed_without_quality_gate(self):
        full_cache = torch.zeros(2, 1, 1, 4, 1, 4, dtype=torch.float16)
        quant_cache = QuantCache(QuantCacheSpec(1, 1, 4, 1, 4, torch.float16))
        physical = PhysicalBlockTable()
        refs = SequenceKVRefTable()
        storage_id = register_full_block(physical, refs, 1, 0, 0, 0, 4, None, False)
        manager = ARKVKVManager(
            full_cache,
            quant_cache,
            physical,
            refs,
            enable_kv_evict=True,
            quality_gate_passed=False,
        )

        with self.assertRaises(EvictNotAllowedError):
            manager.apply_evict_transition(storage_id, step=0, reason="test")


if __name__ == "__main__":
    unittest.main()
