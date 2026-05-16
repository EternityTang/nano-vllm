"""Regression tests for P3 Q8 budget split and rollback-safe commit."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from nanovllm.engine.arkv_kv_manager import (
    ARKVKVManager,
    MetadataCommitError,
    compute_kv_cache_budget,
    full_block_nbytes,
)
from nanovllm.engine.kv_meta import KVBlockState, PhysicalBlockTable, SequenceKVRefTable, register_full_block
from nanovllm.engine.quant_cache import QuantCache, QuantCacheSpec
from nanovllm.engine.visible_tables import VisibleBlockTable, VisibleTableConfig, build_visible_block_table


def fake_config():
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            num_key_value_heads=2,
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=16,
            head_dim=8,
            dtype=torch.float16,
        ),
        tensor_parallel_size=1,
        kvcache_block_size=256,
        total_kv_budget_bytes=0,
        kv_q8_scratch_blocks=1,
        kv_metadata_budget_bytes=4096,
        kv_q8_quant_pool_fraction=0.25,
        min_full_kvcache_blocks=2,
    )


def make_manager():
    torch.manual_seed(2)
    full_cache = torch.randn(2, 2, 4, 256, 2, 8, dtype=torch.float16)
    quant_cache = QuantCache(
        QuantCacheSpec(num_quant_blocks=2, num_layers=2, block_size=256, num_kv_heads=2, head_dim=8, dtype=torch.float16)
    )
    physical = PhysicalBlockTable()
    refs = SequenceKVRefTable()
    storage_id = register_full_block(physical, refs, 1, 0, 1, 0, 256, prefix_hash=None, is_shared_prefix=False)
    visible = VisibleBlockTable()
    visible.add_entries(1, build_visible_block_table(1, refs.refs_for_seq(1), physical, VisibleTableConfig()))
    manager = ARKVKVManager(full_cache, quant_cache, physical, refs, visible)
    return manager, physical, visible, storage_id


class QuantCommitTest(unittest.TestCase):
    def test_budget_split_carves_quant_pool_from_total(self):
        cfg = fake_config()
        total = full_block_nbytes(cfg) * 64

        budget = compute_kv_cache_budget(cfg, total)

        self.assertGreaterEqual(budget.full_pool_blocks, cfg.min_full_kvcache_blocks)
        self.assertGreater(budget.quant_pool_blocks, 0)
        self.assertLessEqual(
            budget.full_pool_bytes
            + budget.quant_pool_bytes
            + budget.scale_bytes
            + budget.scratch_budget
            + budget.metadata_budget,
            budget.total_kv_budget_bytes,
        )
        self.assertLess(budget.full_pool_blocks, total // budget.full_block_bytes)

    def test_quantize_from_full_commits_metadata_but_retains_full_by_default(self):
        manager, physical, visible, storage_id = make_manager()

        result = manager.quantize_from_full(storage_id, reason="test", step=3, allow_release_full=False)

        meta = physical.get(storage_id)
        self.assertEqual(meta.state, KVBlockState.QUANT)
        self.assertEqual(meta.quant_block_id, result.quant_block_id)
        self.assertEqual(meta.full_block_id, 1)
        self.assertIsNone(result.released_full_block_id)
        self.assertEqual(result.reclaimed_full_equiv_blocks, 0)
        self.assertEqual(visible.entries_for_seq(1)[0].state, KVBlockState.QUANT)
        self.assertEqual(len(manager.quant_cache.used_quant_block_ids), 1)

    def test_release_full_is_hard_gated_by_mixed_read_availability(self):
        manager, physical, _visible, storage_id = make_manager()

        result = manager.quantize_from_full(storage_id, reason="test", step=0, allow_release_full=True)

        self.assertIsNone(result.released_full_block_id)
        self.assertEqual(physical.get(storage_id).full_block_id, 1)

    def test_rollback_after_write_keeps_full_and_frees_quant_slot(self):
        manager, physical, visible, storage_id = make_manager()

        with self.assertRaises(Exception):
            manager.quantize_from_full(storage_id, reason="test", step=0, allow_release_full=False, fail_at="after_write")

        meta = physical.get(storage_id)
        self.assertEqual(meta.state, KVBlockState.FULL)
        self.assertEqual(meta.full_block_id, 1)
        self.assertIsNone(meta.quant_block_id)
        self.assertEqual(len(manager.quant_cache.used_quant_block_ids), 0)
        self.assertEqual(visible.entries_for_seq(1)[0].state, KVBlockState.FULL)

    def test_rollback_after_metadata_restores_visible_table(self):
        manager, physical, visible, storage_id = make_manager()

        with self.assertRaises(MetadataCommitError):
            manager.quantize_from_full(storage_id, reason="test", step=0, allow_release_full=False, fail_at="after_metadata")

        meta = physical.get(storage_id)
        self.assertEqual(meta.state, KVBlockState.FULL)
        self.assertEqual(meta.full_block_id, 1)
        self.assertIsNone(meta.quant_block_id)
        self.assertEqual(len(manager.quant_cache.used_quant_block_ids), 0)
        self.assertEqual(visible.entries_for_seq(1)[0].state, KVBlockState.FULL)


if __name__ == "__main__":
    unittest.main()
