"""P5 serving-loop activation test for quality-gated QUANT->EVICT."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
import unittest

import torch

from nanovllm.engine.arkv_kv_manager import ARKVKVManager
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.kv_meta import KVBlockState, PhysicalBlockTable, SequenceKVRefTable
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.quant_cache import QuantCache, QuantCacheSpec
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.visible_tables import VisibleBlockTable


class P5EvictServingActivationTest(unittest.TestCase):
    def test_decode_reclaim_can_commit_quant_to_evict_only_after_gate(self):
        torch.manual_seed(23)
        block_size = 4
        full_cache = torch.randn(2, 1, 4, block_size, 1, 8, dtype=torch.float16)
        quant_cache = QuantCache(QuantCacheSpec(4, 1, block_size, 1, 8, torch.float16))
        block_manager = BlockManager(num_blocks=4, block_size=block_size, bytes_per_block=128)
        seq = Sequence(list(range(9)))
        seq.status = SequenceStatus.RUNNING
        seq.num_cached_tokens = len(seq)
        seq.block_table = [block_manager._allocate_block() for _ in range(3)]
        seq.num_scheduled_tokens = 0

        engine = LLMEngine.__new__(LLMEngine)
        engine.config = SimpleNamespace(
            kvcache_block_size=block_size,
            kv_q8_scratch_blocks=1,
            enable_kv_evict=True,
            enable_quality_gate=True,
            enable_direct_full_evict=False,
        )
        engine.step_count = 0
        engine.scheduler = SimpleNamespace(block_manager=block_manager, running=deque([seq]), waiting=deque())
        engine.model_runner = SimpleNamespace(
            kv_cache=full_cache,
            quant_cache=quant_cache,
            runtime_metrics={"mixed_kv_quant_reads": 0, "visible_quant_entries": 0},
        )
        engine.arkv_runtime_enabled = True
        engine.physical_table = PhysicalBlockTable()
        engine.ref_table = SequenceKVRefTable()
        engine.visible_table = VisibleBlockTable()
        engine._seq_logical_to_storage = {}
        engine._full_block_to_storage = {}
        engine.arkv_metrics = {
            "reclaim_trigger_count": 0,
            "quant_commits_success": 0,
            "quant_commits_rollback": 0,
            "evict_commits_success": 0,
            "evict_commits_rollback": 0,
            "full_blocks_released_after_quant": 0,
            "free_full_blocks_before_reclaim": 0,
            "free_full_blocks_after_reclaim": 0,
            "free_full_blocks_reclaim_delta": 0,
        }
        engine.arkv_manager = ARKVKVManager(
            full_cache,
            quant_cache,
            engine.physical_table,
            engine.ref_table,
            engine.visible_table,
            mixed_kv_read_available=True,
            enable_kv_evict=True,
            quality_gate_passed=True,
            release_full_callback=engine._release_full_block_after_quant,
        )

        engine._prepare_decode_visible_entries([seq])

        entries = list(seq.visible_entries)
        self.assertEqual(engine.arkv_metrics["quant_commits_success"], 1)
        self.assertEqual(engine.arkv_metrics["evict_commits_success"], 1)
        self.assertEqual(engine.arkv_metrics["evict_commits_rollback"], 0)
        self.assertEqual(len(quant_cache.used_quant_block_ids), 0)
        self.assertEqual(sum(1 for meta in engine.physical_table.values() if meta.state == KVBlockState.EVICT), 1)
        self.assertTrue(all(entry.state != KVBlockState.EVICT for entry in entries))
        self.assertTrue(all(entry.state == KVBlockState.FULL for entry in entries))
        self.assertEqual([entry.logical_block_id for entry in entries], [0, 2])
        self.assertEqual(engine.ref_table.get(seq.seq_id, 1).logical_end, 8)

        engine._publish_arkv_metrics()
        metric = block_manager.collect_metrics(step=0)
        self.assertEqual(metric.evicted_blocks, 1)
        self.assertEqual(metric.active_quant_blocks, 0)
        self.assertEqual(metric.quant_commits_success, 1)
        self.assertGreater(metric.free_full_blocks_reclaim_delta, 0)


if __name__ == "__main__":
    unittest.main()
