"""P4a serving-loop activation test for real FULL->QUANT reclaim reads."""

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
from nanovllm.layers.mixed_kv_fallback import AttentionMetadata, run_decode_mixed_kv_fallback


class P4aServingActivationTest(unittest.TestCase):
    def test_forced_reclaim_activates_quant_decode_read_path(self):
        torch.manual_seed(11)
        block_size = 4
        full_cache = torch.randn(2, 1, 4, block_size, 1, 8, dtype=torch.float16)
        full_k_reference = full_cache[0, 0].clone()
        full_v_reference = full_cache[1, 0].clone()
        quant_cache = QuantCache(QuantCacheSpec(4, 1, block_size, 1, 8, torch.float16))
        block_manager = BlockManager(num_blocks=4, block_size=block_size, bytes_per_block=128)
        seq = Sequence(list(range(9)))
        seq.status = SequenceStatus.RUNNING
        seq.num_cached_tokens = len(seq)
        seq.block_table = [block_manager._allocate_block() for _ in range(3)]
        seq.num_scheduled_tokens = 0

        engine = LLMEngine.__new__(LLMEngine)
        engine.config = SimpleNamespace(kvcache_block_size=block_size)
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
            release_full_callback=engine._release_full_block_after_quant,
        )

        free_before = len(block_manager.free_block_ids)
        self.assertEqual(len(block_manager.used_block_ids), 3)

        engine._prepare_decode_visible_entries([seq])

        entries = list(seq.visible_entries)
        quant_entries = [entry for entry in entries if entry.state == KVBlockState.QUANT]
        self.assertEqual(engine.arkv_metrics["reclaim_trigger_count"], 1)
        self.assertEqual(engine.arkv_metrics["quant_commits_success"], 1)
        self.assertEqual(engine.arkv_metrics["quant_commits_rollback"], 0)
        self.assertEqual(engine.arkv_metrics["full_blocks_released_after_quant"], 1)
        self.assertGreater(len(quant_cache.used_quant_block_ids), 0)
        self.assertGreater(len(quant_entries), 0)
        self.assertTrue(all(entry.state != KVBlockState.EVICT for entry in entries))
        self.assertGreater(len(block_manager.free_block_ids), free_before)

        full_cache[:, :, quant_entries[0].full_block_id or 1].fill_(1234)
        q = torch.randn(1, 1, 8, dtype=torch.float16)
        workspace = torch.empty(1, 2, block_size, 1, 8, dtype=torch.float16)
        actual = run_decode_mixed_kv_fallback(
            q,
            [entries],
            full_cache[0, 0],
            full_cache[1, 0],
            quant_cache,
            workspace,
            AttentionMetadata(softmax_scale=8**-0.5),
        )
        full_entries = [
            entry.__class__(
                seq_id=entry.seq_id,
                logical_block_id=entry.logical_block_id,
                storage_id=entry.storage_id,
                state=KVBlockState.FULL,
                full_block_id=entry.logical_block_id,
                quant_block_id=None,
                logical_start=entry.logical_start,
                logical_end=entry.logical_end,
                visible_start=entry.visible_start,
                visible_end=entry.visible_end,
            )
            for entry in entries
        ]
        expected = run_decode_mixed_kv_fallback(
            q,
            [full_entries],
            full_k_reference,
            full_v_reference,
            quant_cache,
            workspace,
            AttentionMetadata(softmax_scale=8**-0.5),
        )
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        actual_logits = torch.stack((actual.float().sum() + 10.0, -actual.float().sum()), dim=-1)
        expected_logits = torch.stack((expected.float().sum() + 10.0, -expected.float().sum()), dim=-1)
        torch.testing.assert_close(actual_logits, expected_logits, atol=3e-2, rtol=3e-2)
        self.assertEqual(actual_logits.argmax(dim=-1).item(), expected_logits.argmax(dim=-1).item())

        engine.model_runner.runtime_metrics["mixed_kv_quant_reads"] = len(quant_entries)
        engine._publish_arkv_metrics()
        metric = block_manager.collect_metrics(step=0)
        self.assertGreater(metric.active_quant_blocks, 0)
        self.assertGreater(metric.quantized_block_ratio, 0.0)
        self.assertEqual(metric.reclaim_trigger_count, 1)
        self.assertEqual(metric.quant_commits_success, 1)
        self.assertEqual(metric.quant_commits_rollback, 0)
        self.assertEqual(metric.full_blocks_released_after_quant, 1)
        self.assertGreater(metric.mixed_kv_quant_reads, 0)
        self.assertGreater(metric.visible_quant_entries, 0)
        self.assertGreater(metric.free_full_blocks_reclaim_delta, 0)
        self.assertEqual(metric.evicted_blocks, 0)


if __name__ == "__main__":
    unittest.main()
