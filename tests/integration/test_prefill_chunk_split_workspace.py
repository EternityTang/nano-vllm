"""P4b prefill workspace planning and split tests."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.kv_meta import KVBlockState
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.tasks import BatchPlan, PrefillTask
from nanovllm.engine.visible_tables import VisibleBlockEntry, VisibleBlockTable
from nanovllm.layers.mixed_kv_fallback import (
    MixedKVWorkspacePlan,
    WorkspacePlanningError,
    plan_prefill_mixed_kv_workspace,
    split_prefill_for_workspace,
)


def visible_entry(seq_id, logical_block_id, state, start, end):
    return VisibleBlockEntry(
        seq_id=seq_id,
        logical_block_id=logical_block_id,
        storage_id=seq_id * 10 + logical_block_id,
        state=state,
        full_block_id=logical_block_id if state == KVBlockState.FULL else None,
        quant_block_id=logical_block_id if state == KVBlockState.QUANT else None,
        logical_start=start,
        logical_end=end,
        visible_start=start,
        visible_end=end,
    )


class PrefillChunkSplitWorkspaceTest(unittest.TestCase):
    def test_prefill_workspace_counts_quant_prefix_blocks(self):
        table = VisibleBlockTable()
        table.add_entries(
            1,
            [
                visible_entry(1, 0, KVBlockState.QUANT, 0, 4),
                visible_entry(1, 1, KVBlockState.FULL, 4, 6),
            ],
        )
        batch = BatchPlan(
            batch_id=0,
            kind="prefill",
            decode_tasks=[],
            prefill_tasks=[PrefillTask(1, "1", start_pos=4, chunk_tokens=2, is_long_prefill=False, skip_count=0)],
            token_budget=2,
        )
        cfg = SimpleNamespace(block_size=4, num_kv_heads=2, head_dim=8, dtype=torch.float16)

        plan = plan_prefill_mixed_kv_workspace(batch, table, cfg)

        self.assertEqual(plan.batch_size, 1)
        self.assertEqual(plan.max_quant_blocks_per_seq, 1)
        self.assertEqual(plan.total_quant_blocks, 1)
        self.assertEqual(plan.scratch_shape, (1, 2, 4, 2, 8))

    def test_prefill_workspace_overflow_is_rejected_before_runtime(self):
        table = VisibleBlockTable()
        table.add_entries(1, [visible_entry(1, 0, KVBlockState.QUANT, 0, 4)])
        batch = BatchPlan(
            batch_id=0,
            kind="prefill",
            decode_tasks=[],
            prefill_tasks=[PrefillTask(1, "1", start_pos=0, chunk_tokens=4, is_long_prefill=False, skip_count=0)],
            token_budget=4,
        )
        cfg = SimpleNamespace(block_size=4, num_kv_heads=2, head_dim=8, dtype=torch.float16, scratch_kv_budget_bytes=1)

        with self.assertRaises(WorkspacePlanningError):
            plan_prefill_mixed_kv_workspace(batch, table, cfg)

    def test_split_prefill_for_workspace_keeps_full_coverage(self):
        task = PrefillTask(1, "1", start_pos=64, chunk_tokens=16, is_long_prefill=True, skip_count=2)
        plan = MixedKVWorkspacePlan(
            batch_size=1,
            max_quant_blocks_per_seq=4,
            total_quant_blocks=4,
            scratch_shape=(4, 2, 4, 2, 8),
            scratch_numel=512,
            scratch_bytes=1024,
        )
        cfg = SimpleNamespace(scratch_kv_budget_bytes=256, prefill_chunk_min_tokens=4)

        chunks = split_prefill_for_workspace(task, plan, cfg)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].start_pos, 64)
        self.assertEqual(sum(chunk.chunk_tokens for chunk in chunks), 16)
        self.assertEqual(chunks[-1].start_pos + chunks[-1].chunk_tokens, 80)

    def test_split_rejects_below_minimum_chunk(self):
        task = PrefillTask(1, "1", start_pos=0, chunk_tokens=4, is_long_prefill=False, skip_count=0)
        plan = MixedKVWorkspacePlan(1, 4, 4, (4, 2, 4, 2, 8), 512, 1024)
        cfg = SimpleNamespace(scratch_kv_budget_bytes=128, prefill_chunk_min_tokens=4)

        with self.assertRaises(WorkspacePlanningError):
            split_prefill_for_workspace(task, plan, cfg)

    def test_serving_workspace_error_splits_scheduled_prefill_chunk(self):
        seq = Sequence(list(range(1, 65)))
        seq.num_cached_tokens = 16
        seq.num_scheduled_tokens = 32
        engine = LLMEngine.__new__(LLMEngine)
        engine.config = SimpleNamespace(
            prefill_chunk_min_tokens=8,
            long_prefill_token_threshold=16,
            kv_q8_scratch_blocks=1,
        )
        engine.model_runner = SimpleNamespace(q8_scratch=torch.empty(1, 2, 4, 1, 2, dtype=torch.float16))

        did_split = engine._split_prefill_chunks_after_workspace_error([seq])

        self.assertTrue(did_split)
        self.assertLess(seq.num_scheduled_tokens, 32)
        self.assertGreaterEqual(seq.num_scheduled_tokens, 8)


if __name__ == "__main__":
    unittest.main()
