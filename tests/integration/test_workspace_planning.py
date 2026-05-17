"""P4a mixed-KV scratch workspace planning tests."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from nanovllm.engine.kv_meta import KVBlockState
from nanovllm.engine.tasks import BatchPlan, DecodeTask
from nanovllm.engine.visible_tables import VisibleBlockEntry, VisibleBlockTable
from nanovllm.layers.mixed_kv_fallback import WorkspacePlanningError, plan_mixed_kv_workspace


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


class WorkspacePlanningTest(unittest.TestCase):
    def test_plan_counts_max_quant_blocks_per_decode_sequence(self):
        table = VisibleBlockTable()
        table.add_entries(
            1,
            [
                visible_entry(1, 0, KVBlockState.FULL, 0, 4),
                visible_entry(1, 1, KVBlockState.QUANT, 4, 8),
            ],
        )
        table.add_entries(
            2,
            [
                visible_entry(2, 0, KVBlockState.QUANT, 0, 4),
                visible_entry(2, 1, KVBlockState.QUANT, 4, 8),
            ],
        )
        batch = BatchPlan(
            batch_id=0,
            kind="decode",
            decode_tasks=[DecodeTask(1, "1"), DecodeTask(2, "2")],
            prefill_tasks=[],
            token_budget=2,
        )
        cfg = SimpleNamespace(block_size=4, num_kv_heads=2, head_dim=8, dtype=torch.float16)

        plan = plan_mixed_kv_workspace(batch, table, cfg)

        self.assertEqual(plan.batch_size, 2)
        self.assertEqual(plan.max_quant_blocks_per_seq, 2)
        self.assertEqual(plan.total_quant_blocks, 3)
        self.assertEqual(plan.scratch_shape, (2, 2, 4, 2, 8))

    def test_plan_rejects_scratch_budget_overflow(self):
        table = VisibleBlockTable()
        table.add_entries(1, [visible_entry(1, 0, KVBlockState.QUANT, 0, 4)])
        batch = BatchPlan(
            batch_id=0,
            kind="decode",
            decode_tasks=[DecodeTask(1, "1")],
            prefill_tasks=[],
            token_budget=1,
        )
        cfg = SimpleNamespace(
            block_size=4,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float16,
            scratch_kv_budget_bytes=1,
        )

        with self.assertRaises(WorkspacePlanningError):
            plan_mixed_kv_workspace(batch, table, cfg)

    def test_plan_allows_evict_logical_gap_when_visible_spans_are_contiguous(self):
        table = VisibleBlockTable()
        table.add_entries(
            1,
            [
                visible_entry(1, 0, KVBlockState.FULL, 0, 4),
                VisibleBlockEntry(
                    seq_id=1,
                    logical_block_id=2,
                    storage_id=12,
                    state=KVBlockState.FULL,
                    full_block_id=2,
                    quant_block_id=None,
                    logical_start=8,
                    logical_end=12,
                    visible_start=4,
                    visible_end=8,
                ),
            ],
        )
        batch = BatchPlan(
            batch_id=0,
            kind="decode",
            decode_tasks=[DecodeTask(1, "1")],
            prefill_tasks=[],
            token_budget=1,
        )
        cfg = SimpleNamespace(block_size=4, num_kv_heads=2, head_dim=8, dtype=torch.float16)

        plan = plan_mixed_kv_workspace(batch, table, cfg)

        self.assertEqual(plan.total_quant_blocks, 0)

    def test_prefill_batch_is_rejected(self):
        batch = BatchPlan(batch_id=0, kind="prefill", decode_tasks=[], prefill_tasks=[], token_budget=1)
        cfg = SimpleNamespace(block_size=4, num_kv_heads=2, head_dim=8, dtype=torch.float16)
        with self.assertRaises(WorkspacePlanningError):
            plan_mixed_kv_workspace(batch, VisibleBlockTable(), cfg)

    def test_missing_decode_visible_entries_are_rejected(self):
        batch = BatchPlan(
            batch_id=0,
            kind="decode",
            decode_tasks=[DecodeTask(3, "3")],
            prefill_tasks=[],
            token_budget=1,
        )
        cfg = SimpleNamespace(block_size=4, num_kv_heads=2, head_dim=8, dtype=torch.float16)
        with self.assertRaises(WorkspacePlanningError):
            plan_mixed_kv_workspace(batch, VisibleBlockTable(), cfg)


if __name__ == "__main__":
    unittest.main()
