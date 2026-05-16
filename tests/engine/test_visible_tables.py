# 中文说明：
# P2 visible table 回归测试，验证 visible/read view 与 logical refs 分离、visible_context_len 不等同于 logical_context_len、缺失物理块和非连续逻辑 span 会报错。
# 这些测试锁住 P4a mixed-KV fallback 前的读视图边界，避免把 VisibleBlockTable 用作写路径或 slot_mapping。
"""Regression tests for P2 visible table invariants and context separation."""

from __future__ import annotations

import unittest

from nanovllm.engine.kv_meta import PhysicalBlockTable, SequenceKVRef, SequenceKVRefTable, register_full_block
from nanovllm.engine.visible_tables import VisibleBlockTable, VisibleTableConfig, VisibleTableError, build_visible_block_table


class VisibleTablesTest(unittest.TestCase):
    def test_visible_table_tracks_visible_context_separately_from_logical_context(self):
        physical = PhysicalBlockTable()
        refs = SequenceKVRefTable()
        register_full_block(physical, refs, 1, 0, 10, 0, 256, prefix_hash=None, is_shared_prefix=False)
        second_storage = physical.register_full_block(1, 1, 11, 256, 512, prefix_hash=None, is_shared_prefix=False)
        refs.add_ref(
            SequenceKVRef(
                seq_id=1,
                logical_block_id=1,
                storage_id=second_storage,
                logical_start=256,
                logical_end=512,
                visible=False,
            ),
            physical,
        )

        table = VisibleBlockTable()
        table.add_entries(1, build_visible_block_table(1, refs.refs_for_seq(1), physical, VisibleTableConfig()))

        self.assertEqual(table.visible_context_len(1), 256)
        self.assertEqual(refs.get(1, 1).logical_end, 512)
        self.assertEqual(len(table.entries_for_seq(1)), 1)
        self.assertFalse(hasattr(table.entries_for_seq(1)[0], "slot_mapping"))

    def test_non_monotonic_logical_span_is_rejected(self):
        physical = PhysicalBlockTable()
        refs = SequenceKVRefTable()
        first = physical.register_full_block(1, 0, 10, 0, 128, prefix_hash=None, is_shared_prefix=False)
        second = physical.register_full_block(1, 1, 11, 256, 512, prefix_hash=None, is_shared_prefix=False)

        with self.assertRaises(VisibleTableError):
            build_visible_block_table(
                1,
                [
                    SequenceKVRef(1, 0, first, 0, 128),
                    SequenceKVRef(1, 1, second, 256, 512),
                ],
                physical,
                VisibleTableConfig(),
            )

    def test_missing_physical_block_is_rejected(self):
        physical = PhysicalBlockTable()
        with self.assertRaises(VisibleTableError):
            build_visible_block_table(
                1,
                [SequenceKVRef(1, 0, storage_id=99, logical_start=0, logical_end=256)],
                physical,
                VisibleTableConfig(),
            )


if __name__ == "__main__":
    unittest.main()
