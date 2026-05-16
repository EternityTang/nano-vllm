# 中文说明：
# P2 KV metadata 回归测试，覆盖 FULL block 注册、SequenceKVRef 建立、shared-prefix owner_refs/ref_count 和跨表 invariant 校验。
# 这些测试防止 PhysicalBlockMeta 与 SequenceKVRef 被重新合并，也防止 shared-prefix 被错误建模为单 owner。
"""Regression tests for P2 physical block metadata and logical owner refs."""

from __future__ import annotations

import unittest

from nanovllm.engine.kv_meta import (
    MetadataConsistencyError,
    PhysicalBlockTable,
    SequenceKVRefTable,
    add_owner_ref,
    register_full_block,
    validate_kv_tables,
)
from nanovllm.engine.visible_tables import VisibleBlockTable, VisibleTableConfig, build_visible_block_table


class KVMetaTest(unittest.TestCase):
    def test_register_full_block_records_physical_and_logical_refs(self):
        physical = PhysicalBlockTable()
        refs = SequenceKVRefTable()

        storage_id = register_full_block(
            physical,
            refs,
            seq_id=7,
            logical_block_id=0,
            full_block_id=3,
            logical_start=0,
            logical_end=256,
            prefix_hash=123,
            is_shared_prefix=False,
        )

        meta = physical.get(storage_id)
        self.assertEqual(meta.full_block_id, 3)
        self.assertEqual(meta.ref_count, 1)
        self.assertEqual(refs.get(7, 0).storage_id, storage_id)

    def test_shared_prefix_uses_owner_refs_and_ref_count(self):
        physical = PhysicalBlockTable()
        refs = SequenceKVRefTable()
        storage_id = register_full_block(physical, refs, 1, 0, 10, 0, 256, prefix_hash=99, is_shared_prefix=False)

        physical.get(storage_id).is_shared_prefix = True
        add_owner_ref(physical, refs, storage_id, seq_id=2, logical_block_id=0)

        self.assertEqual(physical.get(storage_id).ref_count, 2)
        self.assertEqual(len(physical.get(storage_id).copy_owner_refs()), 2)
        visible = VisibleBlockTable()
        visible.add_entries(1, build_visible_block_table(1, refs.refs_for_seq(1), physical, VisibleTableConfig()))
        visible.add_entries(2, build_visible_block_table(2, refs.refs_for_seq(2), physical, VisibleTableConfig()))
        validate_kv_tables(physical, refs, visible)

    def test_duplicate_logical_ref_is_rejected(self):
        physical = PhysicalBlockTable()
        refs = SequenceKVRefTable()
        register_full_block(physical, refs, 1, 0, 10, 0, 256, prefix_hash=None, is_shared_prefix=False)
        with self.assertRaises(MetadataConsistencyError):
            refs.add_ref(refs.get(1, 0), physical)

    def test_shared_prefix_requires_more_than_one_ref(self):
        physical = PhysicalBlockTable()
        refs = SequenceKVRefTable()
        register_full_block(physical, refs, 1, 0, 10, 0, 256, prefix_hash=None, is_shared_prefix=True)
        visible = VisibleBlockTable()
        visible.add_entries(1, build_visible_block_table(1, refs.refs_for_seq(1), physical, VisibleTableConfig()))

        with self.assertRaises(MetadataConsistencyError):
            validate_kv_tables(physical, refs, visible)


if __name__ == "__main__":
    unittest.main()
