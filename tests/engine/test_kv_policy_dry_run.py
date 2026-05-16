# 中文说明：
# P2 reclaim policy dry-run 回归测试，验证 policy 输出确定性、输入 snapshot 不变异、保护 sink/shared/recent/inflight-write，并禁止 P2 规划 EVICT。
# 该测试为 P3 quant shadow 提供安全前提：只有非保护 FULL block 才能成为保守 reclaim 候选。
"""Regression tests for deterministic non-mutating P2 reclaim dry-run policy."""

from __future__ import annotations

from copy import deepcopy
import unittest

from nanovllm.engine.kv_meta import PhysicalBlockTable, SequenceKVRef, SequenceKVRefTable, add_owner_ref, register_full_block
from nanovllm.engine.kv_policy import (
    PolicyConfig,
    PolicyError,
    ReclaimPolicyName,
    build_policy_snapshot,
    plan_reclaim_dry_run,
)


class KVPolicyDryRunTest(unittest.TestCase):
    def make_tables(self):
        physical = PhysicalBlockTable()
        refs = SequenceKVRefTable()
        register_full_block(physical, refs, 1, 0, 10, 0, 256, prefix_hash=1, is_shared_prefix=False)
        shared_id = register_full_block(physical, refs, 1, 1, 11, 256, 512, prefix_hash=2, is_shared_prefix=False)
        physical.get(shared_id).is_shared_prefix = True
        add_owner_ref(physical, refs, shared_id, 2, 1)
        register_full_block(physical, refs, 1, 2, 12, 512, 768, prefix_hash=3, is_shared_prefix=False)
        recent_id = register_full_block(physical, refs, 1, 3, 13, 768, 1024, prefix_hash=4, is_shared_prefix=False)
        recent = refs.get(1, 3)
        refs.replace_ref(
            SequenceKVRef(
                seq_id=recent.seq_id,
                logical_block_id=recent.logical_block_id,
                storage_id=recent_id,
                logical_start=recent.logical_start,
                logical_end=recent.logical_end,
                is_recent=True,
            ),
            physical,
        )
        return physical, refs

    def test_reclaim_plan_is_deterministic_and_non_mutating(self):
        physical, refs = self.make_tables()
        snapshot = build_policy_snapshot(physical, refs, total_full_blocks=8, free_full_blocks=0)
        before = deepcopy(snapshot)

        first = plan_reclaim_dry_run(snapshot, 1, ReclaimPolicyName.ARKV_Q8_DRY_RUN, PolicyConfig())
        second = plan_reclaim_dry_run(snapshot, 1, "arkv_q8_dry_run", PolicyConfig())

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(snapshot, before)
        self.assertEqual(first.selected_storage_ids, (2,))
        self.assertEqual(first.conservative_reclaimable_blocks, 1)
        self.assertEqual(first.protected_blocks, 3)

    def test_policy_never_plans_evict_in_p2(self):
        physical, refs = self.make_tables()
        snapshot = build_policy_snapshot(physical, refs, total_full_blocks=8, free_full_blocks=0)

        with self.assertRaises(PolicyError):
            plan_reclaim_dry_run(snapshot, 1, ReclaimPolicyName.ARKV_Q8_DRY_RUN, PolicyConfig(allow_evict=True))

    def test_free_blocks_can_satisfy_without_selecting_candidates(self):
        physical, refs = self.make_tables()
        snapshot = build_policy_snapshot(physical, refs, total_full_blocks=8, free_full_blocks=2)

        plan = plan_reclaim_dry_run(snapshot, 1, ReclaimPolicyName.ARKV_Q8_DRY_RUN, PolicyConfig())

        self.assertTrue(plan.would_satisfy)
        self.assertEqual(plan.selected_storage_ids, ())


if __name__ == "__main__":
    unittest.main()
