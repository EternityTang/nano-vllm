"""P5 EVICT policy guard tests."""

from __future__ import annotations

import unittest

from nanovllm.engine.kv_meta import KVBlockState, PhysicalBlockTable, SequenceKVRef, SequenceKVRefTable, SequenceKVState
from nanovllm.engine.kv_policy import (
    PolicyConfig,
    PolicyError,
    PolicyInvariantError,
    ReclaimCandidate,
    ReclaimPolicyName,
    build_policy_snapshot,
    plan_reclaim_dry_run,
    select_blocks_to_evict,
)


def make_snapshot_with_quant():
    physical = PhysicalBlockTable()
    refs = SequenceKVRefTable()
    for logical_block_id in range(4):
        storage_id = physical.register_full_block(
            1,
            logical_block_id,
            10 + logical_block_id,
            logical_block_id * 256,
            (logical_block_id + 1) * 256,
            prefix_hash=None,
            is_shared_prefix=False,
        )
        refs.add_ref(
            SequenceKVRef(
                1,
                logical_block_id,
                storage_id,
                logical_block_id * 256,
                (logical_block_id + 1) * 256,
                is_sink=logical_block_id == 0,
                is_recent=logical_block_id == 3,
            ),
            physical,
        )
    physical.get(1).state = KVBlockState.QUANT
    physical.get(1).quant_block_id = 0
    physical.get(1).full_block_id = None
    refs.replace_ref(SequenceKVRef(1, 2, 2, 512, 768, is_inflight_write=True), physical)
    return build_policy_snapshot(physical, refs, total_full_blocks=8, free_full_blocks=0)


class EvictPolicyGuardsTest(unittest.TestCase):
    def test_evict_policy_is_locked_behind_quality_gate(self):
        snapshot = make_snapshot_with_quant()

        with self.assertRaises(PolicyError):
            plan_reclaim_dry_run(
                snapshot,
                1,
                ReclaimPolicyName.ARKV_Q8_EVICT,
                PolicyConfig(allow_evict=True, quality_gate_passed=False),
            )

    def test_dry_run_policy_never_accepts_evict_flag(self):
        snapshot = make_snapshot_with_quant()

        with self.assertRaises(PolicyError):
            plan_reclaim_dry_run(
                snapshot,
                1,
                ReclaimPolicyName.ARKV_Q8_DRY_RUN,
                PolicyConfig(allow_evict=True, quality_gate_passed=True),
            )

    def test_evict_plan_prefers_quant_and_preserves_guards(self):
        snapshot = make_snapshot_with_quant()

        plan = plan_reclaim_dry_run(
            snapshot,
            1,
            ReclaimPolicyName.ARKV_Q8_EVICT,
            PolicyConfig(allow_evict=True, quality_gate_passed=True),
        )

        self.assertEqual(plan.selected_storage_ids, (1,))
        reasons_by_id = {candidate.storage_id: candidate.protected_reasons for candidate in plan.candidates}
        self.assertIn("sink", reasons_by_id[0])
        self.assertIn("inflight_write", reasons_by_id[2])
        self.assertIn("recent", reasons_by_id[3])

    def test_select_blocks_to_evict_rejects_unfinished_prefill(self):
        candidate = ReclaimCandidate(1, KVBlockState.QUANT, 1.0, True, ())

        with self.assertRaises(PolicyInvariantError):
            select_blocks_to_evict(
                [candidate],
                1,
                {1: SequenceKVState.UNFINISHED_PREFILL},
                PolicyConfig(allow_evict=True, quality_gate_passed=True),
            )

    def test_select_blocks_to_evict_rejects_direct_full_when_disabled(self):
        candidate = ReclaimCandidate(1, KVBlockState.FULL, 1.0, True, ())

        with self.assertRaises(PolicyInvariantError):
            select_blocks_to_evict(
                [candidate],
                1,
                {},
                PolicyConfig(allow_evict=True, quality_gate_passed=True),
            )


if __name__ == "__main__":
    unittest.main()
