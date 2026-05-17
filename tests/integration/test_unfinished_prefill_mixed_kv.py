"""P4b unfinished-prefill mixed-KV safety tests."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from nanovllm.engine.kv_meta import KVBlockState
from nanovllm.engine.kv_policy import ReclaimCandidate, ReclaimPlan
from nanovllm.engine.quant_cache import QuantCache, QuantCacheSpec
from nanovllm.engine.visible_tables import VisibleBlockEntry
from nanovllm.layers.mixed_kv_fallback import (
    AttentionMetadata,
    InvalidWriteTargetError,
    PolicyInvariantError,
    run_prefill_mixed_kv_fallback,
    validate_unfinished_prefill_policy,
)


def entry(logical_block_id, state, full_block_id, quant_block_id, start, end):
    return VisibleBlockEntry(
        seq_id=3,
        logical_block_id=logical_block_id,
        storage_id=logical_block_id,
        state=state,
        full_block_id=full_block_id,
        quant_block_id=quant_block_id,
        logical_start=start,
        logical_end=end,
        visible_start=start,
        visible_end=end,
    )


class UnfinishedPrefillMixedKVTest(unittest.TestCase):
    def test_unfinished_prefill_reads_quant_prefix(self):
        torch.manual_seed(22)
        block_size = 4
        num_kv_heads = 1
        head_dim = 8
        dtype = torch.float16
        k_cache = torch.randn(2, block_size, num_kv_heads, head_dim, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        quant_cache = QuantCache(QuantCacheSpec(1, 1, block_size, num_kv_heads, head_dim, dtype))
        quant_block_id = quant_cache.allocate()
        quant_cache.write_from_full(quant_block_id, torch.stack((k_cache[0], v_cache[0])).unsqueeze(1))
        workspace = torch.empty(1, 2, block_size, num_kv_heads, head_dim, dtype=dtype)
        q = torch.randn(2, num_kv_heads, head_dim, dtype=dtype)
        slot_mapping = torch.tensor([block_size, block_size + 1], dtype=torch.int32)
        full_entries = [
            entry(0, KVBlockState.FULL, 0, None, 0, 4),
            entry(1, KVBlockState.FULL, 1, None, 4, 6),
        ]
        mixed_entries = [
            entry(0, KVBlockState.QUANT, None, quant_block_id, 0, 4),
            entry(1, KVBlockState.FULL, 1, None, 4, 6),
        ]
        metadata = AttentionMetadata(query_lengths=(2,), query_start_positions=(4,), softmax_scale=head_dim**-0.5)

        expected = run_prefill_mixed_kv_fallback(
            q, [full_entries], slot_mapping, k_cache, v_cache, quant_cache, workspace, metadata
        )
        actual = run_prefill_mixed_kv_fallback(
            q, [mixed_entries], slot_mapping, k_cache, v_cache, quant_cache, workspace, metadata
        )

        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    def test_slot_mapping_must_write_full_blocks_only(self):
        block_size = 4
        k_cache = torch.zeros(2, block_size, 1, 2, dtype=torch.float16)
        v_cache = torch.zeros_like(k_cache)
        quant_cache = QuantCache(QuantCacheSpec(1, 1, block_size, 1, 2, torch.float16))
        quant_block_id = quant_cache.allocate()
        quant_cache.write_from_full(quant_block_id, torch.stack((k_cache[0], v_cache[0])).unsqueeze(1))
        entries = [
            entry(0, KVBlockState.QUANT, None, quant_block_id, 0, 4),
            entry(1, KVBlockState.FULL, 1, None, 4, 6),
        ]
        metadata = AttentionMetadata(query_lengths=(1,), query_start_positions=(4,))

        with self.assertRaises(InvalidWriteTargetError):
            run_prefill_mixed_kv_fallback(
                torch.zeros(1, 1, 2, dtype=torch.float16),
                [entries],
                torch.tensor([0], dtype=torch.int32),
                k_cache,
                v_cache,
                quant_cache,
                torch.empty(1, 2, block_size, 1, 2, dtype=torch.float16),
                metadata,
            )

    def test_unfinished_prefill_rejects_evict_candidates(self):
        seq_state = SimpleNamespace(is_prefill=True, is_finished=False, num_cached_tokens=4, num_tokens=10)
        plan = ReclaimPlan(
            policy_name="test",
            required_full_equiv=1,
            candidates=(
                ReclaimCandidate(
                    storage_id=7,
                    state=KVBlockState.EVICT,
                    score=0.0,
                    reclaimable=False,
                    protected_reasons=(),
                ),
            ),
            selected_storage_ids=(),
            conservative_reclaimable_blocks=0,
            protected_blocks=0,
            protected_ratio=0.0,
            would_satisfy=False,
        )

        with self.assertRaises(PolicyInvariantError):
            validate_unfinished_prefill_policy(seq_state, plan)


if __name__ == "__main__":
    unittest.main()
