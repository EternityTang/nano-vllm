"""P7 release gate: kernel and mixed-KV fallbacks stay available."""

from __future__ import annotations

import unittest

import torch

from nanovllm.engine.kv_meta import KVBlockState
from nanovllm.engine.quant_cache import QuantCache, QuantCacheSpec
from nanovllm.engine.visible_tables import VisibleBlockEntry
from nanovllm.layers.mixed_kv_fallback import (
    AttentionMetadata,
    FullKVCache,
    materialize_visible_kv_for_decode,
    run_decode_mixed_kv_fallback,
)


def _entry(
    logical_block_id: int,
    block_size: int,
    *,
    state: KVBlockState,
    full_block_id: int | None = None,
    quant_block_id: int | None = None,
) -> VisibleBlockEntry:
    return VisibleBlockEntry(
        seq_id=1,
        logical_block_id=logical_block_id,
        storage_id=100 + logical_block_id,
        state=state,
        full_block_id=full_block_id,
        quant_block_id=quant_block_id,
        logical_start=logical_block_id * block_size,
        logical_end=(logical_block_id + 1) * block_size,
        visible_start=logical_block_id * block_size,
        visible_end=(logical_block_id + 1) * block_size,
    )


def _case():
    torch.manual_seed(71)
    block_size = 16
    num_kv_heads = 2
    head_dim = 8
    full_k = torch.randn(2, block_size, num_kv_heads, head_dim, dtype=torch.float16)
    full_v = torch.randn_like(full_k)
    quant_cache = QuantCache(QuantCacheSpec(2, 1, block_size, num_kv_heads, head_dim, torch.float16))
    quant_id = quant_cache.allocate()
    quant_cache.write_from_full(quant_id, torch.stack([full_k[1], full_v[1]], dim=0).unsqueeze(1))
    entries = [
        _entry(0, block_size, state=KVBlockState.FULL, full_block_id=0),
        _entry(1, block_size, state=KVBlockState.QUANT, quant_block_id=quant_id),
    ]
    q = torch.randn(1, 4, head_dim, dtype=torch.float16)
    workspace = torch.empty(2, 2, block_size, num_kv_heads, head_dim, dtype=torch.float16)
    return q, [entries], full_k, full_v, quant_cache, workspace


class FallbackPathsTest(unittest.TestCase):
    def test_triton_gather_dequant_flag_falls_back_on_unsupported_cpu(self):
        _, entries, full_k, full_v, quant_cache, workspace = _case()

        expected = materialize_visible_kv_for_decode(
            entries[0],
            FullKVCache(full_k, full_v, layer_id=0),
            quant_cache,
            workspace.clone(),
            use_triton_gather_dequant=False,
        )
        actual = materialize_visible_kv_for_decode(
            entries[0],
            FullKVCache(full_k, full_v, layer_id=0),
            quant_cache,
            workspace.clone(),
            use_triton_gather_dequant=True,
        )

        torch.testing.assert_close(actual.k, expected.k)
        torch.testing.assert_close(actual.v, expected.v)

    def test_mixed_kv_decode_kernel_flag_falls_back_on_unsupported_cpu(self):
        q, entries, full_k, full_v, quant_cache, workspace = _case()

        expected = run_decode_mixed_kv_fallback(
            q,
            entries,
            full_k,
            full_v,
            quant_cache,
            workspace.clone(),
            AttentionMetadata(layer_id=0, softmax_scale=0.125),
            use_mixed_kv_decode_kernel=False,
        )
        actual = run_decode_mixed_kv_fallback(
            q,
            entries,
            full_k,
            full_v,
            quant_cache,
            workspace.clone(),
            AttentionMetadata(layer_id=0, softmax_scale=0.125),
            use_mixed_kv_decode_kernel=True,
        )

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
