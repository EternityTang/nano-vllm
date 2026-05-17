"""P6b decode-only mixed-KV attention kernel parity and fallback tests."""

from __future__ import annotations

import unittest

import torch

from nanovllm.engine.kv_meta import KVBlockState
from nanovllm.engine.quant_cache import QuantCache, QuantCacheSpec
from nanovllm.engine.visible_tables import VisibleBlockEntry
from nanovllm.kernels.mixed_kv_decode_attention import (
    FULL_STATE,
    QUANT_STATE,
    mixed_kv_decode_attention,
    mixed_kv_decode_attention_reference,
    mixed_kv_decode_attention_supported,
    visible_entries_to_tensor,
)
from nanovllm.layers.mixed_kv_fallback import AttentionMetadata, run_decode_mixed_kv_fallback


def visible_entry(
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


def make_runtime_case(device: str = "cpu"):
    torch.manual_seed(41)
    block_size = 16
    num_kv_heads = 2
    head_dim = 8
    full_k = torch.randn(2, block_size, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    full_v = torch.randn_like(full_k)
    quant_cache = QuantCache(QuantCacheSpec(2, 1, block_size, num_kv_heads, head_dim, torch.float16, device=device))
    quant_id = quant_cache.allocate()
    full_block = torch.stack([full_k[1], full_v[1]], dim=0).unsqueeze(1)
    quant_cache.write_from_full(quant_id, full_block)
    entries = [
        visible_entry(0, block_size, state=KVBlockState.FULL, full_block_id=0),
        visible_entry(1, block_size, state=KVBlockState.QUANT, quant_block_id=quant_id),
    ]
    q = torch.randn(1, 4, head_dim, dtype=torch.float16, device=device)
    workspace = torch.empty(1, 2, block_size, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    return q, [entries], full_k, full_v, quant_cache, workspace


class MixedKVDecodeAttentionTest(unittest.TestCase):
    def test_unsupported_cpu_shape_reports_false_and_runtime_falls_back(self):
        q, entries, full_k, full_v, quant_cache, workspace = make_runtime_case()

        self.assertFalse(
            mixed_kv_decode_attention_supported(
                head_dim=q.shape[-1],
                dtype=q.dtype,
                block_size=full_k.shape[1],
                device=q.device,
                num_q_heads=q.shape[1],
                num_kv_heads=full_k.shape[2],
            )
        )
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

    def test_torch_reference_reads_full_and_quant_visible_entries(self):
        torch.manual_seed(42)
        q = torch.randn(1, 4, 8, dtype=torch.float16)
        full_k = torch.randn(2, 16, 2, 8, dtype=torch.float16)
        full_v = torch.randn_like(full_k)
        quant_k = torch.randint(-64, 63, (2, 16, 2, 8), dtype=torch.int8)
        quant_v = torch.randint(-64, 63, (2, 16, 2, 8), dtype=torch.int8)
        k_scale = torch.rand(2, 16, 2, 1, dtype=torch.float32) * 0.02
        v_scale = torch.rand_like(k_scale) * 0.02
        visible = torch.tensor([[[FULL_STATE, 0, -1, 0, 16], [QUANT_STATE, -1, 1, 16, 32]]], dtype=torch.int64)
        counts = torch.tensor([2], dtype=torch.int32)

        out = mixed_kv_decode_attention_reference(
            q,
            full_k,
            full_v,
            quant_k,
            quant_v,
            k_scale,
            v_scale,
            visible,
            counts,
            softmax_scale=0.125,
        )

        self.assertEqual(tuple(out.shape), tuple(q.shape))
        self.assertTrue(torch.isfinite(out).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for mixed-KV decode attention parity")
    def test_triton_kernel_matches_torch_reference_on_cuda(self):
        q, entries, full_k, full_v, quant_cache, _ = make_runtime_case(device="cuda")
        entries_tensor, entry_counts = visible_entries_to_tensor(entries, device="cuda")
        layer_id = 0
        reference = mixed_kv_decode_attention_reference(
            q,
            full_k,
            full_v,
            quant_cache.q_cache[:, 0, layer_id],
            quant_cache.q_cache[:, 1, layer_id],
            quant_cache.scales[:, 0, layer_id],
            quant_cache.scales[:, 1, layer_id],
            entries_tensor,
            entry_counts,
            softmax_scale=0.125,
        )

        actual = mixed_kv_decode_attention(
            q,
            full_k,
            full_v,
            quant_cache.q_cache[:, 0, layer_id],
            quant_cache.q_cache[:, 1, layer_id],
            quant_cache.scales[:, 0, layer_id],
            quant_cache.scales[:, 1, layer_id],
            entries_tensor,
            entry_counts,
            softmax_scale=0.125,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, reference, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
