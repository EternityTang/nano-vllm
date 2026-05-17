"""Optional P6b attention-mass output tests."""

from __future__ import annotations

import unittest

import torch

from nanovllm.kernels.mixed_kv_decode_attention import (
    EVICT_STATE,
    FULL_STATE,
    QUANT_STATE,
    mixed_kv_decode_attention,
    mixed_kv_decode_attention_reference,
)


def make_case(device: str = "cpu"):
    torch.manual_seed(43)
    q = torch.randn(1, 4, 8, dtype=torch.float16, device=device)
    full_k = torch.randn(1, 16, 2, 8, dtype=torch.float16, device=device)
    full_v = torch.randn_like(full_k)
    quant_k = torch.randint(-64, 63, (1, 16, 2, 8), dtype=torch.int8, device=device)
    quant_v = torch.randint(-64, 63, (1, 16, 2, 8), dtype=torch.int8, device=device)
    k_scale = torch.rand(1, 16, 2, 1, dtype=torch.float32, device=device) * 0.02
    v_scale = torch.rand_like(k_scale) * 0.02
    visible = torch.tensor(
        [[[FULL_STATE, 0, -1, 0, 16], [QUANT_STATE, -1, 0, 16, 32], [EVICT_STATE, -1, -1, 32, 48]]],
        dtype=torch.int64,
        device=device,
    )
    counts = torch.tensor([3], dtype=torch.int32, device=device)
    return q, full_k, full_v, quant_k, quant_v, k_scale, v_scale, visible, counts


class AttentionMassOutputTest(unittest.TestCase):
    def test_reference_mass_sums_to_one_and_evict_is_zero(self):
        tensors = make_case()
        q = tensors[0]
        mass = torch.empty(q.shape[0], q.shape[1], tensors[7].shape[1], dtype=torch.float32)

        mixed_kv_decode_attention_reference(*tensors, softmax_scale=0.125, block_attn_mass=mass)

        torch.testing.assert_close(mass[:, :, :2].sum(dim=-1), torch.ones(q.shape[0], q.shape[1]), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(mass[:, :, 2], torch.zeros(q.shape[0], q.shape[1]), atol=1e-6, rtol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for mixed-KV decode attention mass parity")
    def test_triton_mass_matches_reference_on_cuda(self):
        tensors = make_case(device="cuda")
        q = tensors[0]
        reference_mass = torch.empty(q.shape[0], q.shape[1], tensors[7].shape[1], dtype=torch.float32, device="cuda")
        actual_mass = torch.empty_like(reference_mass)
        reference = mixed_kv_decode_attention_reference(*tensors, softmax_scale=0.125, block_attn_mass=reference_mass)

        actual = mixed_kv_decode_attention(*tensors, softmax_scale=0.125, block_attn_mass=actual_mass)
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, reference, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(actual_mass, reference_mass, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
