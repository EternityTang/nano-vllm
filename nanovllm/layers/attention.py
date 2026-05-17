import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.mixed_kv_fallback import (
    AttentionMetadata,
    run_decode_mixed_kv_fallback,
    run_prefill_mixed_kv_fallback,
)
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.layer_id = 0

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.use_prefill_mixed_kv_fallback and context.visible_entries is not None:
                if context.quant_cache is None or context.mixed_kv_workspace is None:
                    raise RuntimeError("prefill mixed-KV fallback requires quant_cache and workspace")
                context.mixed_kv_quant_reads += sum(
                    1
                    for entries in context.visible_entries
                    for entry in entries
                    if getattr(getattr(entry, "state", None), "value", None) == "quant"
                )
                o = run_prefill_mixed_kv_fallback(
                    q,
                    context.visible_entries,
                    context.slot_mapping,
                    k_cache,
                    v_cache,
                    context.quant_cache,
                    context.mixed_kv_workspace,
                    AttentionMetadata(
                        layer_id=self.layer_id,
                        softmax_scale=self.scale,
                        query_lengths=context.prefill_query_lengths,
                        query_start_positions=context.prefill_query_start_positions,
                    ),
                    use_triton_gather_dequant=context.use_triton_gather_dequant,
                )
            else:
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            if context.use_mixed_kv_fallback and context.visible_entries is not None:
                if context.quant_cache is None or context.mixed_kv_workspace is None:
                    raise RuntimeError("mixed-KV fallback requires quant_cache and workspace")
                context.mixed_kv_quant_reads += sum(
                    1
                    for entries in context.visible_entries
                    for entry in entries
                    if getattr(getattr(entry, "state", None), "value", None) == "quant"
                )
                o = run_decode_mixed_kv_fallback(
                    q,
                    context.visible_entries,
                    k_cache,
                    v_cache,
                    context.quant_cache,
                    context.mixed_kv_workspace,
                    AttentionMetadata(layer_id=self.layer_id, softmax_scale=self.scale),
                    use_triton_gather_dequant=context.use_triton_gather_dequant,
                    use_mixed_kv_decode_kernel=context.use_mixed_kv_decode_kernel,
                    enable_attention_mass_output=context.enable_attention_mass_output,
                    packed_visible_entries=context.packed_visible_entries,
                    packed_visible_entry_counts=context.packed_visible_entry_counts,
                )
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
        return o
