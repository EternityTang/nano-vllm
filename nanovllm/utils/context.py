from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    use_mixed_kv_fallback: bool = False
    visible_entries: object | None = None
    quant_cache: object | None = None
    mixed_kv_workspace: torch.Tensor | None = None
    mixed_kv_quant_reads: int = 0
    visible_quant_entries: int = 0

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    use_mixed_kv_fallback=False,
    visible_entries=None,
    quant_cache=None,
    mixed_kv_workspace=None,
    mixed_kv_quant_reads=0,
    visible_quant_entries=0,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        use_mixed_kv_fallback,
        visible_entries,
        quant_cache,
        mixed_kv_workspace,
        mixed_kv_quant_reads,
        visible_quant_entries,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
