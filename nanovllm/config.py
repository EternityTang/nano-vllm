import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enable_metrics_hooks: bool = False
    enable_memory_aware_optimizer: bool = False
    enable_memory_aware_scheduler: bool = False
    enable_admission_controller: bool = False
    prefill_chunk_min_tokens: int = 256
    prefill_chunk_max_tokens: int = 2048
    long_prefill_token_threshold: int = 2048
    scheduler_starvation_threshold: int = 4
    enable_arkv_metadata: bool = False
    enable_arkv_policy_dry_run: bool = False
    enable_kv_q8_runtime: bool = False
    enable_kv_q8_shadow: bool = False
    enable_mixed_kv_fallback: bool = False
    enable_kv_evict: bool = False
    enable_direct_full_evict: bool = False
    enable_triton_gather_dequant: bool = False
    enable_mixed_kv_decode_kernel: bool = False
    enable_quality_gate: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.enable_mixed_kv_fallback and not self.enforce_eager:
            raise ValueError("enable_mixed_kv_fallback requires enforce_eager=True until graph safety is proven")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
