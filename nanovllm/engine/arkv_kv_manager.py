"""P3 ARKV KV budget split and rollback-safe FULL->QUANT commit path."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import torch

from nanovllm.engine.kv_meta import KVBlockState, MetadataConsistencyError, PhysicalBlockTable, SequenceKVRefTable
from nanovllm.engine.quant_cache import QuantCache, QuantCacheError, QuantPoolExhaustedError
from nanovllm.engine.visible_tables import VisibleBlockTable, VisibleTableConfig, build_visible_block_table
from nanovllm.kernels.q8_kv import q8_block_nbytes, q8_scale_nbytes
from nanovllm.layers.mixed_kv_fallback import enable_full_reuse_after_quant


class BudgetConfigError(ValueError):
    pass


class QuantizationCommitError(RuntimeError):
    pass


class MetadataCommitError(QuantizationCommitError):
    pass


class RollbackError(QuantizationCommitError):
    pass


@dataclass(frozen=True, slots=True)
class KVCacheBudget:
    total_kv_budget_bytes: int
    full_pool_bytes: int
    quant_pool_bytes: int
    scale_bytes: int
    scratch_budget: int
    metadata_budget: int
    full_pool_blocks: int
    quant_pool_blocks: int
    full_block_bytes: int
    quant_block_bytes: int
    scale_block_bytes: int


@dataclass(frozen=True, slots=True)
class QuantizeCommitResult:
    transaction_id: str
    storage_id: int
    quant_block_id: int
    released_full_block_id: int | None
    reclaimed_full_equiv_blocks: int
    reason: str
    step: int


@dataclass(slots=True)
class _QuantizeTransaction:
    transaction_id: str
    storage_id: int
    quant_block_id: int | None
    old_state: KVBlockState
    old_full_block_id: int | None
    old_quant_block_id: int | None
    metadata_updated: bool = False


def kv_cache_block_shape(config) -> tuple[int, int, int, int, int]:
    hf_config = config.hf_config
    if hf_config is None:
        raise BudgetConfigError("hf_config is required to compute KV cache block shape")
    num_kv_heads = hf_config.num_key_value_heads // config.tensor_parallel_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    return (2, hf_config.num_hidden_layers, config.kvcache_block_size, num_kv_heads, head_dim)


def full_block_nbytes(config) -> int:
    hf_config = config.hf_config
    if hf_config is None:
        raise BudgetConfigError("hf_config is required to compute KV cache block bytes")
    numel = 1
    for dim in kv_cache_block_shape(config):
        numel *= int(dim)
    return numel * hf_config.dtype.itemsize


def compute_kv_cache_budget(config, total_kv_budget_bytes: int | None = None) -> KVCacheBudget:
    """Split the single KV budget into full, Q8, scales, scratch, and metadata pools."""
    total = int(total_kv_budget_bytes if total_kv_budget_bytes is not None else config.total_kv_budget_bytes)
    if total <= 0:
        raise BudgetConfigError("total_kv_budget_bytes must be positive")

    block_shape = kv_cache_block_shape(config)
    full_bytes = full_block_nbytes(config)
    quant_bytes = q8_block_nbytes(block_shape)
    scale_bytes_per_block = q8_scale_nbytes(block_shape)
    scratch_budget = int(config.kv_q8_scratch_blocks) * full_bytes
    metadata_budget = min(int(config.kv_metadata_budget_bytes), max(total - scratch_budget, 0))
    fixed_budget = scratch_budget + metadata_budget
    if fixed_budget >= total:
        raise BudgetConfigError("scratch + metadata budget leaves no room for KV pools")

    remaining = total - fixed_budget
    quant_fraction = float(config.kv_q8_quant_pool_fraction)
    quant_plus_scale_budget = int(remaining * quant_fraction)
    quant_pool_blocks = quant_plus_scale_budget // (quant_bytes + scale_bytes_per_block)
    quant_pool_bytes = quant_pool_blocks * quant_bytes
    scale_bytes = quant_pool_blocks * scale_bytes_per_block
    full_pool_bytes = remaining - quant_pool_bytes - scale_bytes
    full_pool_blocks = full_pool_bytes // full_bytes
    if full_pool_blocks < int(config.min_full_kvcache_blocks):
        raise BudgetConfigError(
            f"full pool has {full_pool_blocks} blocks, below minimum {config.min_full_kvcache_blocks}"
        )

    return KVCacheBudget(
        total_kv_budget_bytes=total,
        full_pool_bytes=full_pool_bytes,
        quant_pool_bytes=quant_pool_bytes,
        scale_bytes=scale_bytes,
        scratch_budget=scratch_budget,
        metadata_budget=metadata_budget,
        full_pool_blocks=full_pool_blocks,
        quant_pool_blocks=quant_pool_blocks,
        full_block_bytes=full_bytes,
        quant_block_bytes=quant_bytes,
        scale_block_bytes=scale_bytes_per_block,
    )


class ARKVKVManager:
    def __init__(
        self,
        full_cache: torch.Tensor,
        quant_cache: QuantCache,
        physical_table: PhysicalBlockTable,
        ref_table: SequenceKVRefTable | None = None,
        visible_table: VisibleBlockTable | None = None,
        mixed_kv_read_available: bool = False,
        release_full_callback=None,
    ):
        self.full_cache = full_cache
        self.quant_cache = quant_cache
        self.physical_table = physical_table
        self.ref_table = ref_table
        self.visible_table = visible_table
        self.mixed_kv_read_available = mixed_kv_read_available
        self.release_full_callback = release_full_callback
        self._transactions: dict[str, _QuantizeTransaction] = {}

    def quantize_from_full(
        self,
        storage_id: int,
        reason: str,
        step: int,
        allow_release_full: bool,
        fail_at: str | None = None,
    ) -> QuantizeCommitResult:
        meta = self.physical_table.get(storage_id)
        if meta.state != KVBlockState.FULL or meta.full_block_id is None:
            raise MetadataCommitError(f"storage_id {storage_id} is not a FULL block")

        transaction_id = str(uuid4())
        tx = _QuantizeTransaction(
            transaction_id=transaction_id,
            storage_id=storage_id,
            quant_block_id=None,
            old_state=meta.state,
            old_full_block_id=meta.full_block_id,
            old_quant_block_id=meta.quant_block_id,
        )
        self._transactions[transaction_id] = tx

        try:
            quant_block_id = self.quant_cache.allocate()
            tx.quant_block_id = quant_block_id
            self._maybe_fail(fail_at, "after_allocate")

            full_block = self.full_cache[:, :, meta.full_block_id].detach()
            self.quant_cache.write_from_full(quant_block_id, full_block)
            self._maybe_fail(fail_at, "after_write")

            if allow_release_full and not self.mixed_kv_read_available:
                allow_release_full = False
            released_full_block_id = meta.full_block_id if allow_release_full else None

            self._maybe_fail(fail_at, "before_metadata")
            meta.state = KVBlockState.QUANT
            meta.quant_block_id = quant_block_id
            if released_full_block_id is not None:
                meta.full_block_id = None
            tx.metadata_updated = True
            self._maybe_fail(fail_at, "after_metadata")

            self._refresh_visible_entries(meta.storage_id)
            self._maybe_fail(fail_at, "after_visible")

            result = QuantizeCommitResult(
                transaction_id=transaction_id,
                storage_id=storage_id,
                quant_block_id=quant_block_id,
                released_full_block_id=released_full_block_id,
                reclaimed_full_equiv_blocks=1 if released_full_block_id is not None else 0,
                reason=reason,
                step=step,
            )
            if released_full_block_id is not None:
                enable_full_reuse_after_quant(storage_id, result)
                if self.release_full_callback is not None:
                    self.release_full_callback(released_full_block_id)
            del self._transactions[transaction_id]
            return result
        except Exception:
            self.rollback_quantize_prepare(transaction_id)
            raise

    def rollback_quantize_prepare(self, transaction_id: str) -> None:
        tx = self._transactions.pop(transaction_id, None)
        if tx is None:
            raise RollbackError(f"unknown transaction_id: {transaction_id}")
        try:
            meta = self.physical_table.get(tx.storage_id)
            meta.state = tx.old_state
            meta.full_block_id = tx.old_full_block_id
            meta.quant_block_id = tx.old_quant_block_id
            if tx.quant_block_id is not None and tx.quant_block_id in self.quant_cache.used_quant_block_ids:
                self.quant_cache.free(tx.quant_block_id)
            if tx.metadata_updated:
                self._refresh_visible_entries(tx.storage_id)
        except (MetadataConsistencyError, QuantCacheError) as exc:
            raise RollbackError(f"failed to rollback {transaction_id}: {exc}") from exc

    def _refresh_visible_entries(self, storage_id: int) -> None:
        if self.ref_table is None or self.visible_table is None:
            return
        meta = self.physical_table.get(storage_id)
        for owner in meta.copy_owner_refs():
            entries = build_visible_block_table(
                owner.seq_id,
                self.ref_table.refs_for_seq(owner.seq_id),
                self.physical_table,
                VisibleTableConfig(include_quant=True),
            )
            self.visible_table.add_entries(owner.seq_id, entries)

    @staticmethod
    def _maybe_fail(fail_at: str | None, point: str) -> None:
        if fail_at == point:
            if point in {"before_metadata", "after_metadata", "after_visible"}:
                raise MetadataCommitError(f"injected failure at {point}")
            if point == "after_allocate":
                raise QuantPoolExhaustedError(f"injected failure at {point}")
            raise QuantizationCommitError(f"injected failure at {point}")
