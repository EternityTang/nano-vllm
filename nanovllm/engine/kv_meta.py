# 中文说明：
# P2 KV metadata truth table 模块，拆分 PhysicalBlockMeta 与 SequenceKVRef，分别记录物理存储块状态和每个序列的 logical_block_id -> storage_id 引用。
# 这里维护 owner_refs、ref_count、shared-prefix 标记、logical/write 表骨架和跨表 invariant 校验；P2 只做 metadata truth，不改变 attention runtime 语义。
"""P2 KV metadata truth tables for physical blocks and per-sequence logical refs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class MetadataConsistencyError(RuntimeError):
    pass


class KVBlockState(Enum):
    FULL = "full"
    QUANT = "quant"
    EVICT = "evict"


class SequenceKVState(Enum):
    ACTIVE = "active"
    PROTECTED = "protected"
    INFLIGHT_WRITE = "inflight_write"
    UNFINISHED_PREFILL = "unfinished_prefill"


@dataclass(frozen=True, slots=True, order=True)
class OwnerRef:
    seq_id: int
    logical_block_id: int


@dataclass(slots=True)
class PhysicalBlockMeta:
    storage_id: int
    state: KVBlockState
    full_block_id: int | None
    logical_start: int
    logical_end: int
    quant_block_id: int | None = None
    prefix_hash: int | None = None
    is_shared_prefix: bool = False
    owner_refs: set[OwnerRef] = field(default_factory=set)

    @property
    def ref_count(self) -> int:
        return len(self.owner_refs)

    def add_owner_ref(self, seq_id: int, logical_block_id: int) -> None:
        owner = OwnerRef(seq_id, logical_block_id)
        if owner in self.owner_refs:
            raise MetadataConsistencyError(f"duplicate owner ref: {owner}")
        self.owner_refs.add(owner)

    def copy_owner_refs(self) -> tuple[OwnerRef, ...]:
        return tuple(sorted(self.owner_refs))


@dataclass(frozen=True, slots=True)
class SequenceKVRef:
    seq_id: int
    logical_block_id: int
    storage_id: int
    logical_start: int
    logical_end: int
    state: SequenceKVState = SequenceKVState.ACTIVE
    is_sink: bool = False
    is_recent: bool = False
    is_inflight_write: bool = False
    visible: bool = True


class PhysicalBlockTable:
    def __init__(self):
        self._blocks: dict[int, PhysicalBlockMeta] = {}
        self._next_storage_id = 0

    def __contains__(self, storage_id: int) -> bool:
        return storage_id in self._blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def __iter__(self):
        return iter(self._blocks.values())

    def get(self, storage_id: int) -> PhysicalBlockMeta:
        try:
            return self._blocks[storage_id]
        except KeyError as exc:
            raise MetadataConsistencyError(f"unknown storage_id: {storage_id}") from exc

    def values(self) -> Iterable[PhysicalBlockMeta]:
        return self._blocks.values()

    def register_full_block(
        self,
        seq_id: int,
        logical_block_id: int,
        full_block_id: int,
        logical_start: int,
        logical_end: int,
        prefix_hash: int | None,
        is_shared_prefix: bool,
    ) -> int:
        _validate_logical_span(logical_start, logical_end)
        storage_id = self._next_storage_id
        self._next_storage_id += 1
        meta = PhysicalBlockMeta(
            storage_id=storage_id,
            state=KVBlockState.FULL,
            full_block_id=full_block_id,
            logical_start=logical_start,
            logical_end=logical_end,
            prefix_hash=prefix_hash,
            is_shared_prefix=is_shared_prefix,
        )
        meta.add_owner_ref(seq_id, logical_block_id)
        self._blocks[storage_id] = meta
        return storage_id

    def add_owner_ref(self, storage_id: int, seq_id: int, logical_block_id: int) -> None:
        self.get(storage_id).add_owner_ref(seq_id, logical_block_id)

    def snapshot(self) -> tuple[PhysicalBlockMeta, ...]:
        return tuple(self._blocks[storage_id] for storage_id in sorted(self._blocks))


class SequenceKVRefTable:
    def __init__(self):
        self._refs: dict[tuple[int, int], SequenceKVRef] = {}

    def __len__(self) -> int:
        return len(self._refs)

    def add_ref(self, ref: SequenceKVRef, physical_table: PhysicalBlockTable | None = None) -> None:
        _validate_logical_span(ref.logical_start, ref.logical_end)
        key = (ref.seq_id, ref.logical_block_id)
        if key in self._refs:
            raise MetadataConsistencyError(f"duplicate logical ref: {key}")
        if physical_table is not None:
            meta = physical_table.get(ref.storage_id)
            owner = OwnerRef(ref.seq_id, ref.logical_block_id)
            if owner not in meta.owner_refs:
                raise MetadataConsistencyError(f"physical block {ref.storage_id} missing owner ref {owner}")
        self._refs[key] = ref

    def replace_ref(self, ref: SequenceKVRef, physical_table: PhysicalBlockTable | None = None) -> None:
        _validate_logical_span(ref.logical_start, ref.logical_end)
        key = (ref.seq_id, ref.logical_block_id)
        if key not in self._refs:
            raise MetadataConsistencyError(f"unknown logical ref: {key}")
        if physical_table is not None:
            meta = physical_table.get(ref.storage_id)
            owner = OwnerRef(ref.seq_id, ref.logical_block_id)
            if owner not in meta.owner_refs:
                raise MetadataConsistencyError(f"physical block {ref.storage_id} missing owner ref {owner}")
        self._refs[key] = ref

    def get(self, seq_id: int, logical_block_id: int) -> SequenceKVRef:
        try:
            return self._refs[(seq_id, logical_block_id)]
        except KeyError as exc:
            raise MetadataConsistencyError(f"unknown logical ref: {(seq_id, logical_block_id)}") from exc

    def refs_for_seq(self, seq_id: int) -> list[SequenceKVRef]:
        refs = [ref for (ref_seq_id, _), ref in self._refs.items() if ref_seq_id == seq_id]
        return sorted(refs, key=lambda ref: ref.logical_block_id)

    def values(self) -> Iterable[SequenceKVRef]:
        return self._refs.values()


class LogicalBlockTable:
    def __init__(self):
        self._logical_to_storage: dict[tuple[int, int], int] = {}

    def bind(self, seq_id: int, logical_block_id: int, storage_id: int) -> None:
        key = (seq_id, logical_block_id)
        if key in self._logical_to_storage:
            raise MetadataConsistencyError(f"duplicate logical binding: {key}")
        self._logical_to_storage[key] = storage_id


class WriteBlockTable:
    def __init__(self):
        self._inflight: set[tuple[int, int]] = set()

    def mark_inflight(self, seq_id: int, logical_block_id: int) -> None:
        self._inflight.add((seq_id, logical_block_id))

    def is_inflight(self, seq_id: int, logical_block_id: int) -> bool:
        return (seq_id, logical_block_id) in self._inflight


def register_full_block(
    physical_table: PhysicalBlockTable,
    ref_table: SequenceKVRefTable,
    seq_id: int,
    logical_block_id: int,
    full_block_id: int,
    logical_start: int,
    logical_end: int,
    prefix_hash: int | None,
    is_shared_prefix: bool,
) -> int:
    storage_id = physical_table.register_full_block(
        seq_id=seq_id,
        logical_block_id=logical_block_id,
        full_block_id=full_block_id,
        logical_start=logical_start,
        logical_end=logical_end,
        prefix_hash=prefix_hash,
        is_shared_prefix=is_shared_prefix,
    )
    ref_table.add_ref(
        SequenceKVRef(
            seq_id=seq_id,
            logical_block_id=logical_block_id,
            storage_id=storage_id,
            logical_start=logical_start,
            logical_end=logical_end,
        ),
        physical_table,
    )
    return storage_id


def add_owner_ref(
    physical_table: PhysicalBlockTable,
    ref_table: SequenceKVRefTable,
    storage_id: int,
    seq_id: int,
    logical_block_id: int,
) -> None:
    meta = physical_table.get(storage_id)
    physical_table.add_owner_ref(storage_id, seq_id, logical_block_id)
    ref_table.add_ref(
        SequenceKVRef(
            seq_id=seq_id,
            logical_block_id=logical_block_id,
            storage_id=storage_id,
            logical_start=meta.logical_start,
            logical_end=meta.logical_end,
        ),
        physical_table,
    )


def validate_kv_tables(physical_table, ref_table: SequenceKVRefTable, visible_table) -> None:
    refs_by_storage: dict[int, set[OwnerRef]] = {}
    for ref in ref_table.values():
        meta = physical_table.get(ref.storage_id)
        owner = OwnerRef(ref.seq_id, ref.logical_block_id)
        if owner not in meta.owner_refs:
            raise MetadataConsistencyError(f"physical block {ref.storage_id} missing owner ref {owner}")
        refs_by_storage.setdefault(ref.storage_id, set()).add(owner)

    for meta in physical_table.values():
        if meta.ref_count != len(meta.owner_refs):
            raise MetadataConsistencyError(f"invalid ref_count for storage_id {meta.storage_id}")
        if meta.ref_count != len(refs_by_storage.get(meta.storage_id, set())):
            raise MetadataConsistencyError(f"owner_refs/ref_table mismatch for storage_id {meta.storage_id}")
        if meta.is_shared_prefix and meta.ref_count < 2:
            raise MetadataConsistencyError(f"shared prefix storage_id {meta.storage_id} has ref_count < 2")

    for seq_id in visible_table.seq_ids():
        previous_end = 0
        for entry in visible_table.entries_for_seq(seq_id):
            ref = ref_table.get(seq_id, entry.logical_block_id)
            if ref.storage_id != entry.storage_id:
                raise MetadataConsistencyError("visible entry points at a different storage_id than logical ref")
            if entry.visible_start != previous_end:
                raise MetadataConsistencyError("visible entries are not contiguous in visible order")
            previous_end = entry.visible_end


def _validate_logical_span(logical_start: int, logical_end: int) -> None:
    if logical_start < 0 or logical_end <= logical_start:
        raise MetadataConsistencyError(f"invalid logical span: {logical_start}:{logical_end}")
