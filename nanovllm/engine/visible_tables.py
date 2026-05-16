# 中文说明：
# P2 visible table 模块，构造按逻辑顺序排列的 attention read view，并显式区分 logical_context_len 与 visible_context_len。
# VisibleBlockTable 只描述读视图，不用于写路径或 slot_mapping；后续 mixed-KV fallback 会在此基础上读取 FULL/QUANT 可见块。
"""P2 visible KV read-view tables separated from logical and write metadata."""

from __future__ import annotations

from dataclasses import dataclass

from nanovllm.engine.kv_meta import (
    KVBlockState,
    MetadataConsistencyError,
    PhysicalBlockTable,
    SequenceKVRef,
)


class VisibleTableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VisibleTableConfig:
    include_quant: bool = True
    include_evict: bool = False


@dataclass(frozen=True, slots=True)
class VisibleBlockEntry:
    seq_id: int
    logical_block_id: int
    storage_id: int
    state: KVBlockState
    full_block_id: int | None
    logical_start: int
    logical_end: int
    visible_start: int
    visible_end: int


class VisibleBlockTable:
    def __init__(self):
        self._entries_by_seq: dict[int, tuple[VisibleBlockEntry, ...]] = {}

    def add_entries(self, seq_id: int, entries: list[VisibleBlockEntry]) -> None:
        self._entries_by_seq[seq_id] = tuple(entries)

    def entries_for_seq(self, seq_id: int) -> tuple[VisibleBlockEntry, ...]:
        return self._entries_by_seq.get(seq_id, ())

    def seq_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._entries_by_seq))

    def visible_context_len(self, seq_id: int) -> int:
        entries = self.entries_for_seq(seq_id)
        return entries[-1].visible_end if entries else 0


def build_visible_block_table(
    seq_id: int,
    logical_refs: list[SequenceKVRef],
    physical_table: PhysicalBlockTable,
    cfg: VisibleTableConfig,
) -> list[VisibleBlockEntry]:
    entries: list[VisibleBlockEntry] = []
    visible_cursor = 0
    previous_logical_end = 0
    refs = sorted(logical_refs, key=lambda ref: ref.logical_block_id)

    for ref in refs:
        if ref.seq_id != seq_id:
            raise VisibleTableError(f"ref seq_id {ref.seq_id} does not match requested seq_id {seq_id}")
        if ref.logical_start != previous_logical_end:
            raise VisibleTableError("non-monotonic logical span")
        previous_logical_end = ref.logical_end
        if not ref.visible:
            continue
        try:
            meta = physical_table.get(ref.storage_id)
        except MetadataConsistencyError as exc:
            raise VisibleTableError(str(exc)) from exc
        if meta.state == KVBlockState.QUANT and not cfg.include_quant:
            raise VisibleTableError("QUANT block is not enabled for visible table")
        if meta.state == KVBlockState.EVICT and not cfg.include_evict:
            raise VisibleTableError("EVICT block is not enabled for visible table")
        if meta.state == KVBlockState.FULL and meta.full_block_id is None:
            raise VisibleTableError("FULL visible entry is missing full_block_id")

        visible_len = ref.logical_end - ref.logical_start
        entries.append(
            VisibleBlockEntry(
                seq_id=seq_id,
                logical_block_id=ref.logical_block_id,
                storage_id=ref.storage_id,
                state=meta.state,
                full_block_id=meta.full_block_id,
                logical_start=ref.logical_start,
                logical_end=ref.logical_end,
                visible_start=visible_cursor,
                visible_end=visible_cursor + visible_len,
            )
        )
        visible_cursor += visible_len

    return entries
