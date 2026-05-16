from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.metrics import KVPoolMetrics


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, bytes_per_block: int = 0):
        self.block_size = block_size
        self.bytes_per_block = bytes_per_block
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.quantized_released_block_ids: set[int] = set()
        self.arkv_metrics: dict[str, int | float] = {}

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        self.quantized_released_block_ids.discard(block_id)
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def release_full_block_after_quant(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block_id not in self.used_block_ids:
            return
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.ref_count = 0
        block.hash = -1
        block.token_ids = []
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        self.quantized_released_block_ids.add(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            if block_id == -1:
                continue
            if block_id in self.quantized_released_block_ids:
                self.quantized_released_block_ids.remove(block_id)
                continue
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        if start == 0 or seq.block_table[start - 1] == -1:
            h = -1
        else:
            h = self.blocks[seq.block_table[start - 1]].hash
        for i in range(start, end):
            if seq.block_table[i] == -1:
                continue
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    def collect_metrics(self, step: int, raw_peak_vram_bytes: int = 0) -> KVPoolMetrics:
        total_blocks = len(self.blocks)
        free_full_blocks = len(self.free_block_ids)
        active_full_blocks = len(self.used_block_ids)
        free_full_block_ratio = free_full_blocks / total_blocks if total_blocks else 0.0
        active_quant_blocks = int(self.arkv_metrics.get("active_quant_blocks", 0))
        return KVPoolMetrics(
            step=step,
            free_full_blocks=free_full_blocks,
            active_full_blocks=active_full_blocks,
            active_quant_blocks=active_quant_blocks,
            evicted_blocks=int(self.arkv_metrics.get("evicted_blocks", 0)),
            free_full_block_ratio=free_full_block_ratio,
            effective_kv_memory_bytes=active_full_blocks * self.bytes_per_block,
            raw_peak_vram_bytes=raw_peak_vram_bytes,
            quantized_block_ratio=active_quant_blocks / max(active_full_blocks + active_quant_blocks, 1),
            reclaim_trigger_count=int(self.arkv_metrics.get("reclaim_trigger_count", 0)),
            quant_commits_success=int(self.arkv_metrics.get("quant_commits_success", 0)),
            quant_commits_rollback=int(self.arkv_metrics.get("quant_commits_rollback", 0)),
            full_blocks_released_after_quant=int(self.arkv_metrics.get("full_blocks_released_after_quant", 0)),
            mixed_kv_quant_reads=int(self.arkv_metrics.get("mixed_kv_quant_reads", 0)),
            visible_quant_entries=int(self.arkv_metrics.get("visible_quant_entries", 0)),
            free_full_blocks_before_reclaim=int(self.arkv_metrics.get("free_full_blocks_before_reclaim", 0)),
            free_full_blocks_after_reclaim=int(self.arkv_metrics.get("free_full_blocks_after_reclaim", 0)),
            free_full_blocks_reclaim_delta=int(self.arkv_metrics.get("free_full_blocks_reclaim_delta", 0)),
        )
