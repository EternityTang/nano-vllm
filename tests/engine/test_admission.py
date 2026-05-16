from __future__ import annotations

# 中文说明：
# P1 admission controller 单元测试，覆盖 future decode reserve、long prefill chunk、admit/shrink/defer 以及 starvation guard 决策。
# 目标是锁住 admission 策略的纯函数行为，避免 scheduler 集成时把 KV reserve 估算和队列副作用混在一起。

import unittest

from nanovllm.engine.admission import (
    AdmitAction,
    KVSnapshot,
    SchedulerConfig,
    SchedulerSnapshot,
    choose_prefill_chunk,
    decide_admission,
    estimate_future_decode_reserve,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def make_seq(tokens: int, max_tokens: int = 64) -> Sequence:
    return Sequence(list(range(1, tokens + 1)), SamplingParams(max_tokens=max_tokens))


class AdmissionTest(unittest.TestCase):
    def test_estimate_future_decode_reserve(self):
        cfg = SchedulerConfig(block_size=16, decode_reserve_blocks_per_seq=1)
        self.assertEqual(estimate_future_decode_reserve(make_seq(4, max_tokens=1), cfg), 1)
        self.assertEqual(estimate_future_decode_reserve(make_seq(4, max_tokens=33), cfg), 3)

    def test_choose_prefill_chunk_caps_long_prefill(self):
        cfg = SchedulerConfig(
            block_size=16,
            max_num_batched_tokens=512,
            prefill_chunk_min_tokens=1,
            prefill_chunk_max_tokens=512,
            long_prefill_token_threshold=128,
        )
        chunk = choose_prefill_chunk(
            make_seq(500),
            SchedulerSnapshot(waiting=1, running=1),
            KVSnapshot(free_full_blocks=64, active_full_blocks=0, total_full_blocks=64),
            cfg,
        )
        self.assertEqual(chunk, 128)

    def test_decide_admission_admits_when_reserve_fits(self):
        cfg = SchedulerConfig(block_size=16, prefill_chunk_min_tokens=1)
        decision = decide_admission(
            make_seq(16, max_tokens=16),
            SchedulerSnapshot(waiting=1, running=0),
            KVSnapshot(free_full_blocks=4, active_full_blocks=0, total_full_blocks=8),
            cfg,
        )
        self.assertEqual(decision.action, AdmitAction.ADMIT)
        self.assertTrue(decision.admitted)

    def test_decide_admission_shrinks_when_decode_reserve_does_not_fit(self):
        cfg = SchedulerConfig(block_size=16, prefill_chunk_min_tokens=1)
        decision = decide_admission(
            make_seq(16, max_tokens=64),
            SchedulerSnapshot(waiting=1, running=0),
            KVSnapshot(free_full_blocks=1, active_full_blocks=0, total_full_blocks=8),
            cfg,
        )
        self.assertEqual(decision.action, AdmitAction.SHRINK)

    def test_decide_admission_defers_when_no_blocks_and_decode_running(self):
        cfg = SchedulerConfig(block_size=16, prefill_chunk_min_tokens=1)
        decision = decide_admission(
            make_seq(16, max_tokens=16),
            SchedulerSnapshot(waiting=1, running=1),
            KVSnapshot(free_full_blocks=0, active_full_blocks=8, total_full_blocks=8),
            cfg,
        )
        self.assertEqual(decision.action, AdmitAction.DEFER)

    def test_starvation_guard_shrinks_with_some_free_blocks(self):
        cfg = SchedulerConfig(block_size=16, prefill_chunk_min_tokens=1, starvation_threshold=2)
        seq = make_seq(128, max_tokens=64)
        seq.scheduler_skip_count = 2
        decision = decide_admission(
            seq,
            SchedulerSnapshot(waiting=1, running=0),
            KVSnapshot(free_full_blocks=1, active_full_blocks=7, total_full_blocks=8),
            cfg,
        )
        self.assertEqual(decision.action, AdmitAction.SHRINK)
        self.assertEqual(decision.reason, "starvation guard")


if __name__ == "__main__":
    unittest.main()
