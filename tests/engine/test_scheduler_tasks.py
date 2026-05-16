from __future__ import annotations

# 中文说明：
# P1 scheduler/task abstraction 回归测试，验证 BatchPlan homogeneous invariant、decode-first 调度、chunked prefill 和 legacy scheduler fallback。
# 该文件确保 memory-aware scheduler 只在 flags 开启时生效，关闭 flags 后仍保持原 Nano-VLLM prefill-first 基线行为。

from types import SimpleNamespace
import unittest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.tasks import BatchPlan, DecodeTask, PrefillTask, SchedulerInvariantError, build_batch_plan
from nanovllm.sampling_params import SamplingParams


def make_config(**overrides):
    values = dict(
        max_num_batched_tokens=512,
        max_num_seqs=8,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=16,
        hf_config=None,
        tensor_parallel_size=1,
        enable_memory_aware_scheduler=True,
        enable_admission_controller=True,
        prefill_chunk_min_tokens=1,
        prefill_chunk_max_tokens=512,
        long_prefill_token_threshold=512,
        scheduler_starvation_threshold=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_seq(tokens: int, max_tokens: int = 4) -> Sequence:
    return Sequence(list(range(1, tokens + 1)), SamplingParams(max_tokens=max_tokens))


class SchedulerTasksTest(unittest.TestCase):
    def test_batch_plan_rejects_mixed_tasks(self):
        with self.assertRaises(SchedulerInvariantError):
            BatchPlan(
                batch_id=0,
                kind="decode",
                decode_tasks=[DecodeTask(seq_id=1, request_id="1")],
                prefill_tasks=[
                    PrefillTask(
                        seq_id=2,
                        request_id="2",
                        start_pos=0,
                        chunk_tokens=1,
                        is_long_prefill=False,
                        skip_count=0,
                    )
                ],
                token_budget=1,
            )

    def test_build_batch_plan_prefers_decode(self):
        running = [make_seq(8)]
        waiting = [make_seq(32)]
        plan = build_batch_plan(
            waiting=waiting,
            running=running,
            sched_snapshot=SimpleNamespace(step=7),
            kv_snapshot=SimpleNamespace(),
            cfg=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=512),
        )
        self.assertEqual(plan.kind, "decode")
        self.assertEqual(len(plan.decode_tasks), 1)
        self.assertFalse(plan.prefill_tasks)

    def test_memory_aware_scheduler_decode_first_and_homogeneous(self):
        scheduler = Scheduler(make_config())
        running = make_seq(1)
        running.status = SequenceStatus.RUNNING
        running.block_table.append(scheduler.block_manager._allocate_block())
        waiting = make_seq(32)
        scheduler.running.append(running)
        scheduler.waiting.append(waiting)

        seqs, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual([seq.seq_id for seq in seqs], [running.seq_id])
        self.assertEqual(waiting.scheduler_skip_count, 1)
        self.assertEqual(scheduler.metrics.steps[-1].batch_kind, "decode")

    def test_memory_aware_scheduler_chunks_prefill(self):
        scheduler = Scheduler(make_config(max_num_batched_tokens=128, prefill_chunk_max_tokens=128))
        waiting = make_seq(600)
        scheduler.waiting.append(waiting)

        seqs, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [waiting])
        self.assertEqual(waiting.num_scheduled_tokens, 128)
        self.assertEqual(scheduler.metrics.steps[-1].batch_kind, "prefill")
        self.assertEqual(scheduler.metrics.steps[-1].scheduled_tokens, 128)

    def test_legacy_scheduler_still_prefills_before_decode(self):
        scheduler = Scheduler(make_config(enable_memory_aware_scheduler=False, enable_admission_controller=False))
        running = make_seq(1)
        running.status = SequenceStatus.RUNNING
        running.block_table.append(scheduler.block_manager._allocate_block())
        waiting = make_seq(32)
        scheduler.running.append(running)
        scheduler.waiting.append(waiting)

        seqs, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [waiting])


if __name__ == "__main__":
    unittest.main()
