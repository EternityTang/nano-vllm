from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.metrics import MetricsRecorder, MetricsStateError


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "benchmark_serving.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_serving", BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MetricsSmokeTest(unittest.TestCase):
    def test_optimizer_flags_default_off(self):
        defaults = {field.name: field.default for field in fields(Config)}
        for name, value in defaults.items():
            if name.startswith("enable_") and name != "enable_metrics_hooks":
                self.assertFalse(value, name)
        self.assertFalse(defaults["enable_metrics_hooks"])

    def test_request_event_ordering(self):
        recorder = MetricsRecorder()
        recorder.add_request("r0", prompt_tokens=8, output_tokens=4, arrival_ts=1.0)
        recorder.record_request_event("r0", "scheduled", 2.0)
        recorder.record_request_event("r0", "first_token", 3.0)
        recorder.record_request_event("r0", "finish", 4.0)
        metric = recorder.request_dicts()[0]
        self.assertEqual(metric["request_id"], "r0")
        self.assertEqual(metric["prompt_tokens"], 8)
        self.assertEqual(metric["output_tokens"], 4)

    def test_invalid_request_event_order_raises(self):
        recorder = MetricsRecorder()
        recorder.add_request("r0", prompt_tokens=8, arrival_ts=1.0)
        with self.assertRaises(MetricsStateError):
            recorder.record_request_event("r0", "first_token", 2.0)

    def test_block_manager_metrics_are_read_only(self):
        manager = BlockManager(num_blocks=4, block_size=256, bytes_per_block=1024)
        before = (len(manager.free_block_ids), len(manager.used_block_ids))
        metric = manager.collect_metrics(step=3)
        after = (len(manager.free_block_ids), len(manager.used_block_ids))
        self.assertEqual(before, after)
        self.assertEqual(metric.free_full_blocks, 4)
        self.assertEqual(metric.active_full_blocks, 0)
        self.assertEqual(metric.effective_kv_memory_bytes, 0)

    def test_benchmark_dry_run_writes_json_and_csv(self):
        benchmark = load_benchmark_module()
        with tempfile.TemporaryDirectory() as tmp:
            output_json = Path(tmp) / "b0_dryrun.json"
            report = benchmark.run_serving_benchmark(
                workload_name="scheduler_stress",
                model="/tmp/qwen3-placeholder",
                concurrency=2,
                max_requests=3,
                output_json=str(output_json),
                dry_run=True,
            )
            csv_path = output_json.with_suffix(".csv")
            loaded = json.loads(output_json.read_text(encoding="utf-8"))
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(report["status"], "ok")
        self.assertEqual(loaded["schema_version"], 1)
        self.assertEqual(loaded["workload"]["name"], "scheduler_stress")
        self.assertEqual(loaded["summary"]["request_count"], 3)
        self.assertEqual(loaded["summary"]["active_quant_blocks"], 0)
        self.assertEqual(loaded["summary"]["evicted_blocks"], 0)
        self.assertTrue(all(value is False for value in loaded["optimizer_flags"].values()))
        self.assertEqual(len(rows), 1)
        self.assertIn("ttft_p50_s", rows[0])
        self.assertIn("raw_peak_vram_bytes", rows[0])


if __name__ == "__main__":
    unittest.main()
