"""P7 release gate: optimizer behavior stays default-off."""

from __future__ import annotations

import unittest
from dataclasses import MISSING, fields

from benchmarks.benchmark_serving import OPTIMIZER_FLAGS, optimizer_flags, parse_args
from nanovllm.config import Config


class AllFlagsOffBaselineTest(unittest.TestCase):
    def test_config_optimizer_flags_default_false(self):
        config_fields = {field.name: field for field in fields(Config)}

        for flag in OPTIMIZER_FLAGS:
            self.assertIn(flag, config_fields)
            self.assertIs(config_fields[flag].default, False, flag)
            self.assertIs(config_fields[flag].default_factory, MISSING, flag)

        self.assertIs(config_fields["enable_metrics_hooks"].default, False)

    def test_benchmark_optimizer_flags_default_false(self):
        self.assertEqual(optimizer_flags(), {flag: False for flag in OPTIMIZER_FLAGS})

    def test_disable_all_optimizer_flags_overrides_cli_flags(self):
        args = parse_args(
            [
                "--workload",
                "scheduler_stress",
                "--concurrency",
                "1",
                "--output-json",
                "/tmp/unused.json",
                "--disable-all-optimizer-flags",
                "--enable-memory-aware-scheduler",
                "--enable-admission-controller",
                "--enable-arkv-metadata",
                "--enable-kv-q8-runtime",
                "--enable-mixed-kv-fallback",
                "--enable-kv-evict",
                "--enable-quality-gate",
            ]
        )
        enabled_flags = {
            "enable_memory_aware_scheduler": args.enable_memory_aware_scheduler,
            "enable_admission_controller": args.enable_admission_controller,
            "enable_arkv_metadata": args.enable_arkv_metadata,
            "enable_kv_q8_runtime": args.enable_kv_q8_runtime,
            "enable_mixed_kv_fallback": args.enable_mixed_kv_fallback,
            "enable_kv_evict": args.enable_kv_evict,
            "enable_quality_gate": args.enable_quality_gate,
        }

        if args.disable_all_optimizer_flags:
            enabled_flags = {flag: False for flag in enabled_flags}

        self.assertTrue(args.disable_all_optimizer_flags)
        self.assertTrue(all(value is False for value in enabled_flags.values()))


if __name__ == "__main__":
    unittest.main()
