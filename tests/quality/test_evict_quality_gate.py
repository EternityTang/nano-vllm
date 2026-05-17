"""P5 quality-gate tests for optional EVICT."""

from __future__ import annotations

import unittest

from nanovllm.engine.quality_gate import QualityGateError, QualityGateThresholds, run_quality_gate


class EvictQualityGateTest(unittest.TestCase):
    def test_quality_gate_passes_at_thresholds(self):
        result = run_quality_gate(
            {
                "quality_metrics": {
                    "passkey_drop_abs": 0.0,
                    "retrieval_drop_abs": 0.0,
                    "greedy_token_agreement": 1.0,
                    "slo_goodput_delta": 0.0,
                }
            }
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "passed")

    def test_quality_gate_fails_when_any_metric_regresses(self):
        result = run_quality_gate(
            {
                "quality_metrics": {
                    "passkey_drop_abs": 0.1,
                    "retrieval_drop_abs": 0.0,
                    "greedy_token_agreement": 0.98,
                    "slo_goodput_delta": -0.1,
                }
            },
            QualityGateThresholds(max_passkey_drop_abs=0.0),
        )

        self.assertFalse(result.passed)
        self.assertIn("passkey_drop_abs", result.reason)
        self.assertIn("greedy_token_agreement", result.reason)
        self.assertIn("slo_goodput_delta", result.reason)

    def test_quality_gate_requires_explicit_metrics(self):
        with self.assertRaises(QualityGateError):
            run_quality_gate({})

        with self.assertRaises(QualityGateError):
            run_quality_gate({"quality_metrics": {"passkey_drop_abs": 0.0}})


if __name__ == "__main__":
    unittest.main()
