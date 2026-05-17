"""P5 fail-closed quality gate for optional KV EVICT experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class QualityGateError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class QualityGateThresholds:
    max_passkey_drop_abs: float = 0.0
    max_retrieval_drop_abs: float = 0.0
    min_greedy_token_agreement: float = 0.99
    min_slo_goodput_delta: float = -0.05


@dataclass(frozen=True, slots=True)
class QualityGateResult:
    passed: bool
    passkey_drop_abs: float
    retrieval_drop_abs: float
    greedy_token_agreement: float
    slo_goodput_delta: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_quality_gate(
    benchmark_results: dict[str, Any],
    thresholds: QualityGateThresholds | None = None,
) -> QualityGateResult:
    """Validate EVICT quality metrics with fail-closed defaults."""
    thresholds = thresholds or QualityGateThresholds()
    metrics = benchmark_results.get("quality_metrics")
    if not isinstance(metrics, dict):
        raise QualityGateError("quality_metrics are required before enabling EVICT")

    required = (
        "passkey_drop_abs",
        "retrieval_drop_abs",
        "greedy_token_agreement",
        "slo_goodput_delta",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise QualityGateError(f"quality_metrics missing keys: {missing}")

    result = QualityGateResult(
        passed=True,
        passkey_drop_abs=float(metrics["passkey_drop_abs"]),
        retrieval_drop_abs=float(metrics["retrieval_drop_abs"]),
        greedy_token_agreement=float(metrics["greedy_token_agreement"]),
        slo_goodput_delta=float(metrics["slo_goodput_delta"]),
        reason="passed",
    )
    failures = []
    if result.passkey_drop_abs > thresholds.max_passkey_drop_abs:
        failures.append("passkey_drop_abs")
    if result.retrieval_drop_abs > thresholds.max_retrieval_drop_abs:
        failures.append("retrieval_drop_abs")
    if result.greedy_token_agreement < thresholds.min_greedy_token_agreement:
        failures.append("greedy_token_agreement")
    if result.slo_goodput_delta < thresholds.min_slo_goodput_delta:
        failures.append("slo_goodput_delta")
    if not failures:
        return result
    return QualityGateResult(
        passed=False,
        passkey_drop_abs=result.passkey_drop_abs,
        retrieval_drop_abs=result.retrieval_drop_abs,
        greedy_token_agreement=result.greedy_token_agreement,
        slo_goodput_delta=result.slo_goodput_delta,
        reason="failed: " + ",".join(failures),
    )
