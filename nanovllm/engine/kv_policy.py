# 中文说明：
# ARKV-inspired reclaim policy 模块，在不可变 KVSnapshot 上计算 block score、保护原因和 conservative reclaim plan。
# P5 之前只允许 FULL->QUANT 选择；P5 的 EVICT 路径必须显式开启 allow_evict 且质量门控通过。
"""ARKV-inspired reclaim policy scoring over immutable KV metadata snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nanovllm.engine.kv_meta import KVBlockState, PhysicalBlockMeta, SequenceKVRef, SequenceKVState


class PolicyError(RuntimeError):
    pass


class PolicyInvariantError(PolicyError):
    pass


class ReclaimPolicyName(Enum):
    ARKV_Q8_DRY_RUN = "arkv_q8_dry_run"
    ARKV_Q8_EVICT = "arkv_q8_evict"


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    protect_sink_blocks: int = 1
    protect_recent_blocks: int = 1
    allow_evict: bool = False
    allow_direct_full_evict: bool = False
    quality_gate_passed: bool = False


@dataclass(frozen=True, slots=True)
class KVSnapshot:
    physical_blocks: tuple[PhysicalBlockMeta, ...]
    sequence_refs: tuple[SequenceKVRef, ...]
    total_full_blocks: int
    free_full_blocks: int


@dataclass(frozen=True, slots=True)
class ReclaimCandidate:
    storage_id: int
    state: KVBlockState
    score: float
    reclaimable: bool
    protected_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReclaimPlan:
    policy_name: str
    required_full_equiv: int
    candidates: tuple[ReclaimCandidate, ...]
    selected_storage_ids: tuple[int, ...]
    conservative_reclaimable_blocks: int
    protected_blocks: int
    protected_ratio: float
    would_satisfy: bool

    def to_dict(self) -> dict:
        return {
            "policy_name": self.policy_name,
            "required_full_equiv": self.required_full_equiv,
            "candidate_count": len(self.candidates),
            "selected_storage_ids": list(self.selected_storage_ids),
            "conservative_reclaimable_blocks": self.conservative_reclaimable_blocks,
            "protected_blocks": self.protected_blocks,
            "protected_ratio": self.protected_ratio,
            "would_satisfy": self.would_satisfy,
            "candidates": [
                {
                    "storage_id": candidate.storage_id,
                    "state": candidate.state.value,
                    "score": candidate.score,
                    "reclaimable": candidate.reclaimable,
                    "protected_reasons": list(candidate.protected_reasons),
                }
                for candidate in self.candidates
            ],
        }


def build_policy_snapshot(physical_table, ref_table, total_full_blocks: int, free_full_blocks: int) -> KVSnapshot:
    return KVSnapshot(
        physical_blocks=tuple(physical_table.snapshot()),
        sequence_refs=tuple(sorted(ref_table.values(), key=lambda ref: (ref.seq_id, ref.logical_block_id))),
        total_full_blocks=total_full_blocks,
        free_full_blocks=free_full_blocks,
    )


def compute_block_score(meta: PhysicalBlockMeta, refs: tuple[SequenceKVRef, ...], cfg: PolicyConfig) -> float:
    if meta.state not in {KVBlockState.FULL, KVBlockState.QUANT}:
        return float("inf")
    latest_logical_end = max((ref.logical_end for ref in refs), default=meta.logical_end)
    sharing_discount = 0.25 if meta.ref_count > 1 else 0.0
    return float(latest_logical_end) + sharing_discount


def plan_reclaim_dry_run(
    snapshot: KVSnapshot,
    required_full_equiv: int,
    policy_name: ReclaimPolicyName | str,
    cfg: PolicyConfig,
) -> ReclaimPlan:
    if required_full_equiv < 0:
        raise PolicyError("required_full_equiv must be non-negative")
    policy = _normalize_policy_name(policy_name)
    if cfg.allow_evict and policy != ReclaimPolicyName.ARKV_Q8_EVICT:
        raise PolicyError("EVICT planning requires arkv_q8_evict policy")
    if policy == ReclaimPolicyName.ARKV_Q8_EVICT:
        if not cfg.allow_evict:
            raise PolicyError("EVICT planning is disabled")
        if not cfg.quality_gate_passed:
            raise PolicyError("EVICT planning requires a passed quality gate")
    elif cfg.allow_evict:
        raise PolicyError("EVICT planning is locked to the P5 quality-gated policy")

    refs_by_storage: dict[int, list[SequenceKVRef]] = {}
    for ref in snapshot.sequence_refs:
        refs_by_storage.setdefault(ref.storage_id, []).append(ref)

    candidates = []
    for meta in sorted(snapshot.physical_blocks, key=lambda item: item.storage_id):
        refs = tuple(sorted(refs_by_storage.get(meta.storage_id, []), key=lambda ref: (ref.seq_id, ref.logical_block_id)))
        reasons = _protected_reasons(meta, refs, cfg)
        reclaimable = _is_reclaimable_for_policy(meta, reasons, policy, cfg)
        candidates.append(
            ReclaimCandidate(
                storage_id=meta.storage_id,
                state=meta.state,
                score=compute_block_score(meta, refs, cfg),
                reclaimable=reclaimable,
                protected_reasons=tuple(reasons),
            )
        )

    reclaimable_candidates = _sort_reclaimable_candidates(candidates, policy)
    needed_after_free = max(required_full_equiv - snapshot.free_full_blocks, 0)
    selected = tuple(candidate.storage_id for candidate in reclaimable_candidates[:needed_after_free])
    protected_blocks = sum(1 for candidate in candidates if candidate.protected_reasons)
    protected_ratio = protected_blocks / len(candidates) if candidates else 0.0
    conservative_reclaimable = len(reclaimable_candidates)
    return ReclaimPlan(
        policy_name=policy.value,
        required_full_equiv=required_full_equiv,
        candidates=tuple(candidates),
        selected_storage_ids=selected,
        conservative_reclaimable_blocks=conservative_reclaimable,
        protected_blocks=protected_blocks,
        protected_ratio=protected_ratio,
        would_satisfy=snapshot.free_full_blocks + len(selected) >= required_full_equiv,
    )


def _normalize_policy_name(policy_name: ReclaimPolicyName | str) -> ReclaimPolicyName:
    if isinstance(policy_name, ReclaimPolicyName):
        return policy_name
    try:
        return ReclaimPolicyName(policy_name)
    except ValueError as exc:
        raise PolicyError(f"unknown reclaim policy: {policy_name}") from exc


def _protected_reasons(meta: PhysicalBlockMeta, refs: tuple[SequenceKVRef, ...], cfg: PolicyConfig) -> list[str]:
    reasons = []
    if meta.state == KVBlockState.EVICT:
        reasons.append(meta.state.value)
    if meta.state == KVBlockState.QUANT and not cfg.allow_evict:
        reasons.append(meta.state.value)
    if meta.state == KVBlockState.FULL and cfg.allow_evict and not cfg.allow_direct_full_evict:
        reasons.append("direct_full_evict_disabled")
    if meta.state not in {KVBlockState.FULL, KVBlockState.QUANT, KVBlockState.EVICT}:
        reasons.append(str(meta.state))
    if meta.is_shared_prefix or meta.ref_count > 1:
        reasons.append("shared_prefix")
    if any(ref.is_sink or ref.logical_block_id < cfg.protect_sink_blocks for ref in refs):
        reasons.append("sink")
    if any(ref.is_recent for ref in refs):
        reasons.append("recent")
    if any(ref.is_inflight_write for ref in refs):
        reasons.append("inflight_write")
    if any(ref.state == SequenceKVState.PROTECTED for ref in refs):
        reasons.append("protected")
    if any(ref.state == SequenceKVState.INFLIGHT_WRITE for ref in refs):
        reasons.append("inflight_write")
    if any(ref.state == SequenceKVState.UNFINISHED_PREFILL for ref in refs):
        reasons.append("unfinished_prefill")
    return reasons


def _is_reclaimable_for_policy(
    meta: PhysicalBlockMeta,
    protected_reasons: list[str],
    policy: ReclaimPolicyName,
    cfg: PolicyConfig,
) -> bool:
    if protected_reasons:
        return False
    if policy == ReclaimPolicyName.ARKV_Q8_EVICT:
        if meta.state == KVBlockState.QUANT:
            return True
        return meta.state == KVBlockState.FULL and cfg.allow_direct_full_evict
    return meta.state == KVBlockState.FULL


def _sort_reclaimable_candidates(
    candidates: list[ReclaimCandidate],
    policy: ReclaimPolicyName,
) -> list[ReclaimCandidate]:
    if policy == ReclaimPolicyName.ARKV_Q8_EVICT:
        return sorted(
            (candidate for candidate in candidates if candidate.reclaimable),
            key=lambda candidate: (0 if candidate.state == KVBlockState.QUANT else 1, candidate.score, candidate.storage_id),
        )
    return sorted(
        (candidate for candidate in candidates if candidate.reclaimable),
        key=lambda candidate: (candidate.score, candidate.storage_id),
    )


def select_blocks_to_evict(
    candidates: list[ReclaimCandidate],
    required_full_equiv: int,
    seq_states: dict[int, SequenceKVState],
    cfg: PolicyConfig,
) -> tuple[list[ReclaimCandidate], int]:
    """Select EVICT candidates, preferring already-QUANT low-score blocks."""
    if required_full_equiv < 0:
        raise PolicyError("required_full_equiv must be non-negative")
    if not cfg.allow_evict:
        raise PolicyInvariantError("EVICT selection is disabled")
    if not cfg.quality_gate_passed:
        raise PolicyInvariantError("EVICT selection requires a passed quality gate")
    if any(state == SequenceKVState.UNFINISHED_PREFILL for state in seq_states.values()):
        raise PolicyInvariantError("unfinished-prefill sequences cannot be EVICTed")

    selected: list[ReclaimCandidate] = []
    for candidate in _sort_reclaimable_candidates(candidates, ReclaimPolicyName.ARKV_Q8_EVICT):
        if candidate.protected_reasons:
            continue
        if candidate.state == KVBlockState.FULL and not cfg.allow_direct_full_evict:
            raise PolicyInvariantError("direct FULL->EVICT selected while disabled")
        if candidate.state not in {KVBlockState.QUANT, KVBlockState.FULL}:
            continue
        selected.append(candidate)
        if len(selected) >= required_full_equiv:
            break
    return selected, len(selected)
