"""Deterministic task graders for the cyber crisis environment.

This module is intentionally light on framework dependencies so it can be
used in unit tests before the full OpenEnv server is wired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Literal


TaskId = Literal["alert_triage", "stakeholder_argument", "full_crisis_episode"]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class TaskSpec:
    id: TaskId
    difficulty: str
    max_steps: int
    description: str


TASK_REGISTRY: Dict[TaskId, TaskSpec] = {
    "alert_triage": TaskSpec(
        id="alert_triage",
        difficulty="easy",
        max_steps=1,
        description="Classify 10 mixed alerts as real/fake.",
    ),
    "stakeholder_argument": TaskSpec(
        id="stakeholder_argument",
        difficulty="medium",
        max_steps=5,
        description="Submit evidence-backed argument for high-impact approval.",
    ),
    "full_crisis_episode": TaskSpec(
        id="full_crisis_episode",
        difficulty="hard",
        max_steps=60,
        description="Manage full multi-step crisis with deceptive alerts.",
    ),
}


def get_task_spec(task_id: str) -> TaskSpec:
    try:
        return TASK_REGISTRY[task_id]  # type: ignore[index]
    except KeyError as exc:
        valid = ", ".join(sorted(TASK_REGISTRY))
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {valid}") from exc


def grade_task1_alert_triage(
    predictions: Mapping[str, str],
    truth_alert_is_fake: Mapping[str, bool],
) -> float:
    """Deterministic score in [0,1] for alert triage.

    predictions maps alert_id -> "real"|"fake"
    truth_alert_is_fake maps alert_id -> bool (True means fake)
    """
    if not truth_alert_is_fake:
        return 0.0

    correct = 0
    total = len(truth_alert_is_fake)
    for alert_id, is_fake in truth_alert_is_fake.items():
        pred = predictions.get(alert_id)
        if pred not in ("real", "fake"):
            continue
        pred_is_fake = pred == "fake"
        if pred_is_fake == bool(is_fake):
            correct += 1

    return _clip01(correct / total)


def _coverage_from_objections(text: str, objections: List[str]) -> float:
    if not objections:
        return 0.0
    lowered = text.lower()
    theme_keywords = {
        "downtime": {"downtime", "availability", "outage", "revenue", "sla"},
        "proof": {"proof", "evidence", "indicator", "alert", "telemetry", "compromise"},
        "trust": {"trust", "customer", "reputation", "confidence", "false", "alarm"},
        "speed": {"delay", "urgent", "immediate", "quick", "contain"},
    }
    covered = 0
    for objection in objections:
        objection_lower = objection.lower().replace("_", " ")
        matched_theme = False
        for theme, keywords in theme_keywords.items():
            if theme in objection_lower and any(keyword in lowered for keyword in keywords):
                matched_theme = True
                break

        # Fallback for arbitrary objection templates outside known themes.
        tokens = [tok for tok in objection_lower.split() if len(tok) > 4]
        if matched_theme or any(token in lowered for token in tokens):
            covered += 1
    return covered / len(objections)


def _consistency_score(text: str) -> float:
    lowered = text.lower()
    contradiction_pairs = [
        ("isolate", "do not isolate"),
        ("breach confirmed", "no breach"),
        ("urgent", "wait indefinitely"),
    ]
    for a, b in contradiction_pairs:
        if a in lowered and b in lowered:
            return 0.0
    return 1.0


def grade_task2_stakeholder_argument(
    *,
    target_node: str,
    objections: List[str],
    argument_text: str,
    citations: List[str],
    truth_alert_is_fake: Mapping[str, bool],
    alert_node_map: Mapping[str, str],
) -> Dict[str, Any]:
    """Deterministic argument rubric score for stakeholder debate.

    Returns:
      {
        "argument_score": float in [0,1],
        "stance": "approve"|"delay"|"block",
        "subscores": {...}
      }
    """
    if not citations:
        evidence_accuracy = 0.0
    else:
        good = 0
        for alert_id in citations:
            is_real = alert_id in truth_alert_is_fake and not truth_alert_is_fake[alert_id]
            is_relevant = alert_node_map.get(alert_id) == target_node
            if is_real and is_relevant:
                good += 1
        evidence_accuracy = good / len(citations)

    objection_coverage = _coverage_from_objections(argument_text, objections)
    consistency = _consistency_score(argument_text)
    brevity_penalty = max(0.0, (len(argument_text) - 600) / 600.0)

    score = _clip01(
        0.4 * evidence_accuracy
        + 0.3 * objection_coverage
        + 0.3 * consistency
        - 0.1 * brevity_penalty
    )

    if score >= 0.6:
        stance = "approve"
    elif score >= 0.3:
        stance = "delay"
    else:
        stance = "block"

    return {
        "argument_score": score,
        "stance": stance,
        "subscores": {
            "evidence_accuracy": _clip01(evidence_accuracy),
            "objection_coverage": _clip01(objection_coverage),
            "consistency": _clip01(consistency),
            "brevity_penalty": max(0.0, float(brevity_penalty)),
        },
    }


def grade_task3_full_episode(
    *,
    episode_reward_totals: List[float],
    baseline_score: float,
    analytic_best_estimate: float,
) -> float:
    """Normalized full-episode score in [0,1] against baseline."""
    if not episode_reward_totals:
        return 0.0
    raw = sum(float(x) for x in episode_reward_totals) / len(episode_reward_totals)
    denom = float(analytic_best_estimate) - float(baseline_score) + 1e-8
    normalized = (raw - float(baseline_score)) / denom
    return _clip01(normalized)

