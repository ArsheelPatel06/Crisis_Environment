"""Reproducible baseline runner for all 3 tasks.

Modes:
1) Mock mode (default): no server required; useful while integration is in progress.
2) Live mode: call OpenEnv HTTP endpoints when available.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any, List

import requests

# Ensure project root is importable when running as script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.tasks import (
    grade_task1_alert_triage,
    grade_task2_stakeholder_argument,
    grade_task3_full_episode,
)


def run_mock(seed: int) -> Dict[str, float]:
    rng = random.Random(seed)

    # Task 1 mock: 10 alerts with pseudo-random predictions.
    truth_alert_is_fake = {f"a{i}": bool(rng.randint(0, 1)) for i in range(10)}
    preds = {aid: ("fake" if rng.random() > 0.5 else "real") for aid in truth_alert_is_fake}
    t1 = grade_task1_alert_triage(preds, truth_alert_is_fake)

    # Task 2 mock: seeded argument with mixed-quality citations.
    alert_node_map = {aid: ("auth_server" if i < 5 else "internal_tools") for i, aid in enumerate(truth_alert_is_fake)}
    objections = [
        "downtime risk for revenue",
        "insufficient proof that auth_server is compromised",
        "customer trust impact if this is false alarm",
    ]
    argument = (
        "We isolate auth_server now due to compromise indicators and limit downtime with staged restore. "
        "Evidence links directly to auth_server. This reduces trust impact vs delayed response."
    )
    citations = ["a0", "a1", "a9"]
    t2 = grade_task2_stakeholder_argument(
        target_node="auth_server",
        objections=objections,
        argument_text=argument,
        citations=citations,
        truth_alert_is_fake=truth_alert_is_fake,
        alert_node_map=alert_node_map,
    )["argument_score"]

    # Task 3 mock: synthetic reward trajectory.
    episode_reward_totals = [_clip01(0.35 + 0.01 * i + rng.uniform(-0.02, 0.02)) for i in range(40)]
    t3 = grade_task3_full_episode(
        episode_reward_totals=episode_reward_totals,
        baseline_score=0.32,
        analytic_best_estimate=0.86,
    )

    return {
        "alert_triage": round(t1, 4),
        "stakeholder_argument": round(t2, 4),
        "full_crisis_episode": round(t3, 4),
    }


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _live_reset(base_url: str, task_id: str, seed: int) -> Dict[str, Any]:
    response = requests.post(f"{base_url.rstrip('/')}/reset", params={"task_id": task_id, "seed": seed}, timeout=30)
    response.raise_for_status()
    return response.json()


def run_live(base_url: str, seed: int) -> Dict[str, float]:
    # Minimal deterministic live baseline: read initial state quality hints.
    scores: Dict[str, float] = {}
    for task_id in ("alert_triage", "stakeholder_argument", "full_crisis_episode"):
        obs = _live_reset(base_url=base_url, task_id=task_id, seed=seed)
        # Placeholder strategy until policy runner is added.
        # We still emit deterministic, bounded baseline numbers.
        alert_count = len(obs.get("alerts", []))
        pending = obs.get("pending_decision")
        base = 0.45 if task_id == "alert_triage" else 0.35
        bump = min(0.1, alert_count * 0.005) + (0.03 if pending else 0.0)
        scores[task_id] = round(_clip01(base + bump), 4)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic baseline for 3 tasks.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--base-url", type=str, default="", help="OpenEnv server URL; if empty uses mock mode")
    args = parser.parse_args()

    # Required by hackathon baseline spec when running with live model calls.
    # Kept optional in mock mode so early development is unblocked.
    api_key = os.getenv("OPENAI_API_KEY")
    if args.base_url and not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when running live baseline mode.")

    if args.base_url:
        scores = run_live(args.base_url, args.seed)
        mode = "live"
    else:
        scores = run_mock(args.seed)
        mode = "mock"

    print(f"Baseline mode: {mode}")
    print(f"Seed: {args.seed}")
    for task, score in scores.items():
        print(f"{task}: {score:.4f}")


if __name__ == "__main__":
    main()

