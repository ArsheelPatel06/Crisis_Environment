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
from typing import Dict, Any, List, Tuple

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError

# Ensure project root is importable when running as script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.tasks import (
    grade_task1_alert_triage,
    grade_task2_stakeholder_argument,
    grade_task3_full_episode,
)

RESULTS_DIR = PROJECT_ROOT / "results"


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


def _extract_observation(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "observation" in payload:
        observation = payload["observation"]
        if isinstance(observation, dict):
            return observation
    return payload


def _live_step(base_url: str, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
    response = requests.post(f"{base_url.rstrip('/')}/step", json=action, timeout=30)
    response.raise_for_status()
    payload = response.json()
    observation = _extract_observation(payload)

    reward_payload = payload.get("reward", {})
    if isinstance(reward_payload, dict):
        reward_total = float(reward_payload.get("total", reward_payload.get("task_score", 0.0)))
    else:
        reward_total = float(reward_payload or 0.0)

    done = bool(payload.get("done", observation.get("done", False)))
    info = payload.get("info", {})
    if not isinstance(info, dict):
        info = {}
    return observation, reward_total, done, info


def _normalize_alerts(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    raw_alerts = observation.get("alerts", [])
    if not isinstance(raw_alerts, list):
        return normalized

    for idx, alert in enumerate(raw_alerts):
        if isinstance(alert, dict):
            alert_id = str(alert.get("id", f"alert_{idx}"))
            normalized.append(
                {
                    "id": alert_id,
                    "node": str(alert.get("node", "auth_server")),
                    "severity": int(alert.get("severity", 3)),
                    "kind": str(alert.get("kind", "")),
                    "signature": str(alert.get("signature", "")),
                }
            )
            continue

        if isinstance(alert, str):
            sev = 3
            upper = alert.upper()
            if "CRITICAL" in upper:
                sev = 5
            elif "WARNING" in upper:
                sev = 4
            elif "NORMAL" in upper:
                sev = 2
            normalized.append(
                {
                    "id": f"alert_{idx}",
                    "node": "auth_server",
                    "severity": sev,
                    "kind": upper.lower(),
                    "signature": upper.lower(),
                }
            )
    return normalized


def _format_action(action_type: str, target: str = "auth_server") -> Dict[str, Any]:
    # Current env-core expects this action schema.
    return {"action_type": action_type, "target": target}


def _score_from_rollout(
    reward_history: List[float], final_observation: Dict[str, Any], final_info: Dict[str, Any]
) -> float:
    if "task_score" in final_info:
        return _clip01(final_info["task_score"])

    reward_dict = final_info.get("reward", {})
    if isinstance(reward_dict, dict) and "task_score" in reward_dict:
        return _clip01(reward_dict["task_score"])

    if reward_history:
        return _clip01(sum(reward_history) / len(reward_history))

    return _clip01(final_observation.get("task_score", 0.0))


def _choose_task1_action(observation: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    classifications: Dict[str, str] = {}
    for alert in _normalize_alerts(observation):
        alert_id = alert.get("id")
        if not alert_id:
            continue
        severity = int(alert.get("severity", 1))
        kind = str(alert.get("kind", "")).lower()
        signature = str(alert.get("signature", "")).lower()
        # Deterministic, weak-but-reasonable baseline heuristic.
        likely_fake = (
            severity <= 2
            and ("scan" in kind or "noise" in kind or "heartbeat" in signature)
        ) or (rng.random() < 0.15)
        classifications[alert_id] = "fake" if likely_fake else "real"

    # Current server action model does not support classify_alerts yet.
    # Use monitor/ignore proxy actions based on fake ratio.
    fake_ratio = (sum(1 for v in classifications.values() if v == "fake") / max(1, len(classifications)))
    if fake_ratio > 0.5:
        return _format_action("monitor")
    return _format_action("patch")


def _choose_task2_action(observation: Dict[str, Any], requested: bool) -> Dict[str, Any]:
    threat = float(observation.get("threat_level", 0.0))
    status = str(observation.get("status", "stable")).lower()
    if threat >= 0.55 or status == "critical":
        return _format_action("isolate")
    if requested:
        return _format_action("communicate")
    return _format_action("monitor")


def _choose_task3_action(observation: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    alerts = _normalize_alerts(observation)
    if alerts:
        # Prioritize highest-severity node.
        highest = max(alerts, key=lambda a: int(a.get("severity", 1)))
        node = highest.get("node")
        if node:
            if int(highest.get("severity", 1)) >= 4 and rng.random() < 0.7:
                return _format_action("isolate", target=node)
            if rng.random() < 0.5:
                return _format_action("monitor", target=node)
            return _format_action("patch", target=node)

    # Small deterministic exploration baseline.
    fallback_nodes = ["api_gateway", "internal_tools", "auth_server", "comms"]
    return _format_action("monitor", target=rng.choice(fallback_nodes))


def _run_task_live(base_url: str, task_id: str, seed: int, max_steps: int) -> float:
    observation = _live_reset(base_url=base_url, task_id=task_id, seed=seed)
    done = bool(observation.get("done", False))
    reward_history: List[float] = []
    final_info: Dict[str, Any] = {}
    requested_approval = False

    step_idx = 0
    task_offsets = {
        "alert_triage": 101,
        "stakeholder_argument": 202,
        "full_crisis_episode": 303,
    }
    rng = random.Random(seed + task_offsets.get(task_id, 0))
    while not done and step_idx < max_steps:
        if task_id == "alert_triage":
            action = _choose_task1_action(observation, rng)
        elif task_id == "stakeholder_argument":
            action = _choose_task2_action(observation, requested=requested_approval)
            if action.get("kind") == "request_approval":
                requested_approval = True
        else:
            action = _choose_task3_action(observation, rng)

        observation, reward_total, done, info = _live_step(base_url, action)
        reward_history.append(_clip01(reward_total))
        final_info = info
        step_idx += 1

    return _score_from_rollout(reward_history, observation, final_info)


def run_live(base_url: str, seed: int, episodes: int, max_steps: int) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    task_ids = ("alert_triage", "stakeholder_argument", "full_crisis_episode")
    for task_id in task_ids:
        task_scores = []
        for ep in range(episodes):
            task_seed = seed + ep * 97
            task_scores.append(_run_task_live(base_url, task_id, task_seed, max_steps=max_steps))
        scores[task_id] = round(_clip01(sum(task_scores) / len(task_scores)), 4)
    return scores


def write_baseline_report(
    *,
    mode: str,
    seed: int,
    scores: Dict[str, float],
    base_url: str = "",
    episodes: int = 0,
    max_steps: int = 0,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "baseline_report.md"
    lines = [
        "# Baseline Report",
        "",
        f"- Mode: `{mode}`",
        f"- Seed: `{seed}`",
    ]
    if mode == "live":
        lines.extend(
            [
                f"- Base URL: `{base_url}`",
                f"- Episodes per task: `{episodes}`",
                f"- Max steps: `{max_steps}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Scores",
            f"- `alert_triage`: {scores['alert_triage']:.4f}",
            f"- `stakeholder_argument`: {scores['stakeholder_argument']:.4f}",
            f"- `full_crisis_episode`: {scores['full_crisis_episode']:.4f}",
            "",
            "## Notes",
            "- Scores are deterministic for the same seed and config.",
            "- Replace heuristic live baseline with LLM policy calls as integration matures.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic baseline for 3 tasks.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--base-url", type=str, default="", help="OpenEnv server URL; if empty uses mock mode")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per task in live mode")
    parser.add_argument("--max-steps", type=int, default=60, help="Step cap per episode in live mode")
    parser.add_argument(
        "--fallback-to-mock",
        action="store_true",
        help="If live server is unreachable, run mock baseline instead of raising.",
    )
    args = parser.parse_args()

    # Required by hackathon baseline spec when running with live model calls.
    # Kept optional in mock mode so early development is unblocked.
    api_key = os.getenv("OPENAI_API_KEY")
    if args.base_url and not api_key:
        print("Warning: OPENAI_API_KEY missing; running heuristic live baseline without model calls.")

    if args.base_url:
        try:
            scores = run_live(args.base_url, args.seed, episodes=args.episodes, max_steps=args.max_steps)
            mode = "live"
        except RequestsConnectionError as exc:
            if not args.fallback_to_mock:
                raise RuntimeError(
                    "Could not connect to live env server.\n"
                    f"Tried: {args.base_url}\n"
                    "Start the server first (example: uvicorn server.main:app --host 0.0.0.0 --port 7860)\n"
                    "or rerun with --fallback-to-mock."
                ) from exc
            print(
                "Live server unreachable; switching to mock mode "
                "(use --base-url with running server for true live rollout)."
            )
            scores = run_mock(args.seed)
            mode = "mock"
    else:
        scores = run_mock(args.seed)
        mode = "mock"

    print(f"Baseline mode: {mode}")
    print(f"Seed: {args.seed}")
    if mode == "live":
        print(f"Episodes per task: {args.episodes}")
        print(f"Max steps: {args.max_steps}")
    for task, score in scores.items():
        print(f"{task}: {score:.4f}")

    report_path = write_baseline_report(
        mode=mode,
        seed=args.seed,
        scores=scores,
        base_url=args.base_url,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()

