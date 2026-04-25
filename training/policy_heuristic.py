"""Shared deterministic heuristic policy for in-process, HTTP, and SFT data generation."""

from __future__ import annotations

import random
from typing import Any, Dict, List

from server.models import Action


def normalize_alerts(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        elif isinstance(alert, str):
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


def classify_alerts_heuristic(observation: Dict[str, Any], rng: random.Random) -> Dict[str, str]:
    classifications: Dict[str, str] = {}
    for alert in normalize_alerts(observation):
        alert_id = alert.get("id")
        if not alert_id:
            continue
        severity = int(alert.get("severity", 1))
        kind = str(alert.get("kind", "")).lower()
        signature = str(alert.get("signature", "")).lower()
        likely_fake = (severity <= 2 and ("scan" in kind or "noise" in kind or "heartbeat" in signature)) or (
            rng.random() < 0.15
        )
        classifications[alert_id] = "fake" if likely_fake else "real"
    return classifications


def _format_action(
    action_type: str, target: str = "auth_server", **kwargs: Any
) -> Dict[str, Any]:
    d: Dict[str, Any] = {"action_type": action_type, "target": target}
    d.update({k: v for k, v in kwargs.items() if v is not None})
    return d


def choose_task1_action_dict(observation: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    classifications = classify_alerts_heuristic(observation, rng)
    if not classifications:
        return _format_action("monitor", target="auth_server")
    return {
        "action_type": "monitor",
        "target": "auth_server",
        "classifications": classifications,
    }


def choose_task2_action_dict(observation: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Match env: first open debate with isolate/patch/monitor; when pending, communicate with argument."""
    del rng  # deterministic for same obs path; kept for API compatibility
    pending = observation.get("pending_debate")
    if pending is not None and isinstance(pending, dict):
        alist = normalize_alerts(observation)
        cits = [a["id"] for a in alist[:3]] if alist else ["a0", "a1", "a2"]
        return {
            "action_type": "communicate",
            "target": "auth_server",
            "argument_text": (
                "We should isolate auth_server: correlated auth anomalies and lateral movement signals meet "
                "our escalation bar. Citing the listed alert ids ties directly to the isolation decision. "
                "Staged recovery limits downtime and preserves customer trust."
            ),
            "citations": cits,
        }
    if float(observation.get("threat_level", 0.0)) >= 0.55 or str(
        observation.get("status", "stable")
    ).lower() in ("critical", "warning"):
        return _format_action("isolate", target="auth_server")
    return _format_action("monitor", target="auth_server")


def choose_task3_action_dict(observation: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    alerts = normalize_alerts(observation)
    if alerts:
        highest = max(alerts, key=lambda a: int(a.get("severity", 1)))
        node = str(highest.get("node", "auth_server"))
        if int(highest.get("severity", 1)) >= 4 and rng.random() < 0.7:
            return _format_action("isolate", target=node)
        if rng.random() < 0.5:
            return _format_action("monitor", target=node)
        return _format_action("patch", target=node)
    fallback = ["api_gateway", "internal_tools", "auth_server", "comms"]
    return _format_action("monitor", target=rng.choice(fallback))


def observation_to_task_action(
    task_id: str, observation: Dict[str, Any], seed: int
) -> Dict[str, Any]:
    off = {"alert_triage": 101, "stakeholder_argument": 202, "full_crisis_episode": 303}
    rng = random.Random(seed + off.get(task_id, 0))
    if task_id == "alert_triage":
        return choose_task1_action_dict(observation, rng)
    if task_id == "stakeholder_argument":
        return choose_task2_action_dict(observation, rng)
    return choose_task3_action_dict(observation, rng)


def dict_to_action(d: Dict[str, Any]) -> Action:
    return Action.model_validate(d)
