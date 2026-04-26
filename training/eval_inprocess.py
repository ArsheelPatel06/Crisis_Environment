"""In-process task scores for baseline / demo curves (no HTTP)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.env import CyberCrisisEnv
from server.models import Action, TaskId
from server.tasks import get_task_spec, grade_task1_alert_triage, grade_task3_full_episode
from training.policy_heuristic import observation_to_task_action


@dataclass
class TaskScores:
    alert_triage: float
    stakeholder_argument: float
    full_crisis_episode: float


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _score_from_rollout(
    reward_history: List[float], final_info: Dict[str, Any]
) -> float:
    if "task_score" in final_info:
        return _clip01(final_info["task_score"])
    if reward_history:
        return _clip01(sum(reward_history) / len(reward_history))
    return 0.0


def run_episode_score(task_id: TaskId, seed: int, max_steps: int) -> float:
    env = CyberCrisisEnv(seed=seed, task_id=task_id)  # type: ignore[arg-type]
    obs = env.reset(seed=seed, task_id=task_id).model_dump()
    spec = get_task_spec(task_id)
    cap = min(max_steps, spec.max_steps)
    rh: List[float] = []
    final_info: Dict[str, Any] = {}
    step_i = 0
    while not bool(obs.get("done")) and step_i < cap:
        act = observation_to_task_action(task_id, obs, seed + step_i * 11)
        out = env.step(Action.model_validate(act))
        o = out["observation"]
        obs = o if isinstance(o, dict) else o.model_dump()
        r = out.get("reward", {})
        total = 0.0
        if isinstance(r, dict):
            total = float(r.get("total", 0.0))
        rh.append(_clip01(total))
        final_info = out.get("info", {}) or {}
        if not isinstance(final_info, dict):
            final_info = {}
        if out.get("done", False) or bool(obs.get("done")):
            break
        step_i += 1
    return _score_from_rollout(rh, final_info)


def eval_tasks_on_seeds(
    seeds: List[int],
    *,
    episodes_per_task: int = 1,
    max_steps: int = 64,
) -> TaskScores:
    acc: Dict[str, float] = {}
    count = max(1, len(seeds) * episodes_per_task)
    for task_id in ("alert_triage", "stakeholder_argument", "full_crisis_episode"):
        c = 0.0
        for seed in seeds:
            for ep in range(episodes_per_task):
                tseed = seed + ep * 97
                c += run_episode_score(
                    task_id,  # type: ignore[arg-type]
                    tseed,
                    max_steps,
                )
        acc[task_id] = _clip01(c / count)
    return TaskScores(
        alert_triage=acc["alert_triage"],
        stakeholder_argument=acc["stakeholder_argument"],
        full_crisis_episode=acc["full_crisis_episode"],
    )


def oracle_task1_perfect_score(seed: int) -> float:
    """In-process only: ground-truth from env state. Use as ceiling, not a deployable policy."""
    env = CyberCrisisEnv(seed=seed, task_id="alert_triage")
    env.reset(seed=seed, task_id="alert_triage")
    st = env.get_state()
    truth: Dict[str, bool] = dict(st.get("truth_alert_is_fake", {}))  # type: ignore[assignment]
    predictions = {k: ("fake" if v else "real") for k, v in truth.items()}
    return _clip01(grade_task1_alert_triage(predictions, truth))


def build_metrics_json(scores: TaskScores, path: Path, label: str = "heuristic_baseline") -> None:
    path.write_text(
        json.dumps(
            {label: asdict(scores), "seeds": "from_eval"},
            indent=2,
        ),
        encoding="utf-8",
    )
