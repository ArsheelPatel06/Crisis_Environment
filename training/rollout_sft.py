"""Log heuristic rollouts for SFT: (chat messages) -> target Action JSON, one line per turn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.env import CyberCrisisEnv
from server.models import Action, TaskId
from server.tasks import get_task_spec
from training.policy_heuristic import observation_to_task_action
from training.prompts import action_to_json_str, build_chat_messages

# Same seed range as README / demo; extend as needed
DEFAULT_SEED_BANK = (42, 84, 126, 7, 2025, 1, 99, 10)


def _run_episode(
    task_id: TaskId, seed: int, max_steps: int
) -> List[Tuple[Dict[str, Any], Action]]:
    env = CyberCrisisEnv(seed=seed, task_id=task_id)  # type: ignore[arg-type]
    obs = env.reset(seed=seed, task_id=task_id).model_dump()
    records: List[Tuple[Dict[str, Any], Action]] = []
    spec = get_task_spec(task_id)
    step_cap = min(max_steps, spec.max_steps)
    step_i = 0
    while not bool(obs.get("done")) and step_i < step_cap:
        act_dict = observation_to_task_action(task_id, obs, seed + step_i * 11)
        action = Action.model_validate(act_dict)
        records.append((dict(obs), action))
        out = env.step(action)
        o = out["observation"]
        obs = o if isinstance(o, dict) else o.model_dump()  # type: ignore[union-attr]
        if bool(out.get("done", False)) or bool(obs.get("done")):
            break
        step_i += 1
    return records


def make_jsonl_line(task_id: TaskId, obs_dict: Dict[str, Any], target: Action) -> str:
    messages = build_chat_messages(task_id, obs_dict)
    assistant = action_to_json_str(target)
    return json.dumps(
        {
            "messages": messages
            + [{"role": "assistant", "content": assistant}],
            "metadata": {
                "task_id": task_id,
                "target_action": json.loads(assistant) if assistant.startswith("{") else {},
            },
        },
        ensure_ascii=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "sft_rollout.jsonl",
        help="Output path for JSONL (chat format).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEED_BANK),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="alert_triage,stakeholder_argument,full_crisis_episode",
    )
    parser.add_argument("--max-steps", type=int, default=64)
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    task_ids: List[TaskId] = [t.strip() for t in args.tasks.split(",") if t.strip()]  # type: ignore[list-item]
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    with out_path.open("w", encoding="utf-8") as f:
        for task_id in task_ids:
            for seed in seeds:
                pairs = _run_episode(task_id, seed, max_steps=args.max_steps)
                for obs_dict, action in pairs:
                    f.write(make_jsonl_line(task_id, obs_dict, action) + "\n")
                    n_lines += 1
    print(f"Wrote {n_lines} JSONL lines to {out_path}")


if __name__ == "__main__":
    main()
