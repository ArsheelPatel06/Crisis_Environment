"""Compare in-process and HTTP (local Space) scores for the same seed bank.

Usage (server running on 7860):
  python -m training.eval_http --base-url http://127.0.0.1:7860
  python -m training.eval_http --base-url https://your-user.hf.space
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.baseline import run_live
from training.eval_inprocess import TaskScores, eval_tasks_on_seeds
from training.train_unsloth_grpo import SEED_LIST_DEFAULT

RESULTS = PROJECT_ROOT / "results" / "deploy_parity.json"


def _inprocess() -> dict[str, float]:
    s: TaskScores = eval_tasks_on_seeds(list(SEED_LIST_DEFAULT), episodes_per_task=1, max_steps=64)
    return {
        "alert_triage": s.alert_triage,
        "stakeholder_argument": s.stakeholder_argument,
        "full_crisis_episode": s.full_crisis_episode,
    }


def _http(base_url: str) -> dict[str, float] | None:
    try:
        return run_live(base_url, seed=42, episodes=3, max_steps=64)
    except (RequestsConnectionError, OSError, requests.exceptions.RequestException) as e:
        print("HTTP eval failed:", e, file=sys.stderr)
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-url",
        type=str,
        default="",
        help="OpenEnv / Space URL. If empty, in-process + placeholder only.",
    )
    args = p.parse_args()
    a = _inprocess()
    b: dict[str, Any] | None = _http(args.base_url) if args.base_url else None
    out = {
        "inprocess_heuristic": a,
        "http": b,
        "seeds": list(SEED_LIST_DEFAULT),
        "note": "Differences small delta expected from floating formatting; large gaps indicate API drift.",
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", RESULTS)
    if b is not None:
        for k in a:
            print(f"{k}: inprocess={a[k]:.4f} http={b.get(k, 0.0):.4f}")


if __name__ == "__main__":
    main()
