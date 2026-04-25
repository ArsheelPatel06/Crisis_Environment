from __future__ import annotations

import sys
from pathlib import Path
from random import Random

sys.path.append(str(Path(__file__).resolve().parents[1]))

from server.env import CyberCrisisEnv
from server.models import Action


def run_smoke_test(seed: int = 123, steps: int = 50) -> None:
    env = CyberCrisisEnv(seed=0)
    env.reset(seed=seed)

    rng = Random(seed)
    action_types = ["isolate", "patch", "monitor", "ignore", "communicate"]

    for i in range(1, steps + 1):
        action = Action(action_type=action_types[rng.randint(0, len(action_types) - 1)])
        result = env.step(action)
        reward_total = float(result["reward"]["total"])
        if not (0.0 <= reward_total <= 1.0):
            raise RuntimeError(f"Reward out of range at step {i}: {reward_total}")
        print(f"step={i}, reward={reward_total:.4f}")

    print("Smoke test passed: no crash, all rewards in [0,1].")


if __name__ == "__main__":
    run_smoke_test()
