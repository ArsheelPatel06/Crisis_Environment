from __future__ import annotations

import sys
from pathlib import Path
from random import Random

sys.path.append(str(Path(__file__).resolve().parents[1]))

from server.env import CyberCrisisEnv
from server.models import Action


def run_determinism_check(seed: int = 123, steps: int = 50) -> None:
    action_rng = Random(999)
    action_space = ["isolate", "patch", "monitor", "ignore", "communicate"]
    actions = [action_space[action_rng.randint(0, len(action_space) - 1)] for _ in range(steps)]

    env_a = CyberCrisisEnv(seed=0)
    env_b = CyberCrisisEnv(seed=0)

    obs_a = env_a.reset(seed=seed).model_dump()
    obs_b = env_b.reset(seed=seed).model_dump()
    assert obs_a == obs_b, "Reset observations differ for same seed"

    infection_path_a: list[int] = []
    infection_path_b: list[int] = []

    for action_name in actions:
        out_a = env_a.step(Action(action_type=action_name))
        out_b = env_b.step(Action(action_type=action_name))
        assert out_a == out_b, "Step outputs differ for same seed + same actions"
        infection_path_a.append(env_a.infected_nodes)
        infection_path_b.append(env_b.infected_nodes)

    assert infection_path_a == infection_path_b, "Infection progression differs"
    assert env_a.get_state() == env_b.get_state(), "Final state differs"

    print("Determinism check passed.")


if __name__ == "__main__":
    run_determinism_check()
