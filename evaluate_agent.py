from __future__ import annotations

import json
import random
from typing import Callable, Dict, Tuple

from server.env import CyberCrisisEnv
from server.models import Action

ACTIONS = ["ignore", "isolate", "patch", "monitor", "communicate"]
POLICY_PATH = "policy.json"
EVAL_EPISODES = 1
STEPS_PER_EPISODE = 50


class StatePolicy:
    def __init__(self) -> None:
        self.table: Dict[Tuple[float, float], Dict[str, float]] = {}

    def get_state_key(self, obs: dict) -> Tuple[float, float]:
        infection = round(float(obs["infection_ratio"]), 1)
        health = round(float(obs["system_health"]), 1)
        return (infection, health)

    def select_action(self, obs: dict) -> str:
        state_key = self.get_state_key(obs)
        if state_key not in self.table:
            return random.choice(ACTIONS)
        action_weights = self.table[state_key]
        actions = list(action_weights.keys())
        weights = list(action_weights.values())
        return random.choices(actions, weights=weights, k=1)[0]

    @staticmethod
    def from_json(path: str = POLICY_PATH) -> "StatePolicy":
        policy = StatePolicy()
        with open(path, "r", encoding="utf-8") as f:
            raw_table = json.load(f)

        for key, action_weights in raw_table.items():
            infection_str, health_str = key.split("|")
            state_key = (float(infection_str), float(health_str))
            policy.table[state_key] = {action: float(weight) for action, weight in action_weights.items()}
        return policy


def smart_policy(obs: dict) -> str:
    infection = float(obs["infection_ratio"])

    if infection > 0.5:
        return "isolate"
    if infection > 0.2:
        return "patch"
    return "monitor"


def evaluate_strategy(action_fn: Callable[[dict], str], seed: int) -> float:
    env = CyberCrisisEnv(seed=42)
    episode_totals = []

    for episode in range(EVAL_EPISODES):
        obs = env.reset(seed=seed + episode).model_dump()
        total_reward = 0.0

        for _ in range(STEPS_PER_EPISODE):
            action = action_fn(obs)
            result = env.step(Action(action_type=action))
            obs = result["observation"]
            total_reward += float(result["reward"]["total"])
            if bool(result["done"]):
                break

        episode_totals.append(total_reward)

    return sum(episode_totals) / len(episode_totals)


if __name__ == "__main__":
    random.seed(42)
    trained_policy = StatePolicy.from_json(POLICY_PATH)

    random_avg = evaluate_strategy(lambda obs: random.choice(ACTIONS), seed=42)
    smart_avg = evaluate_strategy(smart_policy, seed=42)
    trained_avg = evaluate_strategy(trained_policy.select_action, seed=42)

    print(f"Random Avg Reward: {random_avg:.4f}")
    print(f"Smart Avg Reward: {smart_avg:.4f}")
    print(f"Trained Avg Reward: {trained_avg:.4f}")
