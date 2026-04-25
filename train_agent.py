from __future__ import annotations

import json
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from server.env import CyberCrisisEnv
from server.models import Action

ACTIONS = ["ignore", "isolate", "patch", "monitor", "communicate"]
EPISODES = 100
STEPS_PER_EPISODE = 50
POLICY_PATH = "policy.json"
REWARD_CURVE_PATH = "reward_curve.png"


class FastLocalEnv:
    """
    Thin local wrapper to keep train loop simple.
    step(action) returns (observation_dict, reward_dict, done)
    """

    def __init__(self, seed: int = 42) -> None:
        self.env = CyberCrisisEnv(seed=seed)

    def reset(self, seed: int) -> dict:
        return self.env.reset(seed=seed).model_dump()

    def step(self, action: str) -> tuple[dict, dict, bool]:
        result = self.env.step(Action(action_type=action))
        return result["observation"], result["reward"], bool(result["done"])


class StatePolicy:
    def __init__(self) -> None:
        # table[state_key] = {action: weight}
        self.table: Dict[Tuple[float, float], Dict[str, float]] = {}

    def get_state_key(self, obs: dict) -> Tuple[float, float]:
        infection = round(float(obs["infection_ratio"]), 1)
        health = round(float(obs["system_health"]), 1)
        return (infection, health)

    def _ensure_state(self, state_key: Tuple[float, float]) -> None:
        if state_key not in self.table:
            self.table[state_key] = {action: 1.0 for action in ACTIONS}

    def select_action(self, obs: dict) -> str:
        state_key = self.get_state_key(obs)
        self._ensure_state(state_key)
        action_weights = self.table[state_key]
        actions = list(action_weights.keys())
        weights = list(action_weights.values())
        return random.choices(actions, weights=weights, k=1)[0]

    def update(self, obs: dict, action: str, reward: float) -> None:
        state_key = self.get_state_key(obs)
        self._ensure_state(state_key)
        self.table[state_key][action] += 0.2 * (float(reward) - 0.5)
        self.table[state_key][action] = max(0.1, min(self.table[state_key][action], 10.0))

    def decay_weights(self) -> None:
        for state_key in self.table:
            for action in ACTIONS:
                self.table[state_key][action] *= 0.995
                self.table[state_key][action] = max(0.1, min(self.table[state_key][action], 10.0))

    def to_serializable(self) -> dict:
        serializable = {}
        for (infection, health), action_weights in self.table.items():
            key = f"{infection:.1f}|{health:.1f}"
            serializable[key] = action_weights
        return serializable


def save_policy(policy: StatePolicy, path: str = POLICY_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(policy.to_serializable(), f, indent=2)


def plot_rewards(episode_rewards: List[float], output_path: str = REWARD_CURVE_PATH) -> None:
    episodes = list(range(1, len(episode_rewards) + 1))
    window_size = 10
    smoothed_rewards = np.convolve(
        episode_rewards,
        np.ones(window_size) / window_size,
        mode="valid",
    )
    smoothed_episodes = list(range(window_size, len(episode_rewards) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, episode_rewards, alpha=0.3, label="Raw Reward")
    plt.plot(smoothed_episodes, smoothed_rewards, linewidth=2, label="Smoothed Reward")
    plt.title("Training Reward Curve (Raw vs Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def choose_action(policy: StatePolicy, obs: dict, epsilon: float) -> str:
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return policy.select_action(obs)


def train() -> tuple[StatePolicy, List[float]]:
    random.seed(42)
    env = FastLocalEnv(seed=42)
    policy = StatePolicy()
    episode_rewards: List[float] = []

    for episode in range(EPISODES):
        obs = env.reset(seed=42 + episode)
        total_reward = 0.0
        epsilon = max(0.03, 0.3 - episode * 0.005)

        for _ in range(STEPS_PER_EPISODE):
            action = choose_action(policy, obs, epsilon)
            next_obs, reward, done = env.step(action)
            reward_total = float(reward["total"])

            policy.update(obs, action, reward_total)
            total_reward += reward_total
            obs = next_obs

            if done:
                break

        policy.decay_weights()
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1} | Reward: {total_reward:.4f}")

    return policy, episode_rewards


if __name__ == "__main__":
    trained_policy, rewards = train()
    save_policy(trained_policy, POLICY_PATH)
    plot_rewards(rewards, REWARD_CURVE_PATH)

    last_10 = rewards[-10:]
    avg_last_10 = sum(last_10) / len(last_10) if last_10 else 0.0

    print("\nLast 10 episode rewards:")
    for idx, value in enumerate(last_10, start=max(1, len(rewards) - 9)):
        print(f"Episode {idx} | Reward: {value:.4f}")

    print("\nLast 10 rewards:", last_10)
    print("Average (last 10):", avg_last_10)
