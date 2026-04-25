import requests
import random

BASE_URL = "http://localhost:8000"

def reset(seed=42):
    requests.post(f"{BASE_URL}/reset", json={"seed": seed})

def step(action_type):
    return requests.post(f"{BASE_URL}/step", json={
        "action_type": action_type,
        "target": None
    }).json()

def run_strategy(name, action_fn, use_obs=False):
    print(f"\n===== {name} =====")

    reset(seed=42)

    rewards = []
    obs = None

    for i in range(50):
        # Decide action
        if use_obs:
            action = action_fn(i, obs)
        else:
            action = action_fn(i)

        res = step(action)

        obs = res["observation"]
        reward = res["reward"]

        rewards.append(reward["total"])

        print(
            f"Step {i} | "
            f"Infection: {obs['infection_ratio']:.2f} | "
            f"Health: {obs['system_health']:.2f} | "
            f"Reward: {reward['total']:.2f}"
        )

    avg_reward = sum(rewards) / len(rewards)
    print(f"\nAverage Reward: {avg_reward:.3f}")


# 🔴 Strategy 1: Always Ignore
run_strategy("ALWAYS IGNORE", lambda i: "ignore")

# 🟢 Strategy 2: Always Isolate
run_strategy("ALWAYS ISOLATE", lambda i: "isolate")

# 🟡 Strategy 3: Random
run_strategy("RANDOM", lambda i: random.choice([
    "ignore", "isolate", "patch", "monitor", "communicate"
]))


# 🔵 Strategy 4: Smart Policy (NEW)
def smart_policy(i, obs):
    if obs is None:
        return "monitor"

    inf = obs["infection_ratio"]
    health = obs["system_health"]

    # High infection → act aggressively
    if inf > 0.5:
        return "isolate"

    # Medium infection → reduce future spread
    if inf > 0.2:
        return "patch"

    # Low infection and healthy → monitor
    if inf <= 0.2 and health > 0.5:
        return "monitor"

    # Occasionally communicate (not spam)
    if i % 7 == 0:
        return "communicate"

    return "monitor"


run_strategy("SMART POLICY", smart_policy, use_obs=True)