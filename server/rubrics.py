from __future__ import annotations

from typing import Any, Mapping


def clamp_reward(value: float) -> float:
    """Clamp reward to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def _read_01(state: Mapping[str, Any], key: str, default: float) -> float:
    return clamp_reward(float(state.get(key, default)))


def compute_security_score(state: Mapping[str, Any]) -> float:
    infection_ratio = _read_01(state, "infection_ratio", 0.0)
    return clamp_reward(1.0 - infection_ratio)


def compute_uptime_score(state: Mapping[str, Any]) -> float:
    return _read_01(state, "system_health", 0.0)


def compute_speed_score(state: Mapping[str, Any]) -> float:
    time_pressure = max(0, int(state.get("time_pressure", 0)))
    return clamp_reward(1.0 - (time_pressure / 50.0))


def aggregate_reward(state: Mapping[str, Any]) -> dict:
    security = compute_security_score(state)
    uptime = compute_uptime_score(state)
    speed = compute_speed_score(state)

    total = 0.5 * security + 0.3 * uptime + 0.2 * speed
    total = clamp_reward(total)

    return {
        "security_score": security,
        "uptime_score": uptime,
        "speed_score": speed,
        "total": total,
    }
