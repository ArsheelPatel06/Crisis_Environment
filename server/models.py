from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


ActionType = Literal["monitor", "isolate", "restore", "allocate_engineer", "noop"]


class ResetResponse(BaseModel):
    observation: dict


class StepRequest(BaseModel):
    action_type: ActionType = "noop"
    target_node: str | None = None


class StepResponse(BaseModel):
    observation: dict
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: dict


class StateResponse(BaseModel):
    seed: int
    step_count: int
    done: bool
    score_security: float
    score_uptime: float


def _clip_01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class Observation(BaseModel):
    step: int
    system_health: float
    threat_level: float
    alerts: list[str]
    resources_available: int
    infection_ratio: float
    status: Literal["stable", "warning", "critical"]
    history_summary: dict[str, Any]

    @field_validator("system_health", "threat_level", "infection_ratio", mode="before")
    @classmethod
    def clip_observation_scores(cls, value: float) -> float:
        return _clip_01(value)


class Action(BaseModel):
    action_type: Literal["isolate", "patch", "monitor", "ignore", "communicate"]
    target: str | None = None


class Reward(BaseModel):
    security_score: float
    uptime_score: float
    trust_score: float
    speed_score: float
    total: float

    @field_validator(
        "security_score",
        "uptime_score",
        "trust_score",
        "speed_score",
        "total",
        mode="before",
    )
    @classmethod
    def clip_reward_scores(cls, value: float) -> float:
        return _clip_01(value)
