from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


TaskId = Literal["alert_triage", "stakeholder_argument", "full_crisis_episode"]


def _clip_01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _default_stakeholder_trust_scores() -> Dict[str, float]:
    return {"Finance": 0.7, "Engineering": 0.7, "PR": 0.7}


class Observation(BaseModel):
    task_id: TaskId
    step: int
    done: bool = False

    # Back-compat fields used by older baselines/tests
    system_health: float = Field(ge=0.0, le=1.0)
    threat_level: float = Field(ge=0.0, le=1.0)
    infection_ratio: float = Field(ge=0.0, le=1.0)
    status: Literal["stable", "warning", "critical"] = "stable"
    resources_available: int = 0
    history_summary: dict[str, Any] = Field(default_factory=dict)

    # Rich fields for the cyber-crisis simulator
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    pending_debate: Optional[Dict[str, Any]] = None
    task_score: float = Field(default=0.0, ge=0.0, le=1.0)

    stakeholder_messages: List[str] = Field(default_factory=list)
    stakeholder_trust_scores: Dict[str, float] = Field(default_factory=_default_stakeholder_trust_scores)

    @field_validator("system_health", "threat_level", "infection_ratio", mode="before")
    @classmethod
    def clip_scores(cls, value: float) -> float:
        return _clip_01(float(value))


class Action(BaseModel):
    action_type: Literal["isolate", "patch", "monitor", "ignore", "communicate", "noop"]
    target: Optional[str] = None

    # Optional fields for debate / richer policies
    argument_text: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    classifications: Optional[Dict[str, Literal["real", "fake"]]] = None


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
        return _clip_01(float(value))
