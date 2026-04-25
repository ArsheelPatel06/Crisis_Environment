from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Query
from pydantic import BaseModel

from server.env import CyberCrisisEnv
from server.models import Action, Observation, Reward


app = FastAPI(title="OpenEnv Cyber Crisis Simulator")
env = CyberCrisisEnv(seed=0)


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


@app.post("/reset", response_model=Observation)
def reset(seed: int = Query(default=0)) -> Observation:
    observation = env.reset(seed=seed)
    return observation


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    result = env.step(action)
    return StepResponse(**result)


@app.get("/state")
def state() -> dict[str, Any]:
    return env.get_state()
