from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Query
from pydantic import BaseModel

from server.env import CyberCrisisEnv
from server.models import Action, Observation, Reward, TaskId


app = FastAPI(title="OpenEnv Cyber Crisis Simulator", version="0.1.0")
env = CyberCrisisEnv(seed=0, task_id="full_crisis_episode")


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


@app.post("/reset", response_model=Observation)
def reset(seed: int = Query(default=0), task_id: TaskId = Query(default="full_crisis_episode")) -> Observation:
    observation = env.reset(seed=seed, task_id=task_id)
    return observation


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    result = env.step(action)
    return StepResponse(**result)


@app.get("/state")
def state() -> dict[str, Any]:
    return env.get_state()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return {
        "name": "cyber-crisis-env",
        "description": "Cyber crisis incident-response environment with deceptive alerts and stakeholder debate.",
        "tags": ["cybersecurity", "adversarial", "decision-making", "openenv"],
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {
            "type": "object",
            "description": "Debug state snapshot from GET /state",
        },
    }


@app.post("/mcp")
def mcp(payload: dict[str, Any]) -> dict[str, Any]:
    # Minimal JSON-RPC envelope for OpenEnv runtime checks.
    # (This is not a full MCP server implementation.)
    return {"jsonrpc": "2.0", "id": payload.get("id"), "result": {"ok": True}}
