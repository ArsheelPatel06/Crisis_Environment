from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.env import CyberCrisisEnv
from server.models import Action, Observation, Reward, TaskId


app = FastAPI(title="OpenEnv Cyber Crisis Simulator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"
if _DASHBOARD_DIR.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(_DASHBOARD_DIR), html=True), name="dashboard")
env = CyberCrisisEnv(seed=0, task_id="full_crisis_episode")


class ResetRequest(BaseModel):
    seed: int = 0
    task_id: TaskId = "full_crisis_episode"


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "Adversarial Cyber Crisis Simulator",
        "version": "0.2.0",
        "status": "running",
        "description": (
            "Multi-agent incident-response RL environment. "
            "Red-team attacker advances a kill chain while a blue-team LLM agent "
            "classifies alerts, debates stakeholders, and contains the breach."
        ),
        "endpoints": {
            "POST /reset": "Start a new episode — body: {seed, task_id} or query params",
            "POST /step": "Send an action, receive observation + reward",
            "GET /state": "Full internal state (debug)",
            "GET /health": "Health check",
            "GET /metadata": "Environment metadata",
            "GET /schema": "Action / Observation JSON schemas",
            "POST /mcp": "OpenEnv MCP stub",
        },
        "tasks": ["alert_triage", "stakeholder_argument", "full_crisis_episode"],
        "docs": "/docs",
    }


@app.post("/reset", response_model=Observation)
def reset(
    body: Optional[ResetRequest] = None,
    seed: int = Query(default=0),
    task_id: TaskId = Query(default="full_crisis_episode"),
) -> Observation:
    # Accept seed/task_id from either JSON body or query params (body takes priority)
    _seed = body.seed if body is not None else seed
    _task_id = body.task_id if body is not None else task_id
    return env.reset(seed=_seed, task_id=_task_id)


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
