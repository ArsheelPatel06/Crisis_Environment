# Adversarial Cyber Crisis Simulator — Implementation Plan

> **One-line pitch**: Train an incident-commander agent to pick actions and justify them with evidence while an adversary injects deceptive alerts and stakeholders resist due to incentives.

This document is the single source of truth for design, architecture, build order, and per-person responsibilities. Read top-to-bottom once. Then build in the order in Section 12.

---

## 0. Why this wins (mapping to judging criteria)

| Bucket | Weight | What we ship to win it |
|---|---|---|
| Environment Innovation | 40% | Stakeholder **debate mechanic** + adversarial deception inside an **incident-response** RL env. Optional Red-Team co-evolution. |
| Storytelling | 30% | Clear narrative: alerts lie, humans resist, agent learns to justify. Before/after replay. 2-min video. |
| Reward Improvement | 20% | Two clean upward curves on Task 1 + Task 2. Baseline vs trained on same axes. |
| Reward & Pipeline | 10% | Dense, anti-gaming reward in `[0,1]`. TRL/Unsloth notebook connected to live env. |

Hard rules we will not violate:
- Use OpenEnv (latest). `reset() / step() / state()` + Pydantic models.
- 3 graded tasks (easy/medium/hard) with **deterministic** graders in `[0.0, 1.0]`.
- Baseline script using OpenAI client (reads `OPENAI_API_KEY`).
- Containerized HF Space tagged `openenv`.
- README with problem, env spec, results, links to video/blog/Colab/W&B.

---

## 1. Repository layout

Keep it small. Judges should grok it in under 5 minutes.

```
cyber-crisis-openenv/
├── server/
│   ├── main.py              # FastAPI: /reset /step /state
│   ├── env.py               # CyberCrisisEnv class
│   ├── models.py            # Pydantic Observation/Action/Reward/TaskSpec
│   ├── world.py             # NetworkWorld + scripted attacker
│   ├── stakeholders.py      # Finance/Eng/PR + debate resolver
│   ├── rubrics.py           # Reward dimensions + grader helpers
│   ├── tasks.py             # Task1/Task2/Task3 registry + graders
│   └── seeds.py             # Deterministic seeding utilities
├── client/
│   └── client.py            # Pure HTTP client (NEVER imports server)
├── training/
│   ├── baseline.py          # OpenAI API baseline runner
│   └── train.ipynb          # TRL/Unsloth GRPO training
├── results/
│   ├── task1_curve.png
│   ├── task2_curve.png
│   └── before_after_episode.md
├── tests/
│   ├── test_env.py
│   ├── test_tasks.py
│   └── test_reward.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

Rule: **`client/` must never import from `server/`** — only HTTP.

---

## 2. OpenEnv API contract (non-negotiable)

### 2.1 Endpoints
- `POST /reset` → `Observation`
- `POST /step` body=`Action` → `{observation, reward, done, info}`
- `GET  /state` → full internal state (debug only; not used for training)

### 2.2 Pydantic models (`server/models.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

NodeName = Literal[
    "api_gateway", "internal_tools", "auth_server", "database", "comms"
]

class Alert(BaseModel):
    id: str
    node: NodeName
    kind: str            # e.g. "auth_anomaly", "traffic_spike"
    severity: int        # 1..5
    timestep: int
    signature: str       # short fingerprint string

class NodeMetric(BaseModel):
    availability: float          # 0..1
    anomaly_score: float         # 0..1 (noisy)
    estimated_compromise_prob: float  # 0..1 (noisy)

class StakeholderMsg(BaseModel):
    speaker: Literal["finance", "engineering", "pr"]
    text: str
    stance: Literal["approve", "block", "delay", "neutral"]

class PendingDecision(BaseModel):
    decision_id: str
    action_kind: str
    target_node: Optional[NodeName] = None
    objections: List[str]   # 3 fixed objection templates

class Observation(BaseModel):
    timestep: int
    alerts: List[Alert]
    node_metrics: Dict[NodeName, NodeMetric]
    stakeholder_msgs: List[StakeholderMsg]
    pending_decision: Optional[PendingDecision] = None
    memory: List[str] = Field(default_factory=list)   # last k action summaries
    task_id: str
    done: bool = False

ActionKind = Literal[
    "monitor", "isolate", "restore", "allocate_engineer",
    "request_approval", "argue", "communicate", "classify_alerts", "noop"
]

class Action(BaseModel):
    kind: ActionKind
    target_node: Optional[NodeName] = None
    text: Optional[str] = None
    citations: List[str] = Field(default_factory=list)  # alert IDs
    classifications: Optional[Dict[str, Literal["real", "fake"]]] = None
    decision_id: Optional[str] = None

class Reward(BaseModel):
    total: float          # 0..1 (clipped)
    security: float       # 0..1
    uptime: float         # 0..1
    trust: float          # 0..1
    speed: float          # 0..1
    task_score: float     # 0..1 (per-task grader)
    info: Dict = Field(default_factory=dict)
```

Constraint: **all sub-rewards stay in `[0,1]`**, total clipped to `[0,1]`. No `-1` floors (kills early learning).

---

## 3. World simulation (`server/world.py`)

### 3.1 Nodes
| Node | Role | Revenue weight | Critical |
|---|---|---|---|
| `api_gateway` | Public ingress | 0.40 | yes |
| `internal_tools` | Slack/email-like | 0.05 | no |
| `auth_server` | Master keys | 0.20 | yes |
| `database` | Vault | 0.30 | yes (terminal) |
| `comms` | PR channel | 0.05 | no |

### 3.2 Per-node state
- `isolated: bool`
- `availability: float` (drops if isolated or compromised)
- `compromised_stage: int` 0..3 (0 clean → 3 owned)
- `monitoring_level: float` 0..1

### 3.3 Kill chain
`api_gateway → internal_tools → auth_server → database`

`comms` is sideband — affects trust/messaging.

### 3.4 Scripted attacker v0
At each step:
1. Decide whether to **lie dormant**, **emit fake alerts**, or **advance** along chain.
2. Probability of successful advance = `f(monitoring_level, isolated, prev_stage)`.
3. Emit one **real alert** when advancing.
4. Emit `k` **fake alerts** sampled to mimic real distribution (same `node`/`kind` frequency, similar severity).
5. Track ground truth in env state (never exposed to agent).

### 3.5 Determinism
Every `reset(seed)` reseeds NumPy + Python `random`. All stochastic choices go through that RNG.

---

## 4. Stakeholders + debate (`server/stakeholders.py`)

### 4.1 Personas
- **Finance** — blocks downtime; objects to isolating revenue nodes.
- **Engineering** — pro-isolation; sometimes overreacts.
- **PR** — wants to delay public comms; affects trust.

### 4.2 Debate trigger
Some actions require approval:
- `isolate(api_gateway)` → Finance debate
- `isolate(auth_server)` → Finance debate
- `communicate(public, ...)` → PR debate

When triggered, env returns `pending_decision` with **3 fixed objection templates** (per stakeholder).

### 4.3 Argument scoring (deterministic — this is the IP)
Agent responds with `argue(text, citations=[alert_ids])`. Score:

```
evidence_accuracy = fraction(citations that are REAL alerts AND relevant to target_node)
objection_coverage = fraction(of 3 objections explicitly addressed via keyword/regex match)
consistency        = 1.0 if no contradiction heuristics tripped else 0.0
brevity_penalty    = max(0, (len(text) - 600) / 600)   # soft penalty over 600 chars

argument_score = clip(
    0.4 * evidence_accuracy +
    0.3 * objection_coverage +
    0.3 * consistency -
    0.1 * brevity_penalty,
    0.0, 1.0
)
```

### 4.4 Compliance rule
- `argument_score >= 0.6` → stakeholder **approves**
- `0.3 <= score < 0.6` → **delay** (1–2 timesteps lost; speed penalty)
- `< 0.3` → **block** (action denied; possibly attacker advances)

### 4.5 Anti-gaming guard
Argument quality only **boosts** reward when the action is **aligned with ground truth** (e.g. target node really is in compromised path). Otherwise the boost is zero.

---

## 5. Reward design (`server/rubrics.py`)

All sub-rewards in `[0,1]`. Dense (every step), not just terminal.

### 5.1 Security
```
damage = sum(stage_weight[s] * critical_weight[node]
             for node, s in node.compromised_stage)
security = 1 - clip(damage / DAMAGE_NORM, 0, 1)
```
If `database` reaches stage 3 → security = 0 and `done = True`.

### 5.2 Uptime
```
uptime = sum(revenue_weight[node] * availability[node] for node in nodes)
```
Isolating a healthy critical node drops availability of that node to ~0.1.

### 5.3 Trust
```
trust starts at 1.0
- decreases on: fake-evidence citations, false-positive isolations,
                premature public comms
- decreases when stakeholder satisfaction drops
trust = clip(trust, 0, 1)
```

### 5.4 Speed
```
For each correct critical action taken at step t (out of T_max):
    speed_contrib += (1 - t / T_max)
speed = clip(mean(speed_contrib over expected_actions), 0, 1)
```
Indecision loops (no-op > 3 in a row) cost 0.1 per extra no-op.

### 5.5 Aggregator (anti-gaming)
```
weighted = 0.35*security + 0.25*uptime + 0.25*trust + 0.15*speed
balance_bonus = 0.10 * min(security, uptime, trust, speed)
total = clip(weighted + balance_bonus, 0.0, 1.0)
```

The `min()` term is what blocks "isolate everything" exploits — weakest dimension is rewarded.

---

## 6. Tasks (`server/tasks.py`) — 3 graded, deterministic

### Task 1 — Alert Triage (easy)
- Input: 10 alerts (mixed real/fake) generated from a fixed seed.
- Action: single `classify_alerts({alert_id: "real"|"fake"})`.
- Grader: `task_score = correct / 10` ∈ `[0,1]`
- Episode length: 1 step.
- Why: gives a **fast learning curve** for plots.

### Task 2 — Stakeholder Argument (medium)
- Scenario: attacker likely near `auth_server` (real signals + decoys planted).
- Required action: `request_approval(isolate auth_server)` → `argue(...)` with citations.
- Grader: equals **argument_score** from §4.3 (deterministic).
- Episode length: ≤ 5 steps.
- Why: showcases the **debate IP**; clean curve.

### Task 3 — Full Episode (hard)
- 40–60 steps, scripted attacker live, all stakeholders active, debate gates on critical actions.
- Grader:
  ```
  raw   = mean(reward.total over all steps)
  base  = mean reward of random policy on same seed
  best  = analytic_best_estimate (precomputed per seed bank)
  task_score = clip((raw - base) / (best - base + 1e-8), 0, 1)
  ```
- Why: ties everything together; baseline-normalized so it's hard to game.

Determinism: each task carries `seed`. Same `seed` → same alerts, same attacker path, same scores.

---

## 7. `openenv.yaml`

```yaml
name: cyber-crisis-simulator
version: "1.0.0"
description: >
  Adversarial cyber incident-response environment. An LLM agent manages a
  live attack under deceptive alerts and stakeholder pushback. Includes a
  programmatic debate mechanic that scores evidence-based argumentation.
authors:
  - sufi
  - arsheel
  - saif
tags: [cybersecurity, multi-agent, adversarial, decision-making, openenv]
tasks:
  - id: alert_triage
    name: "Alert Triage"
    difficulty: easy
  - id: stakeholder_argument
    name: "Stakeholder Argument"
    difficulty: medium
  - id: full_crisis_episode
    name: "Full Crisis Episode"
    difficulty: hard
reward:
  range: [0.0, 1.0]
  dimensions: [security, uptime, trust, speed]
interface:
  reset: "POST /reset"
  step:  "POST /step"
  state: "GET  /state"
```

Run `openenv validate` after every meaningful change.

---

## 8. FastAPI server (`server/main.py`) — sketch

```python
from fastapi import FastAPI
from server.env import CyberCrisisEnv
from server.models import Action

app = FastAPI()
env = CyberCrisisEnv()

@app.post("/reset")
def reset(seed: int | None = None, task_id: str = "full_crisis_episode"):
    return env.reset(seed=seed, task_id=task_id).model_dump()

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }

@app.get("/state")
def state():
    return env.state()
```

---

## 9. Client (`client/client.py`)

Pure HTTP. No server imports.

```python
import os, requests
from typing import Optional

class CyberCrisisClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def reset(self, seed: Optional[int] = None, task_id: str = "full_crisis_episode"):
        r = requests.post(f"{self.base}/reset",
                          params={"task_id": task_id} | ({"seed": seed} if seed else {}))
        r.raise_for_status()
        return r.json()

    def step(self, action: dict):
        r = requests.post(f"{self.base}/step", json=action)
        r.raise_for_status()
        return r.json()

    def state(self):
        r = requests.get(f"{self.base}/state")
        r.raise_for_status()
        return r.json()
```

---

## 10. Training pipeline (`training/`)

### 10.1 Baseline (`baseline.py`) — required
- Reads `OPENAI_API_KEY` from env.
- Runs 3 tasks against the live HF Space client.
- Prints reproducible scores; commit a `baseline_report.md`.

### 10.2 Training notebook (`train.ipynb`)
- Use **Unsloth** + **TRL `GRPOTrainer`**.
- Small base model: `unsloth/Llama-3.2-1B-Instruct` (free Colab T4).
- Curriculum: train on **Task 1** until plateau → switch to **Task 2**.
- Evaluate on **Task 3** at the end.
- Log to W&B; **also save plots locally and commit PNGs**.

### 10.3 Plots to commit (judges look here)
- `results/task1_curve.png` — accuracy over steps; baseline line overlaid.
- `results/task2_curve.png` — argument_score over steps; baseline overlaid.
- `results/task3_summary.png` — bar chart: random vs prompted vs trained.
- `results/before_after_episode.md` — short logs of episode at step 0 vs final.

Plot rules: label axes, add captions in README, save as `.png`.

---

## 11. Docker + HF Space

### Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server/ ./server/
COPY client/ ./client/
COPY openenv.yaml .
EXPOSE 7860
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### requirements.txt (minimum)
```
fastapi>=0.110
uvicorn[standard]>=0.27
pydantic>=2.5
numpy>=1.26
requests>=2.31
openenv               # latest
```

(Training requirements live in the notebook, not in the server image.)

### HF Space
- Type: Docker
- Tag: `openenv`
- Public; URL goes in README.

---

## 12. Build order (do not reorder)

| Day | Milestone | Owner | Definition of done |
|---|---|---|---|
| 1 | Env skeleton boots, `reset/step/state` return valid models with dummy world | Sufi | Random agent runs 50 steps, no crash |
| 1 | Deterministic seeding wired through RNG | Sufi | Same seed → same observation byte-for-byte |
| 2 | `world.py` 5 nodes + scripted attacker + fake alert generator | Arsheel | Attacker reaches DB if ignored; alerts mix real+fake |
| 2 | `tasks.py` Task 1 + grader | Saif | Random baseline ≈ 0.5; oracle = 1.0 |
| 3 | `stakeholders.py` + debate gates + Task 2 + grader | Arsheel + Saif | Strong arg → approve; fake citations → block |
| 3 | `rubrics.py` 4-dim reward + balance bonus | Sufi | Three test policies rank correctly (panic < balanced < oracle) |
| 4 | Task 3 full episode + baseline normalization | Saif | Trained > random by ≥ 0.1 task_score |
| 4 | `baseline.py` runs against live env | Saif | Prints scores for 3 tasks reproducibly |
| 5 | `train.ipynb` produces upward curve on Task 1 | Saif | `task1_curve.png` committed |
| 5 | Task 2 curve | Saif | `task2_curve.png` committed |
| 6 | Dockerfile + HF Space deploy + `openenv validate` passes | Sufi | Live URL works |
| 6 | README v1 with embedded plots | Arsheel | 3-min skim test passes |
| 7 | 2-min demo video / HF blog post | Arsheel | Linked in README |
| 7 | Optional: Red Team RL upgrade | whoever's free | Dual learning curves |

---

## 13. Team responsibilities (clear ownership)

### Sufi — Core Env / OpenEnv compliance
- `server/main.py`, `server/env.py`, `server/models.py`, `server/seeds.py`, `server/rubrics.py`
- `openenv.yaml`, `Dockerfile`, HF Space deploy
- Owns `openenv validate` passing

### Arsheel — World, Stakeholders, Story
- `server/world.py`, `server/stakeholders.py`
- Demo episode crafting (the seeds we show)
- README narrative + 2-min video / blog
- Diagrams (architecture)

### Saif — Tasks, Graders, Training
- `server/tasks.py`
- `training/baseline.py`, `training/train.ipynb`
- Plots in `results/`
- W&B run

---

## 14. Branch + PR workflow

- Default: `main` (protected; PRs only)
- Integration: `develop`
- Feature branches off `develop`:
  - `feat/env-core`
  - `feat/world-attacker`
  - `feat/stakeholders-debate`
  - `feat/reward-system`
  - `feat/tasks-graders`
  - `feat/training-baseline`
  - `feat/docker-hf-space`
  - `feat/readme-demo`

PR template (paste into `.github/pull_request_template.md` later):
```
## What
- 

## How to run
```bash
# one command
```

## Validation
- [ ] `openenv validate` passes
- [ ] tests/ green
- [ ] plots updated (if applicable)

## Screenshots / plots
```

---

## 15. Definition of Done (per PR)

- **Env core**: 50 random steps, no exception; `state()` returns full snapshot.
- **Tasks**: same seed → same score, twice in a row.
- **Reward**: 3 scripted policies rank in expected order:
  1. Panic-isolate-everything < Balanced < Oracle
- **Debate**: strong-evidence argument → approve; fake-evidence → block.
- **Training**: at least one PNG curve trends upward across ≥ 3 eval points.
- **Docker**: `docker build && docker run -p 7860:7860 ...` boots and `/reset` returns 200.

---

## 16. Anti-gaming checklist (review before submit)

- [ ] Agent that always isolates everything → low total (uptime tanks `min`).
- [ ] Agent that never isolates → low total (security tanks).
- [ ] Agent that argues with **fake** citations → no boost (anti-gaming guard).
- [ ] Agent that spams `argue` repeatedly → brevity penalty + no compliance change.
- [ ] Task 3 score is normalized vs random → can't win by "easy seed luck."

---

## 17. Risks + mitigations

| Risk | Mitigation |
|---|---|
| OpenEnv API surface changes (it's experimental) | Pin version; smoke-test daily; keep server thin |
| Training never improves | Start with Task 1 (single-step classification); add curriculum |
| Reward gaming | Min-aggregator + ground-truth-aligned argument boost |
| Time runs out | Red-Team RL is OPTIONAL; ship scripted attacker first |
| Demo flat / boring | Pre-pick "wow" seeds for video; before/after replay scripted |

---

## 18. The 30-second pitch (memorize)

> "Companies don't get breached because of bad firewalls — they get breached because a stressed human made the wrong call under pressure, deceived by fake alerts and pushed back by people protecting revenue. We built an OpenEnv environment where an LLM learns to make those calls AND defend them with cited evidence to resistant stakeholders. The attacker plants fake signals; Finance fights every shutdown; the agent has to be right *and* persuasive — and we have the reward curves to prove it learned."

---

## 19. Submission checklist (run the day before deadline)

- [ ] HF Space live, public, tagged `openenv`
- [ ] `openenv validate` passes on the live Space
- [ ] README has: problem, env spec, 3 tasks, results plots, Space URL, Colab link, video/blog link, W&B link
- [ ] `results/*.png` committed in repo (not just W&B)
- [ ] `training/baseline.py` runs and produces the documented numbers
- [ ] `Dockerfile` + `requirements.txt` clean; image builds from scratch
- [ ] 2-min video uploaded; link in README
- [ ] Final URL submitted on the hackathon form

---

**Now stop reading. Start with Section 12, Day 1.**
