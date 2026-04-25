---
title: Cyber-Crisis
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - cybersecurity
  - reinforcement-learning
  - adversarial
license: apache-2.0
short_description: >
  Multi-agent incident-response RL env — red-team attacker, deceptive alerts,
  stakeholder debate, three graded tasks.
---

# Adversarial Cyber Crisis Simulator (OpenEnv)

**One-line pitch:** A multi-agent incident-response RL environment where a **red-team attacker** advances a kill chain, injects deceptive alerts, and poisons stakeholder channels — while a **blue-team LLM agent** must classify real vs fake alerts, justify isolation decisions under stakeholder pushback, and contain the breach.

---

## Where we are right now

### System overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ADVERSARIAL CYBER CRISIS SIMULATOR                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌──────────────────────────────────────────────────────────────────────────┐
  │                         FastAPI HTTP Server                              │
  │      server/main.py  ·  POST /reset  POST /step  GET /state             │
  └─────────────────────────────┬────────────────────────────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   CyberCrisisEnv       │  server/environment.py
                    │   (episode loop)       │  ← canonical implementation
                    │                        │  server/env.py = shim
                    └──┬──────────┬──────────┘
                       │          │
           ┌───────────▼──┐  ┌────▼─────────────────────────────────────────┐
           │  CompanyWorld │  │           server/agents.py                   │
           │  simulator/   │  │                                              │
           │  world.py     │  │  NetworkState  (Pydantic snapshot)           │
           │               │  │                                              │
           │  5 nodes:     │  │  StakeholderAgent (abstract, poisonable)     │
           │  api_gateway  │  │    ├─ FinanceStakeholder   opposes isolation │
           │  internal_    │  │    ├─ EngineeringStakeholder wants shutdown  │
           │    tools      │  │    └─ PRStakeholder         delays comms     │
           │  auth_server  │  │                                              │
           │  database     │  │  RedTeamAgent  (fixed strategy)              │
           │  comms        │  │    position: API_Gateway → Internal_Tools    │
           └──────┬────────┘  │             → Auth_Server → Database         │
                  │           │    step(NetworkState) → advance or halt      │
           ┌──────▼────────┐  │    generate_fake_alert(node)                 │
           │ StakeholderSys│  │    plant_false_report(stakeholder, content)  │
           │ simulator/    │  │    attacker_reward() → +1.0 / -1.0 / 0.0    │
           │ stakeholders  │  └──────────────────────────────────────────────┘
           │ .py           │
           │ debate gate   │
           └───────────────┘
```

### Per-step data flow

```
  Blue-team agent sends Action
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  1. DEBATE GATE                                                         │
  │     if pending debate AND action ≠ communicate → force noop             │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  2. BLUE ACTION applied to CompanyWorld                                 │
  │     monitor  → monitoring_level += 1 on target node                    │
  │     isolate  → isolation flag + opens stakeholder debate if auth_server │
  │     patch    → compromise_stage -= 1 on target node                    │
  │     communicate → resolve pending debate via StakeholderSystem          │
  │     ignore / noop → noop_streak++                                       │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  3. BUILD NetworkState from world snapshot                              │
  │     defender_detected_red_team = (monitoring_level ≥ 2) OR isolated    │
  │       on the node where RedTeamAgent currently sits                     │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  4. RED TEAM STEP (server/agents.py RedTeamAgent)                       │
  │     if detected AND position ∈ {API_Gateway, Internal_Tools}            │
  │       → caught_before_auth = True  (frozen, reward = -1.0)             │
  │     else → advance one hop on kill chain                                │
  │     if reaches Database undetected → reward = +1.0                     │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  5. STAKEHOLDER MESSAGES generated (Finance / Engineering / PR)         │
  │     each persona reads NetworkState + any planted poison_payload        │
  │     poison_payload consumed (one-shot) → message includes false intel  │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  6. TRUST SCORE UPDATE                                                  │
  │     if action was ignore or communicate AND stakeholder had poison      │
  │       → stakeholder_trust_scores[name] -= 0.1  (min 0.0)               │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  7. WORLD ATTACKER STEP  world.step_attacker_and_generate_alerts()      │
  │     probabilistic node compromise progression                           │
  │     emits real + fake alerts (world-level attacker, separate from RT)   │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  8. REWARD computed, done checked, Observation built                    │
  │     obs includes: alerts, stakeholder_messages, stakeholder_trust_scores│
  └─────────────────────────────────────────────────────────────────────────┘
```

### Kill-chain vs detection outcomes

```
  RedTeamAgent.position timeline:

  [API_Gateway] ──► [Internal_Tools] ──► [Auth_Server] ──► [Database]
       │                  │                    │                │
       │ detected?        │ detected?          │ detected?      │ reached?
       ▼                  ▼                    ▼                ▼
   caught_before      caught_before      detected_ever=True  reward = +1.0
   _auth=True         _auth=True         but NOT frozen       if NOT detected
   reward = -1.0      reward = -1.0      reward = 0.0         ever
   (frozen here)      (frozen here)
```

### Trust score lifecycle

```
  Stakeholder starts at trust = 0.7

  RedTeamAgent.plant_false_report(stakeholder, content)
    └─► stakeholder.receive_planted_report(content)
         └─► poison_payload = content

  On next step():
    generate_message() weaves poison into utterance  ← obs.stakeholder_messages
    _consume_poison_clause() clears payload (one-shot)

  If blue agent action == "ignore" or "communicate" while payload was set:
    stakeholder_trust_scores[name] -= 0.1   (min 0.0)

  Trust score surfaces in obs.stakeholder_trust_scores and informs
  _compute_reward() trust dimension.
```

---

## Repository layout (current, as of today)

```
Cyber_Crisis/
│
├── server/
│   ├── main.py            FastAPI routes: /reset /step /state /health /metadata /schema /mcp
│   ├── environment.py     CyberCrisisEnv — canonical implementation (NEW)
│   ├── env.py             One-line shim: from server.environment import CyberCrisisEnv
│   ├── agents.py          NetworkState · StakeholderAgent · Finance/Engineering/PR · RedTeamAgent (NEW)
│   ├── models.py          Pydantic: Observation (+ stakeholder_messages, stakeholder_trust_scores)
│   │                               Action · Reward
│   ├── tasks.py           Deterministic graders: Task1 alert_triage · Task2 stakeholder_argument
│   │                                             Task3 full_crisis_episode
│   ├── rubrics.py         Reward helpers
│   ├── seeds.py           Deterministic seeding utilities
│   ├── app.py             ASGI app factory
│   └── run.py             uvicorn entrypoint
│
├── simulator/
│   ├── world.py           CompanyWorld · NodeState · kill-chain progression · alert gen
│   └── stakeholders.py    StakeholderSystem · debate gate · score_argument · evidence-aware
│
├── training/
│   ├── prompts.py         System prompt · JSON Action schema · parse_text_to_action
│   ├── rollout_sft.py     Heuristic trajectory logger → data/sft_rollout.jsonl
│   ├── baseline.py        Mock + live HTTP heuristic baseline
│   ├── eval_inprocess.py  In-process metrics (no GPU required)
│   ├── eval_http.py       HTTP parity check → results/deploy_parity.json
│   ├── train_unsloth_grpo.py  GRPO Task1 trainer (TRL / Unsloth / CPU-fallback)
│   ├── generate_results.py    Curve PNGs + before/after episode markdown
│   ├── policy_heuristic.py    Deterministic policy used by SFT roller
│   └── train.ipynb        Colab runbook
│
├── data/
│   └── sft_rollout.jsonl  182 (observation → Action JSON) training pairs
│
├── results/
│   ├── metrics_baseline.json   Heuristic baseline scores (Task1-3)
│   ├── metrics_oracle.json     Task1 oracle ceiling
│   ├── training_log.csv        GRPO step/reward log (replace with real Colab export)
│   ├── task1_curve.png         Reward curves
│   ├── task2_curve.png
│   ├── before_after_episode.md Episode narrative comparison
│   ├── baseline_report.md      Baseline run summary
│   └── deploy_parity.json      HTTP vs in-process parity
│
├── tests/
│   ├── test_tasks.py       Deterministic grader unit tests
│   └── test_determinism.py Same-seed → same-output env determinism check
│
├── openenv.yaml            OpenEnv manifest (tasks, reward range, interface)
├── Dockerfile              HF Space / container
├── pyproject.toml          Package + server entrypoint + [train] extras
├── requirements.txt        Runtime deps (fastapi, uvicorn, pydantic)
├── requirements-train.txt  GPU/training deps (torch, trl, unsloth, peft …)
├── IMPLEMENTATION.md       Full design + judging rubric map
├── DEMO_SEEDS.md           Demo seeds (14, 108, 23) for video
└── training/HANDOFF.md     Teammate runbook
```

---

## What is fully implemented

| Area | File(s) | Status |
|------|---------|--------|
| **OpenEnv HTTP API** | `server/main.py` | `/reset` `/step` `/state` `/health` `/metadata` `/schema` `/mcp` |
| **World simulator** | `simulator/world.py` | 5-node compromise progression, deceptive alert generation |
| **Stakeholder debate** | `simulator/stakeholders.py` | Evidence-aware scoring, trust penalties, debate gate |
| **Env orchestration** | `server/environment.py` | Full task loop, reward computation, `infected_nodes` property |
| **Pydantic agents** | `server/agents.py` | `NetworkState` · `FinanceStakeholder` · `EngineeringStakeholder` · `PRStakeholder` · `RedTeamAgent` |
| **Red-team agent** | `server/agents.py` | Kill-chain stepping, fake alert generation, stakeholder poisoning, reward signal |
| **Observation enriched** | `server/models.py` | `stakeholder_messages: List[str]` + `stakeholder_trust_scores: Dict[str, float]` |
| **Deterministic graders** | `server/tasks.py` | Task 1 (alert triage) · Task 2 (stakeholder argument) · Task 3 (full episode) |
| **LLM contract** | `training/prompts.py` | System prompt + strict JSON Action schema + `parse_text_to_action` |
| **SFT dataset** | `training/rollout_sft.py` | 182 chat-JSONL pairs in `data/sft_rollout.jsonl` |
| **GRPO training** | `training/train_unsloth_grpo.py` | CPU fallback + GPU 4-bit LoRA; Task1 reward from env |
| **Eval pipeline** | `eval_inprocess.py`, `eval_http.py` | In-process + HTTP parity metrics |
| **Result artifacts** | `training/generate_results.py` | Curve PNGs, before/after markdown |
| **Backward compat** | `server/env.py` (shim) | All existing `from server.env import CyberCrisisEnv` keep working |
| **Tests** | `tests/` | Grader tests + determinism check (both pass) |

---

## What is not done yet

| Item | Priority |
|------|----------|
| **HF Space live URL** — judges need a public endpoint to run `openenv validate --url …` | High |
| **Fill README links** — Space URL, Colab notebook URL, video URL, weights repo | High |
| **Real GRPO training log** — replace synthetic `training_log.csv` with Colab-exported CSV so `task1_curve.png` matches the actual run | High |
| **Video (90 s – 2 min)** — demo seeds 14, 108, 23; baseline vs trained narrative | High |
| **RedTeamAgent wired into RL loop** — currently fixed strategy; RL co-training with blue team | Optional |
| **Task 2 / Task 3 GRPO** — current GRPO targets Task 1 only | Optional |
| **`eval_lora_task1` script** — score saved LoRA adapter on seed bank | Optional |
| **`/mcp` is a stub** — passes OpenEnv validate; not a full MCP server | Note |

---

## Quick start

```bash
# Clone and install
cd ~/Desktop/Cyber_Crisis
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Run tests
python3 tests/test_determinism.py
python3 -m unittest discover -s tests -p 'test_*.py'

# Start server
python3 -m server.run
# → http://127.0.0.1:7860

# Generate metrics + curves (no GPU needed)
python3 -m training.train_unsloth_grpo --eval-all
python3 -m training.generate_results

# Run heuristic baseline against live server (server must be running)
python3 -m training.baseline --base-url http://127.0.0.1:7860 --seed 42 --episodes 1 --max-steps 60

# Build SFT dataset
python3 -m training.rollout_sft --out data/sft_rollout.jsonl
```

---

## Colab GPU training

```python
# Runtime → T4 GPU → Restart
%cd /content
!rm -rf Cyber_Crisis
!git clone -b review/team-pull https://github.com/ArsheelPatel06/Crisis_Environment.git Cyber_Crisis
%cd /content/Cyber_Crisis
!pip install -q -e ".[train]"
!python3 -m training.train_unsloth_grpo --export-dataset
!python3 -m training.train_unsloth_grpo --train \
    --model Qwen/Qwen2-0.5B-Instruct --epochs 1 --output-dir /content/outputs
# Checkpoint: /content/outputs/grpo/final/
!zip -r /content/grpo_artifacts.zip /content/outputs/grpo
```

---

## Three tasks at a glance

| Task ID | Difficulty | One step or multi | How graded |
|---------|------------|-------------------|------------|
| `alert_triage` | Easy | One step | Classify each alert real/fake; F1-style score vs ground truth |
| `stakeholder_argument` | Medium | 2–3 steps | Open debate with isolate/patch; then `communicate` with `argument_text` + `citations` |
| `full_crisis_episode` | Hard | Up to `max_steps` | Multi-step containment; debate required before isolating auth_server; DB safety + uptime + trust + speed |

Every grader returns a score in `[0.0, 1.0]`. Rewards are deterministic given the same seed + action sequence.

---

## Links (fill in before submission)

| Link | URL |
|------|-----|
| GitHub | `https://github.com/ArsheelPatel06/Crisis_Environment` (branch: `review/team-pull`) |
| HF Space | _deploy and add URL_ |
| Colab notebook | _add "Open in Colab" badge_ |
| Video (90 s – 2 min) | _YouTube / Loom_ |
| Trained weights | _HF model repo or release zip of `grpo/final`_ |

---

## Further reading

- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** — full design, judging rubric map, architecture notes
- **[training/HANDOFF.md](training/HANDOFF.md)** — quick teammate runbook for training pipeline
- **[DEMO_SEEDS.md](DEMO_SEEDS.md)** — seeds 14, 108, 23 for repeatable video demos

---

## License / attribution

Follow your course or hackathon rules for attribution of **Qwen**, **Hugging Face TRL**, **OpenEnv**, and **Unsloth** (if used).
