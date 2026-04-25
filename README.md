# Adversarial Cyber Crisis Simulator (OpenEnv)

**One-line pitch:** An incident-response environment where deceptive alerts and a **stakeholder debate** gate force evidence-backed decisions. Three **deterministic** graded tasks (`[0,1]` rewards) expose an agent to deception + incentives—suitable for **RL / GRPO** on a small **instruct LLM** (not training a foundation model from scratch).

---

## Table of contents

1. [Can we run RL on Colab?](#can-we-run-rl-on-colab)
2. [What’s done vs what’s left](#whats-done-vs-whats-left)
3. [Repository layout](#repository-layout)
4. [Quick start (macOS)](#quick-start-macos)
5. [Quick start (Google Colab — GPU training)](#quick-start-google-colab--gpu-training)
6. [OpenEnv API & validation](#openenv-api--validation)
7. [Tasks, actions, and training data](#tasks-actions-and-training-data)
8. [Results & testing](#results--testing)
9. [Roadmap (remaining polish)](#roadmap-remaining-polish)
10. [Links to fill in for submission](#links-to-fill-in-for-submission)
11. [Further reading](#further-reading)

---

## Can we run RL on Colab?

**Yes.** That is the intended setup:

| Piece | Where | Notes |
|--------|--------|--------|
| **RL (GRPO)** | **Google Colab (T4+ GPU)** | `training/train_unsloth_grpo.py --train` uses **TRL `GRPOTrainer`** + in-process **`CyberCrisisEnv`** rewards for **Task 1** (`alert_triage`). |
| **LLM** | **Pretrained instruct model** (e.g. `Qwen/Qwen2-0.5B-Instruct`) | You **download** weights from Hugging Face; GRPO **updates** the policy (typically **LoRA** on top of the base model). |
| **Environment “truth”** | Same code as the **HTTP server** (`server/env.py` + `simulator/`) | Training uses the **same** grading logic as `/step` on your Space. |

**Unsloth** is optional (faster / less VRAM on CUDA). The repo runs **HF + PEFT + 4-bit** on GPU without Unsloth.

---

## What’s done vs what’s left

### Done (working today)

| Area | Status |
|------|--------|
| **OpenEnv HTTP API** | `POST /reset`, `POST /step`, `GET /state` + `GET /health`, `/metadata`, `/schema`, `POST /mcp` (minimal JSON-RPC stub for validator). Spec: [openenv.yaml](openenv.yaml). |
| **Simulator** | `simulator/world.py` (deception / progression) + `simulator/stakeholders.py` (debate, evidence-aware scoring). |
| **Env orchestration** | `server/env.py` wires **CompanyWorld** + **StakeholderSystem** to tasks and rewards. |
| **Models** | `server/models.py` — `Observation`, `Action`, `Reward`; JSON-friendly `Action` for LLM policies (`classifications`, `argument_text`, `citations`). |
| **Deterministic graders** | `server/tasks.py` — Task 1–3 scores in `[0,1]`; covered by `tests/test_tasks.py`. |
| **Baseline** | `training/baseline.py` — mock + **live HTTP**; Task 1 posts **`classifications`**; Task 2 opens debate then **`communicate`** when `pending_debate` is set. |
| **LLM contract** | `training/prompts.py` — system prompt + JSON schema + `parse_text_to_action`. |
| **SFT dataset builder** | `training/rollout_sft.py` → `data/sft_rollout.jsonl` (heuristic trajectories, chat JSONL). |
| **In-process metrics** | `training/eval_inprocess.py` + `python -m training.train_unsloth_grpo --eval-all` → `results/metrics_baseline.json`, `results/metrics_oracle.json` (oracle = Task1 ceiling, analysis only). |
| **GRPO training script** | `training/train_unsloth_grpo.py` — export Task1 dataset, **GRPO** with env reward; **CPU fallback** (`use_cpu`) + **GPU** (4-bit LoRA); TRL-version-safe `GRPOConfig` kwargs (`num_generations≥2`, etc.). |
| **Plots / before-after** | `training/generate_results.py` reads metrics + optional `results/training_log.csv`. |
| **HTTP vs local parity check** | `training/eval_http.py` → `results/deploy_parity.json`. |
| **Packaging** | `pyproject.toml`, `Dockerfile`, `uv.lock`, optional `[train]` extras + [requirements-train.txt](requirements-train.txt). |
| **Colab workflow** | Clone branch `review/team-pull`, `pip install -e ".[train]"`, T4 GPU, `--train` completes; checkpoints under `outputs/grpo/{checkpoint-*,final}/`. |

### Left (submission / polish)

| Item | Why it matters |
|------|----------------|
| **HF Space live URL** | Judges hit a public URL; run `openenv validate --url https://YOUR_SPACE`. |
| **README links** | Fill [Links to fill in](#links-to-fill-in-for-submission) (Space, Colab notebook, optional W&B, **90s–2min video**). |
| **Video + narrative** | Storytelling criterion: seed replay (`DEMO_SEEDS.md`), baseline vs “trained” story. |
| **Curves tied to real GRPO** | Replace or augment synthetic ramp in `results/training_log.csv` with **steps/rewards exported from Colab** (CSV), then rerun `generate_results` so `task1_curve.png` / `task2_curve.png` match your run. |
| **Eval script for LoRA `final/`** | Optional: small `python -m training.eval_lora_task1 --adapter path` to score the saved adapter on a seed bank (not required if Colab logs + checkpoint zip are enough). |
| **Task 2 / Task 3 GRPO or SFT** | Current GRPO path is **Task 1–focused**; extend curriculum when time allows. |
| **`/mcp` is a stub** | Passes OpenEnv validate; not a full MCP server—fine for hackathon, note if asked. |
| **OpenAI baseline (optional)** | `IMPLEMENTATION.md` mentions API baseline; current live path is **heuristic** unless you wire `OPENAI_API_KEY` + model calls in `baseline.py`. |

---

## Repository layout

```
Cyber_Crisis/
├── openenv.yaml              # OpenEnv manifest (tasks, reward range, interface)
├── Dockerfile                # Space / container (slim Python, uv, port 7860)
├── pyproject.toml            # Package + server entrypoint; optional [train] deps
├── requirements-train.txt  # Heavy training stack (Colab / GPU machine)
├── README.md                 # This file
├── IMPLEMENTATION.md         # Deep design + judging map (keep in sync mentally)
├── DEMO_SEEDS.md             # Demo seeds for video (14, 108, 23)
├── server/                   # FastAPI app + env
│   ├── main.py               # Routes: reset, step, state, health, metadata, schema, mcp
│   ├── env.py                # CyberCrisisEnv + task wiring
│   ├── models.py             # Pydantic Observation / Action / Reward
│   ├── tasks.py, rubrics.py, seeds.py, run.py, app.py
├── simulator/                # World + stakeholders (drives env)
├── training/
│   ├── train_unsloth_grpo.py # --eval-all, --export-dataset, --train (GRPO Task 1)
│   ├── baseline.py, rollout_sft.py, prompts.py, policy_heuristic.py
│   ├── eval_inprocess.py, eval_http.py, generate_results.py
│   └── train.ipynb           # Runbook-style cells
├── data/                     # e.g. sft_rollout.jsonl (generated)
├── results/                  # metrics, curves, baseline_report, deploy_parity
└── tests/                    # Deterministic task tests
```

---

## Quick start (macOS)

Use **`python3`** (or activate `.venv` first so `python` exists).

```bash
cd ~/Desktop/Cyber_Crisis
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
pip install -e .
```

**Tests:**

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

**Server:**

```bash
python3 -m server.run
# → http://127.0.0.1:7860  (see /docs in FastAPI if enabled, /health, etc.)
```

**Metrics + plots (no GPU):**

```bash
python3 -m training.train_unsloth_grpo --eval-all
python3 -m training.generate_results
```

**Live baseline (server running in another terminal):**

```bash
python3 -m training.baseline --base-url http://127.0.0.1:7860 --seed 42 --episodes 1 --max-steps 60
```

**With `uv` (if installed):** `uv sync` then `uv run server` / `uv run python -m …`.

---

## Quick start (Google Colab — GPU training)

1. **Runtime → Change runtime type → T4 GPU** (or better). **Restart** after switching.
2. Verify: `!nvidia-smi` and `import torch; print(torch.cuda.is_available())` → **True**.

```text
%cd /content
!rm -rf Cyber_Crisis
!git clone -b review/team-pull https://github.com/ArsheelPatel06/Crisis_Environment.git Cyber_Crisis
%cd /content/Cyber_Crisis
!pip install -q -U pip
!pip install -q -e ".[train]"
!python3 -m training.train_unsloth_grpo --export-dataset
!python3 -m training.train_unsloth_grpo --train --model Qwen/Qwen2-0.5B-Instruct --epochs 1 --output-dir /content/outputs
```

**Artifacts:** `/content/outputs/grpo/final/` (and `checkpoint-*`). Zip before disconnect:

```text
!zip -r /content/grpo_artifacts.zip /content/outputs/grpo
```

**HF Hub rate limits:** optional `HF_TOKEN` in Colab secrets.

---

## OpenEnv API & validation

| Check | Command |
|--------|---------|
| Local package | `openenv validate` (from repo root, with `openenv` CLI installed) |
| Running server | `openenv validate --url http://127.0.0.1:7860` |
| Deployed Space | `openenv validate --url https://YOUR_USERNAME.hf.space` |

---

## Tasks, actions, and training data

| Task ID | Difficulty | Idea |
|---------|------------|------|
| `alert_triage` | Easy | Classify alerts **real vs fake**; graded in one step when `classifications` are sent. |
| `stakeholder_argument` | Medium | Open debate with isolate/patch/monitor; then **`communicate`** with `argument_text` + `citations`. |
| `full_crisis_episode` | Hard | Multi-step crisis + debate when isolating auth. |

**GRPO training (implemented):** Task **1** only, in-process reward = env **triage** score. **SFT data:** `rollout_sft` logs heuristic `(messages → assistant JSON)` pairs for optional supervised warm-start.

---

## Results & testing

| Artifact | Produced by |
|----------|-------------|
| `results/metrics_baseline.json` | `--eval-all` |
| `results/metrics_oracle.json` | `--eval-all` (Task1 oracle, not deployable) |
| `results/task1_curve.png`, `task2_curve.png` | `generate_results` |
| `results/before_after_episode.md` | `generate_results` |
| `results/baseline_report.md` | `baseline.py` |
| `results/deploy_parity.json` | `eval_http.py` |

**Determinism:** task unit tests under `tests/`.

---

## Roadmap (remaining polish)

1. **Publish / pin HF Space** + add URL to this README.  
2. **Short video** using `DEMO_SEEDS.md` seeds + one baseline vs trained comparison (even if “trained” is shown via Colab metrics + checkpoint).  
3. **Export Colab training logs** into `results/training_log.csv` (or extend `generate_results` to read TRL’s CSV) so figures match the real GRPO run.  
4. **Optional:** SFT on `data/sft_rollout.jsonl` before GRPO; Task 2 GRPO; small `eval_lora_task1` script.  
5. **Optional:** OpenAI-powered baseline per `IMPLEMENTATION.md` (needs `OPENAI_API_KEY` + code path).

---

## Links to fill in for submission

Replace placeholders when you have them:

| Link | URL |
|------|-----|
| **GitHub** | `https://github.com/ArsheelPatel06/Crisis_Environment` (branch: **`review/team-pull`** for latest integration) |
| **HF Space** | _Add after deploy_ |
| **Colab (training)** | _Add “Open in Colab” or notebook URL_ |
| **Video (90s–2min)** | _YouTube / Loom_ |
| **Weights / adapter** | _HF model repo or release zip of `grpo/final`_ |

---

## Further reading

- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** — Full design, judging rubric mapping, and architecture notes.  
- **[training/HANDOFF.md](training/HANDOFF.md)** — Training pipeline commands.  
- **[DEMO_SEEDS.md](DEMO_SEEDS.md)** — Seeds for repeatable demos.

---

## License / attribution

Follow your course or hackathon rules for attribution of **Qwen**, **Hugging Face**, **OpenEnv**, **TRL**, and **Unsloth** (if used).
