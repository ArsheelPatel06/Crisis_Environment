# Saif Handoff Runbook (Tasks + Baseline + Training Artifacts)

This runbook lets teammates continue work quickly on top of `tasks-training`.

## What is complete

- Deterministic graders in `server/tasks.py`:
  - `grade_task1_alert_triage`
  - `grade_task2_stakeholder_argument`
  - `grade_task3_full_episode`
- Baseline runner in `training/baseline.py`:
  - mock mode
  - live rollout mode (`--base-url`)
  - graceful fallback (`--fallback-to-mock`)
  - report writer: `results/baseline_report.md`
- Artifact generation helper: `training/generate_results.py`
- Tests: `tests/test_tasks.py`

## Quick commands

### 1) Run task tests

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

### 2) Run baseline in mock mode

```bash
python3 training/baseline.py --seed 42
```

### 3) Run baseline against live server

Start server:

```bash
python3 -m uvicorn server.main:app --host 0.0.0.0 --port 7860
```

In another terminal:

```bash
python3 training/baseline.py --base-url http://127.0.0.1:7860 --seed 42 --episodes 3 --max-steps 60
```

If server is not up yet:

```bash
python3 training/baseline.py --base-url http://127.0.0.1:7860 --seed 42 --episodes 3 --max-steps 60 --fallback-to-mock
```

### 4) Generate checkpoint artifact curves

```bash
python3 training/generate_results.py
```

Outputs:
- `results/task1_curve.png`
- `results/task2_curve.png`
- `results/before_after_episode.md`

## Training pipeline (solo, current)

- Response contract: `training/prompts.py` (JSON `Action` schema, parse with repair in code).
- SFT JSONL: `python -m training.rollout_sft --out data/sft_rollout.jsonl`
- In-process metrics (no GPU): `python -m training.train_unsloth_grpo --eval-all` → `results/metrics_*.json`
- GRPO Task1: `python -m training.train_unsloth_grpo --train` (see `requirements-train.txt` / `pip install -e .[train]`)
- Curves: `python -m training.generate_results` (reads `metrics_baseline.json`; `training_log.csv` may be from GRPO)
- HTTP parity: `python -m training.eval_http --base-url http://127.0.0.1:7860` → `results/deploy_parity.json`

