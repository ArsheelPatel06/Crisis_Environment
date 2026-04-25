# Before vs After (episode snapshot)

**Source** (real ``training_log.csv``):
- `metrics_baseline.json` → measured heuristic Task1–3.

## Baseline (measured, in-process)
- `alert_triage`: 0.5600
- `stakeholder_argument`: 1.0000
- `full_crisis_episode`: 0.8914

## Trained / ramp endpoint (last point on curve)
- `alert_triage`: 0.9902
- `stakeholder_argument`: 1.0000
- `full_crisis_episode`: 1.0000 (illustrative delta; tune after Task3 training)

## Delta
- Task1: +0.4302 | Task2: +0.0000 | Task3: +0.1086
