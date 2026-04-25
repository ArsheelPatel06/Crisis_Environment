# Baseline Report

- Mode: `live`
- Seed: `42`
- Base URL: `http://127.0.0.1:7860`
- Episodes per task: `1`
- Max steps: `60`

## Scores
- `alert_triage`: 0.5455
- `stakeholder_argument`: 1.0000
- `full_crisis_episode`: 0.9186

## Notes
- Scores are deterministic for the same seed and config when using the same heuristic policy.
- `alert_triage` live step posts `classifications` in the action JSON; Task2 opens debate then uses `communicate` when `pending_debate` is set.
