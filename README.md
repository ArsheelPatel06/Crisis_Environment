# Adversarial Cyber Crisis Simulator (OpenEnv)

**Server (HF Space / local):** `uv sync` then `uv run server` (port 7860 by default; see [openenv.yaml](openenv.yaml)).

**Validation:** `openenv validate` and with a running server, `openenv validate --url http://127.0.0.1:7860`.

**Training (CPU):** metrics and plots without a GPU:
```text
uv run python -m training.train_unsloth_grpo --eval-all
uv run python -m training.generate_results
```

**Training (GPU, TRL + optional Unsloth):** `pip install -e .[train]` (or [requirements-train.txt](requirements-train.txt)) then:
```text
python -m training.train_unsloth_grpo --train --model Qwen/Qwen2-0.5B-Instruct
```

**SFT rollouts (JSONL):** `python -m training.rollout_sft --out data/sft_rollout.jsonl`

**HTTP vs in-process check:** with the server up, `python -m training.eval_http --base-url http://127.0.0.1:7860` → [results/deploy_parity.json](results/deploy_parity.json)

Design and judging notes: [IMPLEMENTATION.md](IMPLEMENTATION.md).
