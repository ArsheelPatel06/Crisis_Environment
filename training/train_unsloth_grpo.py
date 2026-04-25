"""GRPO (Task 1) training entry and in-process metrics (no GPU needed for --eval / --export).

- Baseline + oracle metrics: always works: ``python -m training.train_unsloth_grpo --eval-all``
- Export GRPO JSONL: ``python -m training.train_unsloth_grpo --export-dataset`` (seeds, prompts, seeds column)
- Full training: needs CUDA, ``pip install -r requirements-train.txt``, ``pip install unsloth`` (optional speedup)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEED_LIST_DEFAULT = (42, 84, 126, 7, 2025)


def _load_seeds() -> list[int]:
    return list(SEED_LIST_DEFAULT)


def _run_eval_baseline(out: Path) -> None:
    from training.eval_inprocess import TaskScores, build_metrics_json, eval_tasks_on_seeds

    scores: TaskScores = eval_tasks_on_seeds(_load_seeds(), episodes_per_task=1, max_steps=64)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_metrics_json(scores, out, "heuristic_inprocess")
    print("Wrote", out)


def _run_eval_oracle(out: Path) -> None:
    """In-process only: perfect Task-1 grader; documents ceiling, not a deployable policy."""
    from training.eval_inprocess import oracle_task1_perfect_score

    o = {f"seed_{s}": float(oracle_task1_perfect_score(s)) for s in _load_seeds()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "oracle_task1_triage_ceiling": o,
                "note": "Uses ground truth from env.get_state; for analysis only, not a fair deploy baseline.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Wrote", out)


def _action_from_completion(text: str) -> "Any":
    from server.models import Action
    from training.prompts import parse_text_to_action

    try:
        return parse_text_to_action(str(text))
    except Exception:  # noqa: BLE001
        return Action(action_type="noop", target=None)


def _task1_env_reward(
    prompts: list[Any] | str | None,
    completions: list[Any] | str | None,
    **kwargs: Any,
) -> list[float]:
    from server.env import CyberCrisisEnv
    from server.models import Action

    if completions is None:
        return [0.0]
    if not isinstance(completions, (list, tuple)):
        comps: Sequence[Any] = [completions]
    else:
        comps = completions

    seeds: Sequence | None = kwargs.get("seed")
    if seeds is None:
        seeds = [0] * len(comps)
    if hasattr(seeds, "tolist"):
        seeds = seeds.tolist()  # type: ignore[no-untyped-call, assignment]
    if not isinstance(seeds, (list, tuple)):
        seeds = [seeds] * len(comps)

    rewards: list[float] = []
    for i, comp in enumerate(comps):
        s = int(list(seeds)[i] if i < len(seeds) else list(seeds)[-1])  # type: ignore[call-overload, index]
        action = _action_from_completion(str(comp)) if not isinstance(comp, Action) else comp
        if not isinstance(action, Action):
            action = Action(action_type="noop", target=None)  # type: ignore[unreachable]

        env = CyberCrisisEnv(seed=s, task_id="alert_triage")
        env.reset(seed=s, task_id="alert_triage")
        out = env.step(action)
        r = out.get("reward", {}) or {}
        if isinstance(r, dict):
            rewards.append(float(r.get("total", 0.0)))
        else:
            rewards.append(0.0)
    return rewards


def _build_task1_grpo_jsonl_path(seeds: list[int], out_path: Path) -> None:
    from server.env import CyberCrisisEnv
    from training.prompts import build_chat_messages

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in seeds:
            env = CyberCrisisEnv(seed=s, task_id="alert_triage")
            obs = env.reset(seed=s, task_id="alert_triage").model_dump()
            msgs = build_chat_messages("alert_triage", obs)
            row = {
                "seed": s,
                "task_id": "alert_triage",
                "messages": msgs,
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print("Wrote", out_path)


def _apply_chat_template(msgs: list[dict[str, str]], model_name: str) -> str:
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return str(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))  # type: ignore[no-untyped-call]


def _build_grpo_config(
    grpo_config_cls: Any,
    *,
    run_dir: Path,
    epochs: int,
) -> Any:
    """TRL versions differ (e.g. ``max_prompt_length`` removed in newer GRPOConfig). Only pass supported kwargs."""
    from inspect import signature

    allowed = set(signature(grpo_config_cls).parameters)
    # Prefer small generations when batch=1 (global batch must divide num_generations in many TRL versions).
    candidates: dict[str, Any] = {
        "output_dir": str(run_dir),
        "num_train_epochs": float(epochs),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "max_completion_length": 384,
        "max_prompt_length": 1024,
        "num_generations": 1,
    }
    kwargs = {k: v for k, v in candidates.items() if k in allowed}
    return grpo_config_cls(**kwargs)  # type: ignore[no-untyped-call, misc, call-arg, dict-item, truthy, truthy-bool]


def run_grpo_train(
    model_name: str, seeds: list[int], output_dir: Path, epochs: int, use_unsloth: bool
) -> None:
    from importlib import import_module
    from inspect import signature

    import torch  # type: ignore[import-not-untyped, attr-defined, misc, assignment]
    from datasets import Dataset  # type: ignore[import-untyped]
    from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
    from transformers import (  # type: ignore[import-untyped, misc, operator]
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-untyped]

    out_path = output_dir / "grpo_task1.jsonl"
    _build_task1_grpo_jsonl_path(seeds, out_path)
    lines = [json.loads(p) for p in out_path.read_text(encoding="utf-8").strip().splitlines() if p.strip()]

    prompt_texts: list[str] = []
    seed_list: list[int] = []
    for row in lines:
        prompt_texts.append(_apply_chat_template(row["messages"], model_name))
        seed_list.append(int(row["seed"]))

    ds: Any = Dataset.from_dict({"prompt": prompt_texts, "seed": seed_list})

    bnb4 = torch.float16
    if torch.cuda.is_available():
        try:
            bnb4 = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:  # noqa: BLE001
            bnb4 = torch.float16
    bnb = BitsAndBytesConfig(  # type: ignore[call-arg, misc, truthy, truthy-bool]
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bnb4,
    )

    model: Any
    tok: Any
    unsloth_ok = False
    if use_unsloth:
        try:
            u = import_module("unsloth")
            _flm: Any = getattr(u, "FastLanguageModel", None)  # noqa: N802
            if _flm is not None:
                _lc = LoraConfig(  # type: ignore[call-arg, misc, truthy, truthy-bool]
                    r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                )
                model, tok = _flm.from_pretrained(  # type: ignore[no-untyped-call, call-arg, misc, assignment, dict-item, truthy, truthy-bool]
                    model_name,
                    max_seq_length=1024,
                    load_in_4bit=True,
                    full_finetuning=False,
                )
                model = _flm.get_peft_model(  # type: ignore[no-untyped-call, call-arg, misc, assignment, dict-item, truthy, attr-defined]  # noqa: E501
                    model, _lc, use_gradient_checkpointing="unsloth", random_state=42, max_seq_length=1024, use_rslora=True
                )  # noqa: E501
                unsloth_ok = True
        except Exception as exc:  # noqa: BLE001
            print("Unsloth not used:", exc, file=sys.stderr)

    if not unsloth_ok:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # type: ignore[no-untyped-call]
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(  # type: ignore[no-untyped-call, misc, operator, dict-item, truthy, truthy-bool]
            model_name,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        lora = LoraConfig(  # type: ignore[call-arg, misc, truthy, truthy-bool]
            r=8, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        )  # noqa: PIE800
        model = get_peft_model(model, lora)  # type: ignore[no-untyped-call, assignment, misc, arg-type, dict-item, truthy, truthy-bool]

    run_dir = output_dir / "grpo"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg: Any = _build_grpo_config(GRPOConfig, run_dir=run_dir, epochs=epochs)  # type: ignore[assignment, misc, truthy, truthy-bool]
    pnames = set(signature(GRPOTrainer).parameters)  # type: ignore[no-untyped-call, type-arg, misc, truthy, arg-type, truthy-bool]
    kw: dict[str, Any] = {
        "model": model,
        "args": cfg,
        "train_dataset": ds,  # type: ignore[dict-item, arg-type, misc, truthy, truthy-bool]
        "reward_funcs": [_task1_env_reward],  # type: ignore[dict-item, list-item, misc, truthy, truthy-bool]
    }
    if "processing_class" in pnames:
        kw["processing_class"] = tok
    elif "tokenizer" in pnames:
        kw["tokenizer"] = tok
    if "reward_funcs" not in pnames:
        raise RuntimeError("Your TRL is too old for this script: GRPOTrainer must accept reward_funcs=.")
    trainer: Any = GRPOTrainer(**kw)  # type: ignore[no-untyped-call, call-overload, misc, call-arg, dict-item, truthy, truthy-bool]
    trainer.train()
    trainer.save_model(str(run_dir / "final"))  # type: ignore[no-untyped-call, call-arg, misc, truthy, arg-type, truthy-bool]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="In-process eval and optional GRPO (Task1). Training requires trl+torch+peft+CUDA."
    )
    parser.add_argument("--eval-baseline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Write results/metrics_baseline.json and results/metrics_oracle.json",
    )
    parser.add_argument("--export-dataset", action="store_true", help="Write data/grpo_task1.jsonl (prompt+seed for GRPO).")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run GRPO (requires trl, transformers, peft, torch, CUDA, bitsandbytes; optional: unsloth for speed).",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in SEED_LIST_DEFAULT),
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "training" / "outputs" / "grpo_task1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--use-unsloth",
        action="store_true",
        help="If Unsloth is installed, use FastLanguageModel; else 4bit HF + PEFT",
    )
    args = parser.parse_args()

    if args.eval_all or args.eval_baseline:
        _run_eval_baseline(PROJECT_ROOT / "results" / "metrics_baseline.json")
        _run_eval_oracle(PROJECT_ROOT / "results" / "metrics_oracle.json")
    if args.export_dataset:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        _build_task1_grpo_jsonl_path(seeds, PROJECT_ROOT / "data" / "grpo_task1.jsonl")
    if args.train:
        try:
            seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
            run_grpo_train(args.model, seeds, args.output_dir, args.epochs, use_unsloth=bool(args.use_unsloth))
        except Exception as e:  # noqa: BLE001
            print(
                "Training failed (common: missing trl/torch/cuda, or API mismatch for your TRL version). "
                f"Error: {e!s}\n"
                "Use: pip install -r requirements-train.txt; install matching torch for your CUDA.",
                file=sys.stderr,
            )
            raise
    if not (args.eval_all or args.eval_baseline or args.export_dataset or args.train):
        _run_eval_baseline(PROJECT_ROOT / "results" / "metrics_baseline.json")
        _run_eval_oracle(PROJECT_ROOT / "results" / "metrics_oracle.json")
        _build_task1_grpo_jsonl_path(
            [int(x) for x in args.seeds.split(",") if x.strip()], PROJECT_ROOT / "data" / "grpo_task1.jsonl"
        )
        print("Wrote data/grpo_task1.jsonl and results/metrics_*.json (use --help for all options).")


if __name__ == "__main__":
    main()
