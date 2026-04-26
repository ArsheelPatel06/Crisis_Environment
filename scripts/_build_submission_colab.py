"""One-off: emit colabs/OPENENV_HACKATHON_SUBMISSION.ipynb"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "colabs"
ROOT.mkdir(exist_ok=True)
OUT = ROOT / "OPENENV_HACKATHON_SUBMISSION.ipynb"


def c(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"id": "id"},
        "source": [line + "\n" for line in src.strip("\n").split("\n")],
        "outputs": [],
        "execution_count": None,
    }


def m(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {"id": "m"}, "source": [line + "\n" for line in src.strip("\n").split("\n")]}


def join_cells(*cells):
    nbs = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
        },
        "cells": list(cells),
    }
    OUT.write_text(json.dumps(nbs, indent=2) + "\n", encoding="utf-8")
    print("Wrote", OUT)


# --- cells (string blocks; \\n for explicit newline in f-strings not needed) ---
C0 = m(
    r"""# Cyber Crisis — OpenEnv + Hugging Face hackathon (submission Colab)

Train **GRPO** on **Qwen2-0.5B-Instruct** with **4-bit qLoRA** for Task 1 (`alert_triage`), then iterate from checkpoint.

| | |
|:---|:---|
| **Space** | [ArsheelPatel06/Cyber-Crisis](https://huggingface.co/spaces/ArsheelPatel06/Cyber-Crisis) |
| **API (OpenEnv)** | `https://arsheelpatel06-cyber-crisis.hf.space` — `openenv validate` |
| **LoRA upload target (optional)** | [ArsheelPatel06/cyber-crisis-qwen2-lora](https://huggingface.co/ArsheelPatel06/cyber-crisis-qwen2-lora) |
| **Code** | [Crisis_Environment](https://github.com/ArsheelPatel06/Crisis_Environment) |

### Open in Colab (after you push `colabs/OPENENV_HACKATHON_SUBMISSION.ipynb` to GitHub)
Template: `https://colab.research.google.com/github/<user>/<repo>/blob/<branch>/colabs/OPENENV_HACKATHON_SUBMISSION.ipynb`  
Example (if the file lives on `review/team-pull`): [open in Colab](https://colab.research.google.com/github/ArsheelPatel06/Crisis_Environment/blob/review/team-pull/colabs/OPENENV_HACKATHON_SUBMISSION.ipynb)

1. **Runtime** → **Change runtime type** → **T4 GPU** (or better).  
2. **Run all** (or run cells in order).  
3. (Optional) Colab **secrets**: add `HF_TOKEN` to push the adapter.  
4. **Download** the zip in the last cell.
"""
)

C1 = c(
    r"""# === Edit for your fork / branch ===
GITHUB = "https://github.com/ArsheelPatel06/Crisis_Environment"
BRANCH = "review/team-pull"  # e.g. "main"
QUICK = False  # True = 5 seeds, 1 epoch (smoke test)
WDIR = "Cyber_Crisis"
import os, shutil
EPOCHS = 1 if QUICK else 3
SEEDS = "1,2,3,4,5" if QUICK else ",".join(str(i) for i in range(1, 21))
os.environ["WANDB_DISABLED"] = "true"
HF_LORA_REPO = "ArsheelPatel06/cyber-crisis-qwen2-lora"  # change for your model repo
print("BRANCH:", BRANCH, " EPOCHS:", EPOCHS, " QUICK:", QUICK)
"""
)

C2 = c(
    r"""# Clone (Python only — no shell magic) + install
import os, subprocess, sys, shutil, pathlib
from pathlib import Path

def sh(cmd, **kw):
    return subprocess.check_call(cmd, **kw)

if Path(WDIR).is_dir() and (Path(WDIR) / ".git").is_file():
    sh(["git", "-C", WDIR, "fetch", "origin", BRANCH])
    sh(["git", "-C", WDIR, "checkout", BRANCH])
    sh(["git", "-C", WDIR, "pull", "origin", BRANCH])
else:
    if Path(WDIR).exists():
        shutil.rmtree(WDIR, ignore_errors=True)
    sh(["git", "clone", "-b", BRANCH, GITHUB, WDIR])

os.chdir(WDIR)
Path("results").mkdir(parents=True, exist_ok=True)
Path("data").mkdir(parents=True, exist_ok=True)

# Training deps
sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".[train]"])
sh([sys.executable, "-m", "pip", "install", "-q", "trl", "peft", "bitsandbytes", "sentencepiece", "accelerate", "datasets", "huggingface_hub"])
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"]
)
if r.returncode:
    print("unsloth install had issues; omit --use-unsloth in training (HF+PEFT still works).")
else:
    print("unsloth install ok (optional).")

print("Working directory:", os.getcwd())
"""
)

C3 = c(
    r"""# Require GPU
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Set Runtime → T4 GPU and restart, then re-run from the top.")
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print("GPU:", name, " VRAM ~", round(vram, 1), "GB")
"""
)

C4 = c(
    r"""# Task 1 baselines (in-process) — random vs heuristic, 20 seeds
import sys, random, pathlib, csv
from pathlib import Path
sys.path.insert(0, ".")
from server.environment import CyberCrisisEnv
from server.models import Action
from training.policy_heuristic import observation_to_task_action

SEED_LIST = list(map(int, SEEDS.split(",")))

def one(seed, pol):
    env = CyberCrisisEnv(seed=seed, task_id="alert_triage")
    obs = env.reset(seed=seed, task_id="alert_triage").model_dump()
    rng = random.Random(seed)
    if pol == "random":
        a = Action(
            action_type=rng.choice(["isolate", "monitor", "patch", "ignore", "noop"]),
            target=rng.choice(["api_gateway", "internal_tools", "auth_server"]),
        )
    else:
        a = Action.model_validate(observation_to_task_action("alert_triage", obs, seed))
    r = env.step(a)
    return r["reward"]["total"]

R = [one(s, "random") for s in SEED_LIST]
H = [one(s, "heuristic") for s in SEED_LIST]
baseline = sum(R) / len(R)
heuristic_mean = sum(H) / len(H)
print("Random     mean (Task1):", round(baseline, 4))
print("Heuristic  mean (Task1):", round(heuristic_mean, 4))
"""
)

C5 = c(
    r"""# === GRPO Run 1 — from base Qwen2-0.5B, qLoRA, num_generations=4 ===
# Use --output-dir (not --output) with the trainer.
import sys, subprocess, time
t0 = time.time()
try:
    import unsloth  # noqa: F401
    use_u = True
except Exception:
    use_u = False
args = [
    sys.executable, "-m", "training.train_unsloth_grpo",
    "--train",
    "--model", "Qwen/Qwen2-0.5B-Instruct",
    "--task", "alert_triage",
    "--seeds", SEEDS,
    "--epochs", str(EPOCHS),
    "--num-generations", "4",
    "--output-dir", "results/run1",
]
if use_u:
    args.append("--use-unsloth")
    print("Using unsloth fast path")
else:
    print("unsloth not importable; using standard HF+PEFT")
print("Running:", " ".join(args))
r = subprocess.run(args)
print("Exit", r.returncode, "  Time min:", round((time.time() - t0) / 60, 2))
assert r.returncode == 0, "Training failed — read stderr above"
print("Adapter dir: results/run1/grpo/final/")
"""
)

C6 = c(
    r"""# === GRPO Run 2 — continue from Run1 adapter (iterative training) ===
import sys, subprocess, time
from pathlib import Path
ck = Path("results/run1/grpo/final")
if not ck.is_dir():
    print("No checkpoint at", ck, "— run Run1 first.")
else:
    try:
        import unsloth  # noqa: F401
        use_u = True
    except Exception:
        use_u = False
    t0 = time.time()
    args = [
        sys.executable, "-m", "training.train_unsloth_grpo",
        "--train",
        "--model", str(ck.resolve()),
        "--task", "alert_triage",
        "--seeds", SEEDS,
        "--epochs", str(EPOCHS),
        "--num-generations", "4",
        "--output-dir", "results/run2",
    ]
    if use_u:
        args.append("--use-unsloth")
    r = subprocess.run(args)
    print("Exit", r.returncode, "  Time min:", round((time.time() - t0) / 60, 2))
    assert r.returncode == 0, "Run2 failed"
"""
)

C7 = c(
    r"""# Optional: read TRL log if present (rewards/loss in trainer state)
import json, pathlib, glob, os
for label in ("run1", "run2"):
    p = pathlib.Path(f"results/{label}/grpo/trainer_state.json")
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        lh = d.get("log_history", [])
        print(f"--- {label} log_history len =", len(lh), "---")
        if lh:
            print("last", lh[-1])
    else:
        print(f"no {p} (ok if your TRL version uses a different filename)")
"""
)

C8 = c(
    r"""# Generate the paper-style PNGs in results/ (curves, before/after) — from metrics + optional log
import sys, subprocess
# Baseline + oracle json for plots
r = subprocess.run([sys.executable, "-m", "training.train_unsloth_grpo", "--eval-all"], capture_output=True, text=True)
if r.returncode:
    print(r.stderr)
else:
    print("metrics written under results/")

# PNGs
r2 = subprocess.run([sys.executable, "-m", "training.generate_results"], capture_output=True, text=True)
print("generate_results exit", r2.returncode)
if r2.stdout:
    print(r2.stdout[-2000:])

# Show if pictures exist
from pathlib import Path
for name in [
    "before_after.png", "iteration_improvement.png", "training_progress.png", "reward_signal.png",
    "task1_curve.png", "task2_curve.png"
]:
    q = Path("results") / name
    print(name, "OK" if q.is_file() else "missing (expected if no training_log export)")
"""
)

C9 = m("""## Optional: push LoRA to Hugging Face
Add **HF_TOKEN** in **Secrets** (or log in with `huggingface-cli login` in a cell you delete before sharing).""")
C10 = c(
    r"""import os, pathlib
from huggingface_hub import HfApi, login

# Prefer your token from Colab secrets
def _hf_token():
    t = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN", "")
    if t:
        return t
    try:
        from google.colab import userdata
        t = userdata.get("HF_TOKEN")
    except Exception:
        t = ""
    return t or ""

tok = _hf_token()
if not tok:
    print("No HF token — set HF_TOKEN in Colab secrets to upload.")
else:
    login(token=tok, add_to_git_credential=False)
    src = pathlib.Path("results/run2/grpo/final" if pathlib.Path("results/run2/grpo/final").is_dir() else "results/run1/grpo/final")
    if not src.is_dir():
        print("No adapter to upload at run1 or run2 final/")
    else:
        api = HfApi()
        api.upload_folder(folder_path=str(src), repo_id=HF_LORA_REPO, repo_type="model", commit_message="OpenEnv Colab run")
        print("https://huggingface.co/" + HF_LORA_REPO)
"""
)

C11 = m("""## Zip + download (LoRA + plots + metrics)
Run after training. Uses `files.download` on Colab; otherwise saves `grpo_artifacts_*.zip` in cwd.""")
# Simplify C12: avoid google.colab import in non-colab; use try/except
C12b = c(
    r"""# Zip and download
import os, zipfile, pathlib, datetime
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
zname = f"grpo_artifacts_{ts}.zip"
with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as z:
    for d in (pathlib.Path("results/run1/grpo"), pathlib.Path("results/run2/grpo")):
        if d.is_dir():
            for f in d.rglob("*"):
                if f.is_file():
                    z.write(f, arcname=pathlib.Path("submission") / f.relative_to("."))
    res = pathlib.Path("results")
    for f in sorted(res.glob("*.png")) + sorted(res.glob("*.json")) + sorted(res.glob("*.csv")):
        if f.is_file():
            z.write(f, arcname=pathlib.Path("submission") / f.relative_to(res.parent))
print("Created", zname, "size", pathlib.Path(zname).stat().st_size // 1024, "KB")
try:
    from google.colab import files
    files.download(zname)
    print("Browser download started.")
except ImportError:
    print("Not on Colab — file saved at", pathlib.Path(zname).resolve())
"""
)

join_cells(C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12b)
