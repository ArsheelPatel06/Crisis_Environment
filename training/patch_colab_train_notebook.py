"""Regenerate install cell + chdir preambles in train.ipynb. Run: python3 training/patch_colab_train_notebook.py"""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent / "train.ipynb"

MARK = """# --- Colab: project root (set by cell 1) ---
import os, sys
try:
    _ROOT = open("/content/.cyber_crisis_root", encoding="utf-8").read().strip()
except OSError as _e:
    raise RuntimeError("Run the first code cell (clone + pip) first, then re-run this cell.") from _e
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
"""

CELL1 = r"""# === Cell 1: Colab — clone + pip (run first; set Runtime -> GPU) ===
import os, sys, shutil, subprocess

REPO = "https://github.com/ArsheelPatel06/Crisis_Environment.git"
BRANCH = "review/team-pull"
ROOT = "/content/Cyber_Crisis"
if os.path.isdir(ROOT):
    shutil.rmtree(ROOT, ignore_errors=True)

subprocess.check_call(["git", "clone", "-b", BRANCH, "--depth", "1", REPO, ROOT])
pproj = os.path.join(ROOT, "pyproject.toml")
if not os.path.isfile(pproj):
    raise FileNotFoundError(
        f"Missing {pproj}. Branch {BRANCH} on GitHub must have pyproject.toml at repo root."
    )

os.chdir(ROOT)
sys.path.insert(0, ROOT)
with open("/content/.cyber_crisis_root", "w", encoding="utf-8") as f:
    f.write(ROOT)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "pip", "setuptools", "wheel"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".[train]"])
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "trl", "peft", "bitsandbytes", "sentencepiece", "accelerate", "datasets", "huggingface_hub",]
)
subprocess.call(
    [sys.executable, "-m", "pip", "install", "-q", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",]
)
import importlib
importlib.invalidate_caches()
import server
print("OK:", os.getcwd(), "| import server OK")
"""

def to_src(text: str) -> list:
    return [x + "\n" for x in text.strip().split("\n")]


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    first = None
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        if first is None:
            c["source"] = to_src(CELL1)
            first = c["id"]
            continue
        s = c["source"]
        body = s if isinstance(s, str) else "".join(s)
        if "project root (set by cell 1" in body:
            continue
        if body.lstrip().startswith('"""') and "python" in body[:5]:
            pass
        new_body = MARK + body
        c["source"] = [x + "\n" for x in new_body.rstrip().splitlines()] + [""]
        bj = "".join(c["source"])
        bj = bj.replace("['python',", "[sys.executable,")
        c["source"] = [x + "\n" for x in bj.rstrip().splitlines()] + [""]

    NOTEBOOK.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("Patched", NOTEBOOK)


if __name__ == "__main__":
    main()
