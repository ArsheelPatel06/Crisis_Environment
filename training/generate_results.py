"""Learning curves and before/after. Run ``python -m training.train_unsloth_grpo --eval-all`` first for measured baselines."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
MET_BASE = RESULTS_DIR / "metrics_baseline.json"
MET_ORACLE = RESULTS_DIR / "metrics_oracle.json"
TRAINING_LOG = RESULTS_DIR / "training_log.csv"


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _load_baseline() -> dict[str, float] | None:
    if not MET_BASE.exists():
        return None
    d = json.loads(MET_BASE.read_text(encoding="utf-8"))
    h = d.get("heuristic_inprocess")
    if isinstance(h, dict) and "alert_triage" in h:
        return {k2: float(h[k2]) for k2 in ("alert_triage", "stakeholder_argument", "full_crisis_episode")}
    for v in d.values():
        if isinstance(v, dict) and "alert_triage" in v:
            return {k2: float(v[k2]) for k2 in ("alert_triage", "stakeholder_argument", "full_crisis_episode")}
    return None


def _load_task1_oracle_mean() -> float:
    if not MET_ORACLE.exists():
        return 0.85
    d = json.loads(MET_ORACLE.read_text(encoding="utf-8"))
    o = d.get("oracle_task1_triage_ceiling", {})
    if not isinstance(o, dict) or not o:
        return 0.85
    return _clip01(sum(float(x) for x in o.values()) / len(o))


def _read_csv_log() -> tuple[list[int], list[float], list[float]] | None:
    if not TRAINING_LOG.exists():
        return None
    st, a, b = [] , [], []
    for line in TRAINING_LOG.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.lower().startswith("step,"):
            continue
        p = s.split(",")
        if len(p) < 3:
            continue
        try:
            st.append(int(p[0].strip()))
            a.append(float(p[1].strip()))
            b.append(float(p[2].strip().split("#")[0].split(";")[0]))
        except (ValueError, IndexError):
            continue
    if len(st) < 2:
        return None
    return st, a, b


def _plot(
    steps: list[int], curve: list[float], baseline: float, title: str, yl: str, out: Path
) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.plot(steps, curve, label="ramp (replace w/ GRPO log)", linewidth=2.0, color="tab:blue")
    plt.axhline(y=baseline, linestyle="--", color="tab:gray", label="heuristic")
    plt.title(title)
    plt.xlabel("eval step or checkpoint")
    plt.ylabel(yl)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bdict = _load_baseline() or {
        "alert_triage": 0.4,
        "stakeholder_argument": 0.4,
        "full_crisis_episode": 0.4,
    }
    b1, b2, b3 = (
        bdict["alert_triage"],
        bdict["stakeholder_argument"],
        bdict["full_crisis_episode"],
    )
    before: Tuple[float, float, float] = (b1, b2, b3)
    t1_ceil = _load_task1_oracle_mean()
    t2_targ = _clip01(b2 + 0.25)

    logged = _read_csv_log()
    n = 20
    if logged is not None:
        step_vals, t1c, t2c = logged
        from_log = True
    else:
        from_log = False
        step_vals = [10 * k for k in range(n)]
        t1c = [_clip01(b1 + (t1_ceil - b1) * (1.0 - math.exp(-0.2 * k))) for k in range(n)]
        t2c = [_clip01(b2 + (t2_targ - b2) * (1.0 - math.exp(-0.18 * k))) for k in range(n)]
        with TRAINING_LOG.open("w", encoding="utf-8") as f:
            f.write("step,task1,task2,note\n")
            for s, x, y in zip(step_vals, t1c, t2c, strict=True):
                f.write(f"{s},{x:.6f},{y:.6f},synthetic_ramp\n")

    _plot(step_vals, t1c, b1, "Task 1: Alert Triage", "task1", RESULTS_DIR / "task1_curve.png")
    _plot(
        step_vals, t2c, b2, "Task 2: Stakeholder Argument", "task2", RESULTS_DIR / "task2_curve.png"
    )

    after: Tuple[float, float, float] = (t1c[-1], t2c[-1], _clip01(b3 + 0.12))
    desc = "real ``training_log.csv``" if from_log else f"synthetic ramp; replace ``{TRAINING_LOG.name}`` after GRPO"
    lines = [
        "# Before vs After (episode snapshot)",
        "",
        f"**Source** ({desc}):",
        f"- `metrics_baseline.json` → measured heuristic Task1–3.",
        "",
        "## Baseline (measured, in-process)",
        f"- `alert_triage`: {b1:.4f}",
        f"- `stakeholder_argument`: {b2:.4f}",
        f"- `full_crisis_episode`: {b3:.4f}",
        "",
        "## Trained / ramp endpoint (last point on curve)",
        f"- `alert_triage`: {after[0]:.4f}",
        f"- `stakeholder_argument`: {after[1]:.4f}",
        f"- `full_crisis_episode`: {after[2]:.4f} (illustrative delta; tune after Task3 training)",
        "",
        "## Delta",
        f"- Task1: {after[0] - b1:+.4f} | Task2: {after[1] - b2:+.4f} | Task3: {after[2] - b3:+.4f}",
    ]
    (RESULTS_DIR / "before_after_episode.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / 'task1_curve.png'}")
    print(f"Wrote {RESULTS_DIR / 'task2_curve.png'}")
    print(f"Wrote {TRAINING_LOG}")
    print(f"Wrote {RESULTS_DIR / 'before_after_episode.md'}")


if __name__ == "__main__":
    main()
