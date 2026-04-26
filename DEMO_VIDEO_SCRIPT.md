# Demo video script — Cyber Crisis (OpenEnv)

Use as a read-along while screen recording. Target **4–6 minutes** for judges; a **90-second** cut is listed at the end.

**Before you record:** T4 or quiet room for voice; 1080p; hide unrelated browser tabs; test mic once.

| Asset | URL |
|------|-----|
| Space | [huggingface.co/spaces/ArsheelPatel06/Cyber-Crisis](https://huggingface.co/spaces/ArsheelPatel06/Cyber-Crisis) |
| War room | [arsheelpatel06-cyber-crisis.hf.space/dashboard/index.html](https://arsheelpatel06-cyber-crisis.hf.space/dashboard/index.html) |
| OpenEnv check | `openenv validate --url https://arsheelpatel06-cyber-crisis.hf.space` |
| LoRA | [huggingface.co/ArsheelPatel06/cyber-crisis-qwen2-lora](https://huggingface.co/ArsheelPatel06/cyber-crisis-qwen2-lora) |

**Terminology on camera:** the live API uses `argument_text` in JSON, not `argument` (if you show code).

---

## SCENE 0 — Title (5–10 sec)

**Show:** Project title on a simple slide, or the Space page loading.

**Say:** “This is Cyber Crisis — an OpenEnv incident-response environment where a blue-team LLM handles a live attack, fake alerts, and real office politics, trained with GRPO on a half-billion-parameter model.”

---

## SCENE 1 — The problem (25–40 sec)

**Show:** A single slide or Notion with three short lines (or speak over a static diagram: network + people icons).

- Post-mortems: “alerts fired, people made bad calls.”
- “Coordination, not just malware.”
- “We asked: can RL train the human layer, not just packet rules?”

**Say:** “Every big breach post-mortem says the same thing. The tools saw something, but teams disagreed, Finance pushed back, someone trusted a bad report. The failure is often coordination, not a missing signature. We wanted an RL environment that goes past the network, into Finance, Engineering, and PR, with a red team that fools both the alerts and the humans. Classic cyber sims stop at the firewall. We did not.”

---

## SCENE 2 — What the environment is (30–45 sec)

**Show:** The **Hugging Face Space** UI, or a simple diagram: 5 nodes in a line; small labels: API, tools, **auth**, DB, comms. Optionally flash “~55% alerts are fake” as on-screen text.

**Say:** “The company has a small network: five nodes. A red team moves through a quiet kill chain. The blue team only sees a noisy stream — roughly half the alerts are decoys. On top of that, three internal stakeholders have different goals. Engineering wants to cut access. Finance does not want downtime. PR does not want headlines. The agent is not classifying a PCAP file. It is managing a crisis with conflicting humans.”

---

## SCENE 3 — The twist (20–30 sec)

**Show:** Same screen; add one line: “Attacker also poisons *human* messages.”

**Say:** “Here is the twist. The attacker does not only send fake *technical* alerts. It can plant a story with Finance, or a fake all-clear in Engineering. So when someone says do not take that server offline, that might be caution — or the adversary’s script from two turns ago. The model has to reason about *trust*, not just severity.”

---

## SCENE 4 — OpenEnv (15–25 sec)

**Show:** A terminal. Run:

```bash
openenv validate --url https://arsheelpatel06-cyber-crisis.hf.space
```

**Say:** “We ship a live API on Hugging Face. OpenEnv validation — six of six checks — proves the server speaks the right health, OpenAPI, metadata, and MCP contract. That matters if another agent is going to run against the same spec.”

*If the command is slow, record a successful run in advance and cut the wait.*

---

## SCENE 5 — War room dashboard (40–60 sec)

**Show:** [Dashboard](https://arsheelpatel06-cyber-crisis.hf.space/dashboard/index.html) full width. **Zoom** so the jury can read **alerts panel**, **trust or stakeholder** strip, and **threat** / step counter if present.

**Say:** “We built a war room so you are not only reading raw JSON. Here you can see the alert list, the stakeholder pressure, and how the episode evolves step by step. The same run you would feed to an eval agent is also legible to a person.”

**Optional micro-move:** If the dashboard has a seed field or you drive actions — set **seed 108**, press play or step once, **pause**.

**Say (if seed 108):** “This is seed 108, our deception case. The API gateway is noisy, but the real move is on auth. A naive policy chases the loud node. A trained one learns to look past the decoy and argue with evidence.”

---

## SCENE 6 — Three seeds (20–30 sec)

**Show:** A simple on-screen list (or this repo’s [DEMO_SEEDS.md](DEMO_SEEDS.md)).

| Seed | Story |
|------|--------|
| 14 | Fail: attacker reaches auth in ~8 steps if you drown in noise |
| 108 | Deception: gateway screams; attacker already at auth |
| 23 | Debate: Finance resists; you need citations and a good argument |

**Say:** “We are not picking lucky runs. We publish three fixed seeds. Fourteen is the fail case — drown, hesitate, you lose. One-oh-eight is the head-fake. Twenty-three is where Finance is wrong and you have to *win a debate* with real alert IDs and a tight argument. Those are the failure modes the env is designed to teach.”

---

## SCENE 7 — Debate mechanic (30–45 sec)

**Show:** A code or JSON slide (your blog snippet is fine). **Highlight** `action_type`, `citations` array, and the one-line argument.

**Say:** “To isolate a critical node you often cannot just click isolate. A debate opens. The agent has to *communicate* with a written argument and **real** alert IDs. If the citation is a decoy, Finance can block the move, and the red team wins the round without owning every host. That is the closest we got to a real SOC: make the case, cite the right tickets, not just a confident paragraph.”

---

## SCENE 8 — Training (35–50 sec)

**Show:** In order, **one chart per 5–6 seconds** (or a single montage):  
`results/before_after.png` → `iteration_improvement.png` → `training_progress.png` → (optional) `task1_curve.png`.

**Say:** “We used GRPO on Qwen2 half-B with four-bit qLoRA — it fits a free Colab T4. We did not do one long run. We **iterated**: each run started from the last LoRA. Mean Task One reward went from the random regime up to about 0.64 over runs, with peaks around 0.8, ninety-five percent above the random baseline. The early gotcha was `grad_norm` stuck at zero. One completion per step meant no reward variance. We used **four** generations per prompt, so GRPO had a signal. Not every step updates — the curve shows where real gradients land — but the line trends up.”

---

## SCENE 9 — Why it matters (15–20 sec)

**Show:** You on camera, or a single “research” slide with 2 bullets.

**Say:** “The debate line connects to older ‘AI safety through debate’ work, but the judge is a P-and-L, not a chalkboard. The same structure shows up in prompt injection and multi-agent systems when **trusted** channels are wrong on purpose. We are not saying we fixed that. We are saying here is a **trainable** benchmark and a live Space you can point agents at.”

---

## SCENE 10 — Close + CTA (20–30 sec)

**Show:** Space URL + your Colab or GitHub (optional QR off-camera).

**Say:** “The environment is live on Hugging Face. The LoRA is public. The Colab in our repo replays the training. If you have five minutes, open seed one-oh-eight on the dashboard and watch a loud gateway against a quiet auth. Tell us if you would have made the same call. Thanks.”

**End card (5 sec):** Title, [Space](https://huggingface.co/spaces/ArsheelPatel06/Cyber-Crisis), team names, OpenEnv + HF hackathon 2026.

---

## 90-SECOND FAST CUT (optional)

| Sec | Show | Say (very short) |
|-----|------|------------------|
| 0–10 | Title + Space | “Cyber Crisis — multi-agent crisis sim with a trained half-B policy.” |
| 10–25 | Dashboard, seed 108 | “Noisy API, real risk on auth — the env teaches ignoring decoys.” |
| 25–40 | `openenv validate` (clip or screenshot) | “Six of six on the spec.” |
| 40–55 | Debate JSON on screen | “You cite real alerts or you lose the debate.” |
| 55–75 | Before/after or iteration figure | “GRPO plus iteration on a T4, big lift over random.” |
| 75–90 | Space URL | “Link in the repo — try the seeds.” |

---

## Recording tips

- **One tab for demo:** Dashboard or Space; do not switch twelve times in ten seconds.  
- **B-roll:** If you are shy on camera, voiceover + full-screen browser is fine.  
- **Loud parts:** When you read JSON, **slow down**; spell “one-zero-eight” for seed 108.  
- **Filler to cut:** long pip installs, model download, failed validate — use a pre-baked good take.

---

## Checklist the day of

- [ ] `openenv validate` passes in recording environment  
- [ ] Dashboard loads; pick one primary seed to demo (108 or 23)  
- [ ] Charts exported to `results/` in case you present offline  
- [ ] **Authors line** in video matches your submission (blog lists Sufyan, Arsheel, Saif)  
- [ ] End card shows correct year if the hackathon requires it (2026)
