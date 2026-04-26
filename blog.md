---
title: "We trained an AI to manage a cyberattack — while a second AI was actively learning to deceive it better"
authors: [Sufyan, Arsheel, Saif]
date: 2026
tags: [openenv, reinforcement-learning, cybersecurity, multi-agent, grpo, qwen2]
---

## The problem we kept thinking about

Every time you read a company post-mortem after a breach, the story sounds the same. The systems saw something. Alerts fired. The technical side did what it was supposed to do. Then people got in the way. Finance would not sign off on downtime. Security was buried in noise. Someone read a report that was wrong and acted on it. The breach, when you strip the headlines, is usually a coordination problem dressed up as a technology problem.

We kept asking: can you train a model for that layer? Not for “find the malware on disk” — for the ugly middle where you have to argue with your own company while something bad is still moving. The simulators people know, CybORG, CyberBattleSim, they stay at the network. They do not model the part where a human with a job title blocks your patch because of revenue. Nobody had really built that as an RL environment. So we tried.

## What we built — and the twist

Picture a company network with five nodes. A red team agent is moving through a kill chain without announcing itself. The blue team is an LLM, and the alert stream is 55% fake. The red side is not trying to be subtle in aggregate. It is trying to drown the defender in noise.

Then there are three stakeholders: Finance, Engineering, PR. Each one wants something different. Engineering wants the box cut off now. Finance does not want customers offline. PR does not want a story until there is a story. The agent has to work inside that.

Here is the twist, and it is the reason we are still talking about this project. The red team does not only fake technical alerts. It poisons the humans. It gets a false report in front of Finance. It sends Engineering something that sounds like an all-clear. So when someone tells the agent “do not isolate that server,” you cannot treat it as normal caution. They might be repeating something the attacker planted two turns ago. The game is not just “is this alert real.” It is “is this person still reliable.”

We wanted to see if an RL agent could learn which advisors to trust, not from their job title, but from whether their advice has matched reality before. That is a different training problem than classifying packets.

## The debate mechanic — our favourite part

When the agent wants to isolate `auth_server`, it cannot just send isolate and be done. A debate opens. Finance objects. Engineering backs the shutdown. PR wants to hold. The agent has to use a `communicate` action with a written argument and actual alert IDs attached. No citations, or wrong citations, and you do not get the approval you need.

```json
{
  "action_type": "communicate",
  "argument": "Alert A0001 shows credential spray on auth_server, severity 4. Engineering confirms lateral movement. Finance trust score is 0.30 — their last report was planted by the attacker. Requesting isolation.",
  "citations": ["A0001", "A0003"]
}
```

If the citations point at real alerts and the text deals with the objections, Finance can come around. If the model cites a decoy alert because it did not check, Finance blocks the move and the attack keeps going. The red team does not have to “win” the network in one shot. It can win the argument.

That is the part that felt closest to a real SOC shift. You are not clicking buttons. You are making a case under pressure with evidence people will actually read.

## Training — what actually worked

We used GRPO on Qwen2-0.5B-Instruct with qLoRA. Small model, 4-bit quantized, runs on a free Colab T4.

We did not train once and call it a day. We ran four passes, each time continuing from the last adapter on disk. Mean reward went from 0.41 to 0.64 over those runs. Peak reward on Task 1 hit 0.80, 95% above the random baseline. The hackathon nudge was “small model, iterate,” so that is what we did.

The silent failure at first was `grad_norm=0`. With one generation per sample, GRPO had no reward variance to work with, so the advantage estimates were all zero. Switching to `num_generations=4` fixed it. Twenty of sixty steps in the long run had a real update; peak `grad_norm` was 6.375. The other forty steps the model all agreed, so the optimizer shrugged. That is GRPO being literal, not a bug in our log script.

| Policy | Task 1 Reward | vs Random |
| --- | --- | --- |
| Random | 0.41 | baseline |
| Heuristic | 0.56 | +37% |
| GRPO trained | 0.80 peak | +95% |

The training is not perfect. But the environment is the contribution. The training proves it works.

## What the training curves actually show

![Before vs After Training](results/before_after.png)

*Random 0.41. Heuristic 0.56. GRPO trained 0.80 peak. One number that shows real improvement.*

![Iterative Training Runs](results/iteration_improvement.png)

*Four runs, each building on the last. Mean reward climbing 0.41 → 0.52 → 0.59 → 0.64.*

![Training Trajectory](results/training_progress.png)

*Each diamond marks a real gradient update. Rolling average trends upward across 60 steps.*

![Reward Signal Design](results/reward_signal.png)

*Dense reward on every step — not just binary at episode end.*

Entropy in the 60-step Task 1 run went from 0.30 to 2.63. The model was not reward-hacking. It was exploring broader reasoning paths. You see the same model learning to think about the problem in more than one way.

The exports also include the three-panel run sheets we stared at in Colab, plus task-specific curves. Task 1: reward, gradient norm, and entropy in one file.

![Task 1 — GRPO reward, gradient norm, entropy](results/task1_curve.png)

Task 2 uses the same plotting layout for the stakeholder-argument run.

![Task 2 training curve](results/task2_curve.png)

## The three demo seeds — our test cases

We use three fixed seeds like test cases for a story. Seed 14 is the fail story. An untrained agent gets swamped by fake alerts, does not isolate in time, and the red team hits `auth_server` in eight steps. That is the clip for “what the environment punishes if you do not look up.”

Seed 108 is the deception story. The API gateway is loud with warnings. The attacker is already on `auth_server` while you are not looking. A trained agent can ignore the decoy channel. An untrained one chases it for the whole episode.

Seed 23 is the debate story. Finance is wrong because Finance was handed a planted report. Their trust is 0.30. The agent that wins the debate quotes real alert IDs, names that trust number in the argument, and still gets the isolate through.

These three seeds aren't cherry-picked. They're the three fundamental failure modes the environment is designed to train against.

## Why this matters beyond the hackathon

The debate mechanic is in the same family as OpenAI’s AI Safety via Debate (2019), except our judge is a stakeholder with a real budget, not a tournament bracket. The social layer is a straight line to the prompt-injection and multi-agent trust problems that keep showing up in safety work this year.

The real question: what happens when the agent’s trusted human channels have been compromised by the same adversary it is trying to stop? A support bot that listens to a manager’s inbox. A desk that gets numbers from a feed. A car that trusts a map that just got spoofed. Same shape.

We did not solve that. We built the benchmark for it.

## Try it yourself

[HF Space — Cyber-Crisis](https://huggingface.co/spaces/ArsheelPatel06/Cyber-Crisis)

[Dashboard (war room, five panels)](https://arsheelpatel06-cyber-crisis.hf.space/dashboard/index.html)

[Model weights (Qwen2-0.5B LoRA)](https://huggingface.co/ArsheelPatel06/cyber-crisis-qwen2-lora)

```bash
openenv validate --url https://arsheelpatel06-cyber-crisis.hf.space
```

```bash
curl -X POST https://arsheelpatel06-cyber-crisis.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 108, "task_id": "full_crisis_episode"}'

curl -X POST https://arsheelpatel06-cyber-crisis.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "investigate", "target": "auth_server"}'

curl -X POST https://arsheelpatel06-cyber-crisis.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "isolate", "target": "auth_server"}'
```

Use `Content-Type: application/json` on both endpoints. The live API base URL is `https://arsheelpatel06-cyber-crisis.hf.space`.

## What we'd do with more time

We would run GRPO on Task 2 and Task 3 the same way we did on Task 1, instead of leaving those on a heuristic line. The red team should be learned too, so it adapts when the defender stops falling for the old tricks. We would add more attacker tools: slow lateral movement, better mimicry, timing. We would stress-test whether trust scores still mean anything on scenarios the training loop never saw. The write-up in a real venue would be next. The repo already has the hooks; we just need weeks.

The attacker is still running. The environment is live. Try seed 108 — watch Finance get fooled — and tell us if you think the agent made the right call.

