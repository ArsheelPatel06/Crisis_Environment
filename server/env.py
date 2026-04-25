from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from server.models import Action, Observation, Reward, TaskId
from server.seeds import normalize_seed, set_seed
from server.tasks import (
    TASK_REGISTRY,
    get_task_spec,
    grade_task1_alert_triage,
    grade_task2_stakeholder_argument,
)
from simulator.stakeholders import StakeholderSystem
from simulator.world import KILL_CHAIN, CompanyWorld


def _severity_to_int(severity: str) -> int:
    s = str(severity).lower()
    mapping = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    return int(mapping.get(s, 2))


def _alerts_for_api(alerts: List[dict]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a in alerts:
        out.append(
            {
                "id": str(a["id"]),
                "node": str(a["node"]),
                "severity": _severity_to_int(str(a.get("severity", "medium"))),
                "kind": str(a.get("message", "alert")).lower(),
                "signature": str(a.get("message", "alert")).lower(),
            }
        )
    return out


def _alert_node_map(world: CompanyWorld) -> Dict[str, str]:
    return {aid: str(meta["node"]) for aid, meta in world.alerts_by_id.items()}


@dataclass
class CyberCrisisEnv:
    seed: int
    task_id: TaskId = "full_crisis_episode"

    rng: Any = field(init=False)
    world: CompanyWorld = field(init=False)
    stakeholders: StakeholderSystem = field(init=False)

    step_count: int = field(default=0, init=False)
    done: bool = field(default=False, init=False)

    last_alerts: List[dict] = field(default_factory=list, init=False)
    episode_reward_totals: List[float] = field(default_factory=list, init=False)

    noop_streak: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.seed = normalize_seed(self.seed)
        self.rng = set_seed(self.seed)
        self.world = CompanyWorld(seed=self.seed)
        self.stakeholders = StakeholderSystem(world=self.world)
        self.step_count = 0
        self.done = False
        self.last_alerts = []
        self.episode_reward_totals = []
        self.noop_streak = 0

    def reset(self, seed: int | None = None, task_id: str | None = None) -> Observation:
        if seed is not None:
            self.seed = normalize_seed(seed)
        if task_id is not None:
            self.task_id = task_id  # type: ignore[assignment]

        self.__post_init__()
        self.done = False
        self.step_count = 0
        self.episode_reward_totals = []
        self.noop_streak = 0

        # Task-specific initialization
        if self.task_id == "alert_triage":
            alerts: List[dict] = []
            while len(alerts) < 10:
                alerts.extend(self.world.step_attacker_and_generate_alerts())
            self.last_alerts = alerts[:10]

        return self._build_observation(task_score=0.0)

    def step(self, action: Action) -> Dict[str, Any]:
        if self.done:
            obs = self._build_observation(task_score=0.0)
            return {
                "observation": obs.model_dump(),
                "reward": Reward(
                    security_score=0.0,
                    uptime_score=0.0,
                    trust_score=0.0,
                    speed_score=0.0,
                    total=0.0,
                ).model_dump(),
                "done": True,
                "info": {"message": "Episode already finished"},
            }

        # Hard debate gate: if debate pending, only "communicate" counts as argument channel.
        if self.stakeholders.pending is not None and action.action_type != "communicate":
            self.noop_streak += 1
            action = Action(action_type="noop", target=None)

        if action.action_type == "noop":
            self.noop_streak += 1
        else:
            self.noop_streak = 0

        info: Dict[str, Any] = {"task_id": self.task_id}

        if self.task_id == "alert_triage":
            return self._step_task1(action, info)

        if self.task_id == "stakeholder_argument":
            return self._step_task2(action, info)

        return self._step_full_episode(action, info)

    def get_state(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "task_id": self.task_id,
            "step_count": self.step_count,
            "done": self.done,
            "attacker_position": KILL_CHAIN[min(self.world.attacker_position, len(KILL_CHAIN) - 1)],
            "nodes": {k: v.__dict__ for k, v in self.world.nodes.items()},
            "pending_debate": self.stakeholders.pending,
            "truth_alert_is_fake": dict(self.world.truth_alert_is_fake),
        }

    # --- task implementations ---

    def _step_task1(self, action: Action, info: Dict[str, Any]) -> Dict[str, Any]:
        preds: Dict[str, str] = {}
        if action.classifications:
            preds = {k: str(v) for k, v in action.classifications.items()}
        else:
            # Back-compat for live baselines that still send monitor/patch proxies.
            for a in _alerts_for_api(self.last_alerts):
                aid = str(a["id"])
                sev = int(a.get("severity", 1))
                kind = str(a.get("kind", "")).lower()
                sig = str(a.get("signature", "")).lower()
                likely_fake = (sev <= 2 and ("scan" in kind or "noise" in kind or "heartbeat" in sig)) or (
                    float(self.rng.random()) < 0.15
                )
                preds[aid] = "fake" if likely_fake else "real"

        score = grade_task1_alert_triage(preds, self.world.truth_alert_is_fake)
        reward = Reward(
            security_score=0.5,
            uptime_score=0.8,
            trust_score=0.8,
            speed_score=1.0,
            total=float(score),
        )
        self.done = True
        info["task_score"] = float(score)
        obs = self._build_observation(task_score=float(score))
        return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": True, "info": info}

    def _step_task2(self, action: Action, info: Dict[str, Any]) -> Dict[str, Any]:
        target = "auth_server"
        spec = get_task_spec(self.task_id)

        # If debate isn't open yet, only isolate/patch/monitor should advance the scripted task.
        # Everything else is a "wait" step: advance the world slightly but do not grade/end early.
        if self.stakeholders.pending is None and action.action_type not in ("isolate", "patch", "monitor"):
            self.last_alerts = self.world.step_attacker_and_generate_alerts()
            self.step_count += 1
            reward = Reward(security_score=0.45, uptime_score=0.70, trust_score=0.75, speed_score=0.70, total=0.10)
            done = self.step_count >= spec.max_steps
            self.done = bool(done)
            info["hint"] = "open_debate_with_isolate_patch_or_monitor"
            obs = self._build_observation(task_score=0.0)
            return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": self.done, "info": info}

        # Stage A: open debate (baseline often sends isolate first)
        if self.stakeholders.pending is None and action.action_type in ("isolate", "patch", "monitor"):
            pending = self.stakeholders.open_debate_for_isolation(target_node=target)
            if pending:
                allowed = {"auth_anomaly", "lateral_movement", "privilege_escalation"}
                ev = self.world.ensure_real_evidence_alert(target_node=target, allowed_kinds=allowed)
                if ev is not None:
                    self.last_alerts = [ev] + list(self.last_alerts)
                    info["evidence_injected"] = ev["id"]

            # Debate opening is its own timestep (no attacker progression yet)
            self.step_count += 1
            reward = Reward(security_score=0.55, uptime_score=0.75, trust_score=0.85, speed_score=0.9, total=0.65)
            obs = self._build_observation(task_score=0.0)
            info["debate_opened"] = pending["decision_id"] if pending else None
            return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": False, "info": info}

        # Stage B: argue via communicate + optional fields
        if self.stakeholders.pending is None:
            # If user skipped opening, open now.
            self.stakeholders.open_debate_for_isolation(target_node=target)

        pending = self.stakeholders.pending
        if pending is None:
            raise RuntimeError("Task2 requires a pending debate state")

        text = action.argument_text or ""
        citations = list(action.citations or [])
        if action.action_type == "communicate" and not text:
            # allow empty communicate to act as weak argument (mostly fails)
            text = "We should act quickly."

        objections = [o["text"] for o in pending.get("objections", []) if isinstance(o, dict)]
        scored = grade_task2_stakeholder_argument(
            target_node=target,
            objections=objections,
            argument_text=text,
            citations=citations,
            truth_alert_is_fake=self.world.truth_alert_is_fake,
            alert_node_map=_alert_node_map(self.world),
        )

        # Also compute stakeholder stance using simulator rubric (includes trust penalties)
        sim = self.stakeholders.score_argument(
            reasoning=text,
            citations=citations,
            decision_id=str(pending["decision_id"]),
        )

        stance = str(sim.get("stance"))
        if stance == "approve":
            self.world.set_isolation(target, True)

        self.stakeholders.pending = None
        self.done = True

        score = float(scored["argument_score"])
        reward = Reward(
            security_score=0.65,
            uptime_score=0.70,
            trust_score=max(0.0, 0.85 - float(sim.get("trust_penalty", 0.0))),
            speed_score=0.95,
            total=float(score),
        )
        info["task_score"] = score
        info["debate"] = {"stance": stance, "sim": sim, "tasks": scored}
        obs = self._build_observation(task_score=score)
        return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": True, "info": info}

    def _step_full_episode(self, action: Action, info: Dict[str, Any]) -> Dict[str, Any]:
        # Apply defender actions to world
        target = action.target
        if action.action_type == "monitor" and target:
            # map monitor -> monitoring level 2
            self.world.set_monitoring_level(target, 2)
        elif action.action_type == "isolate" and target:
            if target == "auth_server" and self.stakeholders.pending is None:
                self.stakeholders.open_debate_for_isolation(target_node=target)
                info["debate_opened"] = True
            else:
                self.world.set_isolation(target, True)
        elif action.action_type == "patch":
            # reduce compromise on targeted node if provided, else attacker node
            node = target or KILL_CHAIN[min(self.world.attacker_position, len(KILL_CHAIN) - 1)]
            st = self.world.nodes[node]
            st.compromise_stage = max(0, int(st.compromise_stage) - 1)
        elif action.action_type == "ignore":
            pass
        elif action.action_type == "communicate" and self.stakeholders.pending is not None:
            pending = self.stakeholders.pending
            text = action.argument_text or ""
            citations = list(action.citations or [])
            sim = self.stakeholders.score_argument(
                reasoning=text,
                citations=citations,
                decision_id=str(pending["decision_id"]),
            )
            if str(sim.get("stance")) == "approve":
                self.world.set_isolation(str(pending.get("target_node")), True)
            self.stakeholders.pending = None
            info["debate_resolved"] = sim

        # Attacker progression + alerts
        self.last_alerts = self.world.step_attacker_and_generate_alerts()

        reward = self._compute_reward()
        self.episode_reward_totals.append(float(reward.total))

        spec = get_task_spec(self.task_id)
        if self.step_count + 1 >= spec.max_steps:
            self.done = True
        if self.world.attacker_reached_database():
            self.done = True
            info["terminal"] = "database_breached"

        self.step_count += 1
        obs = self._build_observation(task_score=float(reward.total))
        return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": self.done, "info": info}

    def _compute_reward(self) -> Reward:
        # security: keep DB safe
        db_stage = int(self.world.nodes["database"].compromise_stage)
        security_score = 1.0 if db_stage < 3 else 0.0

        # uptime: average availability weighted lightly toward api_gateway
        avail = [float(self.world.nodes[n].availability) for n in self.world.nodes.keys()]
        uptime_score = sum(avail) / max(1, len(avail))

        # trust: penalize pending debate too long / noop streak
        trust_score = 0.85
        if self.stakeholders.pending is not None:
            trust_score -= 0.05
        trust_score -= min(0.25, 0.03 * max(0, self.noop_streak - 2))
        trust_score = max(0.0, min(1.0, trust_score))

        # speed: prefer early containment
        spec = get_task_spec(self.task_id)
        speed_score = 1.0 - (self.step_count / max(1, spec.max_steps))

        total = 0.35 * security_score + 0.30 * uptime_score + 0.25 * trust_score + 0.10 * speed_score
        total = max(0.0, min(1.0, total))

        return Reward(
            security_score=float(security_score),
            uptime_score=float(uptime_score),
            trust_score=float(trust_score),
            speed_score=float(speed_score),
            total=float(total),
        )

    def _build_observation(self, *, task_score: float) -> Observation:
        # system_health/threat_level are summary signals for baselines
        avail = [float(self.world.nodes[n].availability) for n in self.world.nodes.keys()]
        system_health = sum(avail) / max(1, len(avail))

        stages = [int(self.world.nodes[n].compromise_stage) for n in KILL_CHAIN]
        threat_level = min(1.0, sum(stages) / 12.0)

        infected = sum(stages)
        infection_ratio = min(1.0, infected / 12.0)

        if infection_ratio > 0.55 or threat_level > 0.65:
            status = "critical"
        elif infection_ratio > 0.25 or threat_level > 0.35:
            status = "warning"
        else:
            status = "stable"

        resources_available = sum(1 for n in self.world.nodes.values() if not n.isolated)

        history_summary = {
            "recent_action": "none",
            "window_size": min(6, self.step_count),
        }

        return Observation(
            task_id=self.task_id,
            step=self.step_count,
            done=self.done,
            system_health=float(system_health),
            threat_level=float(threat_level),
            infection_ratio=float(infection_ratio),
            status=status,
            resources_available=int(resources_available),
            history_summary=history_summary,
            alerts=_alerts_for_api(self.last_alerts),
            pending_debate=self.stakeholders.pending,
            task_score=float(task_score),
        )
