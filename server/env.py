from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from server.models import Action, Observation, Reward
from server.seeds import normalize_seed, set_seed

MAX_STEPS = 50


@dataclass
class CyberCrisisEnv:
    seed: int
    rng: Any = field(init=False)
    step_count: int = field(default=0, init=False)
    done: bool = field(default=False, init=False)
    system_health: float = field(default=1.0, init=False)
    threat_level: float = field(default=0.2, init=False)
    infected_nodes: int = field(default=1, init=False)
    total_nodes: int = field(default=10, init=False)
    time_pressure: int = field(default=0, init=False)
    last_action: str = field(default="none", init=False)
    monitor_reduction_next_step: bool = field(default=False, init=False)
    attacker_type: str = field(default="stealth", init=False)
    action_history: list[str] = field(default_factory=list, init=False)
    infection_ratio_history: list[float] = field(default_factory=list, init=False)
    reward_history: list[float] = field(default_factory=list, init=False)
    pending_effects: list[dict[str, Any]] = field(default_factory=list, init=False)
    isolation_active: bool = field(default=False, init=False)
    prev_infected_nodes: int = field(default=1, init=False)
    prev_system_health: float = field(default=1.0, init=False)
    low_infection_streak: int = field(default=0, init=False)
    steps_since_patch: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.seed = normalize_seed(self.seed)
        self.rng = set_seed(self.seed)

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.seed = normalize_seed(seed)
        # Reset RNG first so all downstream state updates are deterministic.
        self.rng = set_seed(self.seed)
        self.step_count = 0
        self.done = False
        self.system_health = 1.0
        self.threat_level = 0.2
        self.infected_nodes = 1
        self.total_nodes = 10
        self.time_pressure = 0
        self.last_action = "none"
        self.monitor_reduction_next_step = False
        self.attacker_type = "stealth" if self.rng.random() < 0.5 else "aggressive"
        self.action_history = []
        self.infection_ratio_history = []
        self.reward_history = []
        self.pending_effects = []
        self.isolation_active = False
        self.prev_infected_nodes = self.infected_nodes
        self.prev_system_health = self.system_health
        self.low_infection_streak = 0
        self.steps_since_patch = 0
        return self._random_observation()

    def step(self, action: Action) -> Dict[str, Any]:
        if self.done:
            return {
                "observation": self._random_observation().model_dump(),
                "reward": self._random_reward().model_dump(),
                "done": True,
                "info": {"message": "Episode already finished"},
            }

        self.step_count += 1
        self.last_action = action.action_type
        self.action_history.append(action.action_type)
        if len(self.action_history) > 6:
            self.action_history.pop(0)
        self.time_pressure = min(50, self.time_pressure + 1)
        self._update_attacker_strategy(action.action_type)
        self._apply_pending_effects()

        if action.action_type == "isolate":
            self.infected_nodes = max(0, self.infected_nodes - 1)
            if self.threat_level > 0.6:
                self.infected_nodes = max(2, self.infected_nodes)
        elif action.action_type == "patch":
            # Strongest long-term effect: direct threat reduction.
            self.threat_level = self.threat_level - 0.2
            self.threat_level = max(0.0, min(1.0, self.threat_level))
        elif action.action_type == "monitor":
            self.monitor_reduction_next_step = True
        elif action.action_type == "ignore":
            self.threat_level = max(0.0, min(1.0, self.threat_level + 0.05))
        elif action.action_type == "communicate":
            pass

        if action.action_type == "patch":
            self.steps_since_patch = 0
        else:
            self.steps_since_patch += 1

        spread = self.threat_level * 0.15
        if self.steps_since_patch >= 3:
            # Without patching, attacker pressure steadily increases spread.
            spread *= 1.0 + min(0.5, 0.05 * (self.steps_since_patch - 2))
        if self.monitor_reduction_next_step:
            spread *= 0.5
            self.monitor_reduction_next_step = False
        self.infected_nodes += int(spread * self.total_nodes)
        self.infected_nodes = min(self.infected_nodes, self.total_nodes)

        self.system_health -= self.infected_nodes * 0.01
        self.system_health = max(0.0, self.system_health)

        self.threat_level += 0.02
        if self.attacker_type == "aggressive":
            self.threat_level += 0.01
        if self.steps_since_patch >= 3:
            # No patch for several steps -> deterministic threat drift upward.
            self.threat_level += min(0.08, 0.01 * (self.steps_since_patch - 2))
        self.threat_level = max(0.0, min(1.0, self.threat_level))

        if self.step_count >= MAX_STEPS:
            self.done = True

        current_infection_ratio = self.infected_nodes / max(1, self.total_nodes)
        if current_infection_ratio < 0.3:
            self.low_infection_streak += 1
        else:
            self.low_infection_streak = 0

        observation = self._random_observation()
        reward = self._random_reward()
        infection_ratio = self.infected_nodes / max(1, self.total_nodes)
        self.infection_ratio_history.append(max(0.0, min(1.0, infection_ratio)))
        self.reward_history.append(max(0.0, min(1.0, reward.total)))
        if len(self.infection_ratio_history) > 5:
            self.infection_ratio_history.pop(0)
        if len(self.reward_history) > 5:
            self.reward_history.pop(0)
        self.prev_infected_nodes = self.infected_nodes
        self.prev_system_health = self.system_health
        return {
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "done": self.done,
            "info": {"step_count": self.step_count},
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "step_count": self.step_count,
            "done": self.done,
            "system_health": self.system_health,
            "threat_level": self.threat_level,
            "infected_nodes": self.infected_nodes,
            "total_nodes": self.total_nodes,
            "time_pressure": self.time_pressure,
            "last_action": self.last_action,
            "attacker_type": self.attacker_type,
            "isolation_active": self.isolation_active,
            "pending_effects": len(self.pending_effects),
            "steps_since_patch": self.steps_since_patch,
        }

    def _apply_pending_effects(self) -> None:
        due: list[dict[str, Any]] = []
        future: list[dict[str, Any]] = []
        for effect in self.pending_effects:
            if effect.get("apply_at_step", 0) <= self.step_count:
                due.append(effect)
            else:
                future.append(effect)
        self.pending_effects = future

        for effect in due:
            effect_type = effect.get("type")
            if effect_type == "patch_reduce_threat":
                # Patch is the main significant threat reducer.
                self.threat_level = max(0.0, min(1.0, self.threat_level - 0.2))
            elif effect_type == "end_isolation":
                self.isolation_active = False

    def _update_attacker_strategy(self, action_type: str) -> None:
        if action_type == "ignore":
            self.attacker_type = "aggressive"
            return

        isolate_count = self.action_history.count("isolate")
        if isolate_count >= 3:
            self.attacker_type = "stealth"

    def _random_observation(self) -> Observation:
        observed_threat_level = self.threat_level + self.rng.uniform(-0.03, 0.03)
        observed_threat_level = max(0.0, min(1.0, observed_threat_level))

        if self.threat_level > 0.5:
            severity = "CRITICAL_ALERT"
        elif self.threat_level > 0.3:
            severity = "WARNING"
        else:
            severity = "NORMAL"

        # 20% deterministic chance to mislabel severity.
        if self.rng.random() < 0.2:
            if severity == "CRITICAL_ALERT":
                severity = "WARNING"
            elif severity == "WARNING":
                severity = "CRITICAL_ALERT"
            else:
                severity = "CRITICAL_ALERT"

        alerts = [severity]

        if self.attacker_type == "stealth":
            # Stealth attacker emits fewer visible alerts.
            if self.rng.random() < 0.5:
                alerts = []
        else:
            # Aggressive attacker emits noisier/high-volume alert stream.
            if alerts:
                alerts = [alerts[0], alerts[0]]

        infection_ratio = self.infected_nodes / max(1, self.total_nodes)
        infection_ratio = max(0.0, min(1.0, infection_ratio))
        if infection_ratio > 0.6:
            status = "critical"
        elif infection_ratio > 0.3:
            status = "warning"
        else:
            status = "stable"

        history_summary = self._history_summary()

        return Observation(
            step=self.step_count,
            system_health=self.system_health,
            threat_level=observed_threat_level,
            alerts=alerts,
            resources_available=max(0, self.total_nodes - self.infected_nodes),
            infection_ratio=infection_ratio,
            status=status,
            history_summary=history_summary,
        )

    def _history_summary(self) -> dict[str, Any]:
        history_len = min(
            len(self.action_history), len(self.infection_ratio_history), len(self.reward_history)
        )
        if history_len == 0:
            return {
                "window_size": 0,
                "recent_action": "none",
                "avg_infection_ratio": 0.0,
                "infection_trend": "flat",
                "avg_reward": 0.0,
                "reward_trend": "flat",
            }

        recent_action = self.action_history[-1]
        avg_infection_ratio = sum(self.infection_ratio_history) / len(self.infection_ratio_history)
        avg_reward = sum(self.reward_history) / len(self.reward_history)

        infection_trend = "flat"
        if len(self.infection_ratio_history) >= 2:
            if self.infection_ratio_history[-1] < self.infection_ratio_history[0]:
                infection_trend = "down"
            elif self.infection_ratio_history[-1] > self.infection_ratio_history[0]:
                infection_trend = "up"

        reward_trend = "flat"
        if len(self.reward_history) >= 2:
            if self.reward_history[-1] > self.reward_history[0]:
                reward_trend = "up"
            elif self.reward_history[-1] < self.reward_history[0]:
                reward_trend = "down"

        return {
            "window_size": history_len,
            "recent_action": recent_action,
            "avg_infection_ratio": round(avg_infection_ratio, 4),
            "infection_trend": infection_trend,
            "avg_reward": round(avg_reward, 4),
            "reward_trend": reward_trend,
        }

    def _random_reward(self) -> Reward:
        # Simple deterministic reward: no extra bonuses/penalties.
        security_score = 1.0 - (self.infected_nodes / max(1, self.total_nodes))
        security_score = max(0.0, min(1.0, security_score))

        uptime_score = max(0.0, min(1.0, self.system_health))

        # Communicate has no reward impact for now.
        trust_score = 0.7
        trust_score = max(0.0, min(1.0, trust_score))

        speed_score = 1.0 - (self.time_pressure / 50.0)
        speed_score = max(0.0, min(1.0, speed_score))

        # Weighted deterministic priority: survival > uptime > behavior.
        total = (
            0.6 * security_score
            + 0.25 * uptime_score
            + 0.1 * speed_score
            + 0.05 * trust_score
        )

        total = max(0.0, min(1.0, total))

        return Reward(
            security_score=security_score,
            uptime_score=uptime_score,
            trust_score=trust_score,
            speed_score=speed_score,
            total=total,
        )
