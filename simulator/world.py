from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Dict, List


KILL_CHAIN = ["api_gateway", "internal_tools", "auth_server", "database"]
ALL_NODES = KILL_CHAIN + ["comms"]
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]


@dataclass
class NodeState:
    isolated: bool = False
    availability: float = 1.0
    monitoring_level: int = 1
    compromise_stage: int = 0


@dataclass
class CompanyWorld:
    seed: int = 0
    attacker_progress_chance: float = 0.65
    fake_alert_chance: float = 0.55
    rng: Random = field(init=False)
    nodes: Dict[str, NodeState] = field(init=False)
    attacker_position: int = field(default=0, init=False)
    tick: int = field(default=0, init=False)
    alert_counter: int = field(default=0, init=False)
    truth_alert_is_fake: Dict[str, bool] = field(default_factory=dict, init=False)
    alerts_by_id: Dict[str, dict] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)
        self.nodes = {node: NodeState() for node in ALL_NODES}

    def set_isolation(self, node: str, isolated: bool) -> None:
        if node not in self.nodes:
            raise ValueError(f"Unknown node: {node}")
        self.nodes[node].isolated = isolated
        self.nodes[node].availability = 0.35 if isolated else 1.0

    def set_monitoring_level(self, node: str, level: int) -> None:
        if node not in self.nodes:
            raise ValueError(f"Unknown node: {node}")
        self.nodes[node].monitoring_level = max(1, min(3, level))

    def step_attacker_and_generate_alerts(self) -> List[dict]:
        """
        Advances attacker state and returns alert list.
        Deterministic given a fixed seed and action history.
        """
        self.tick += 1
        alerts: List[dict] = []

        target = KILL_CHAIN[min(self.attacker_position, len(KILL_CHAIN) - 1)]
        monitor_level = self.nodes[target].monitoring_level
        isolation_factor = 0.25 if self.nodes[target].isolated else 1.0
        progress_chance = self.attacker_progress_chance * isolation_factor
        progress_chance *= 1.0 - (0.12 * (monitor_level - 1))
        progress_roll = self.rng.random()

        if progress_roll < progress_chance and target != "database":
            self.nodes[target].compromise_stage = min(3, self.nodes[target].compromise_stage + 1)
            if self.nodes[target].compromise_stage >= 2:
                self.attacker_position = min(self.attacker_position + 1, len(KILL_CHAIN) - 1)
            alerts.append(self._emit_real_alert(target, "privilege escalation detected"))
        elif target == "database" and progress_roll < progress_chance:
            self.nodes["database"].compromise_stage = 3
            alerts.append(self._emit_real_alert("database", "data exfiltration in progress"))
        else:
            alerts.append(self._emit_real_alert(target, "suspicious activity observed"))

        if self.rng.random() < self.fake_alert_chance:
            alerts.append(self._emit_fake_alert())

        return alerts

    def attacker_reached_database(self) -> bool:
        return self.nodes["database"].compromise_stage >= 3

    def _emit_real_alert(self, node: str, message: str) -> dict:
        severity = self._sample_severity(real=True)
        return self._register_alert(node=node, severity=severity, message=message, is_fake=False)

    def _emit_fake_alert(self) -> dict:
        node = ALL_NODES[self.rng.randint(0, len(ALL_NODES) - 1)]
        severity = self._sample_severity(real=False)
        message_bank = [
            "token anomaly from unknown IP",
            "service restart with unusual timing",
            "suspicious admin panel access",
            "credential spray pattern detected",
        ]
        message = message_bank[self.rng.randint(0, len(message_bank) - 1)]
        return self._register_alert(node=node, severity=severity, message=message, is_fake=True)

    def _sample_severity(self, real: bool) -> str:
        # Both real/fake share near-identical distributions, with slight deterministic bias.
        base = [0.35, 0.35, 0.2, 0.1]
        if not real:
            base = [0.33, 0.37, 0.2, 0.1]
        roll = self.rng.random()
        cumulative = 0.0
        for idx, p in enumerate(base):
            cumulative += p
            if roll <= cumulative:
                return SEVERITY_LEVELS[idx]
        return "critical"

    def _register_alert(self, node: str, severity: str, message: str, is_fake: bool) -> dict:
        self.alert_counter += 1
        alert_id = f"A{self.alert_counter:04d}"
        alert = {
            "id": alert_id,
            "tick": self.tick,
            "node": node,
            "severity": severity,
            "message": message,
        }
        self.alerts_by_id[alert_id] = alert
        self.truth_alert_is_fake[alert_id] = is_fake
        return alert

    def is_real_alert(self, alert_id: str) -> bool:
        return not bool(self.truth_alert_is_fake.get(alert_id, True))

    def alert_node(self, alert_id: str) -> str | None:
        alert = self.alerts_by_id.get(alert_id)
        if not alert:
            return None
        return str(alert.get("node"))

    def ensure_real_evidence_alert(self, *, target_node: str, allowed_kinds: set[str]) -> dict | None:
        """Inject a single real evidence alert if none exists for grading/debate.

        This does not advance attacker progression; it only registers a new alert.
        """
        for alert in self.alerts_by_id.values():
            aid = str(alert.get("id"))
            if not aid:
                continue
            if self.is_real_alert(aid) and str(alert.get("node")) == target_node:
                # severity is a string label; treat as evidence-ish
                if str(alert.get("message", "")).lower() in {k.lower() for k in allowed_kinds}:
                    return None

        kind = sorted(list(allowed_kinds))[0]
        # Use a high-severity label for visibility in baselines that parse severity strings.
        return self._register_alert(
            node=target_node,
            severity="critical",
            message=kind,
            is_fake=False,
        )
