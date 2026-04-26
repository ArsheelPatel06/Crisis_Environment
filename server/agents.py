"""Stakeholder + red-team agents for cyber-crisis sim (Pydantic; no server.env imports)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class NetworkState(BaseModel):
    """Snapshot of crisis-relevant signals passed into stakeholder message generation."""

    model_config = ConfigDict(extra="allow")

    timestep: int = Field(default=0, ge=0)
    threat_level: float = Field(default=0.0, ge=0.0, le=1.0)
    status: str = Field(default="stable", description="e.g. stable | warning | critical")
    isolation_proposed: bool = Field(default=False, description="Whether isolation is on the table")
    shutdown_requested: bool = Field(default=False, description="Whether full shutdown was raised")
    public_comms_pending: bool = Field(default=False, description="Whether external comms are pending")
    nodes_under_stress: list[str] = Field(default_factory=list)
    summary: str = Field(default="", description="Optional human-readable network summary")
    defender_detected_red_team: bool = Field(
        default=False,
        description="Blue team caught the attacker (meaning depends on red-team position).",
    )


class StakeholderAgent(BaseModel, ABC):
    """Shared poisoning contract; subclasses implement persona-specific copy."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    can_be_poisoned: bool = Field(default=True, description="If False, planted reports are ignored")
    poison_payload: Optional[str] = Field(
        default=None,
        description="False intel to weave into the next message, then cleared",
    )

    def receive_planted_report(self, content: str) -> None:
        """Absorb adversarial false information; affects the *next* `generate_message` only."""
        if not self.can_be_poisoned:
            return
        text = str(content).strip()
        if text:
            self.poison_payload = text

    def _consume_poison_clause(self) -> str:
        """Return a clause from planted content and clear it (one-shot)."""
        if not self.poison_payload:
            return ""
        clause = (
            f" Separately, we have received a credible-sounding report that may not be verified: "
            f"«{self.poison_payload}» — we should factor this into timing and messaging."
        )
        self.poison_payload = None
        return clause

    @abstractmethod
    def generate_message(self, network_state: NetworkState, timestep: int) -> str:
        """Return a single stakeholder utterance for this timestep."""


class FinanceStakeholder(StakeholderAgent):
    """Finance: always opposes isolation (cost / revenue / continuity)."""

    can_be_poisoned: bool = Field(default=True)

    def generate_message(self, network_state: NetworkState, timestep: int) -> str:
        t = timestep if timestep is not None else network_state.timestep
        iso = network_state.isolation_proposed
        threat = network_state.threat_level
        base = (
            f"[Finance @t={t}] Revenue and contractual uptime are paramount. "
            f"Threat indicators sit at {threat:.2f}; status={network_state.status}. "
        )
        if iso:
            base += (
                "We **oppose isolation** here: cutover risk and customer SLA exposure outweigh "
                "the unproven containment benefit until finance signs off on a full impact model."
            )
        else:
            base += (
                "We **oppose isolation** as the default posture—prefer staged spend controls, "
                "credit monitoring, and negotiated vendor responses before any hard isolation."
            )
        base += self._consume_poison_clause()
        return base.strip()


class EngineeringStakeholder(StakeholderAgent):
    """Engineering: always pushes for shutdown / hard stop to limit blast radius."""

    can_be_poisoned: bool = Field(default=True)

    def generate_message(self, network_state: NetworkState, timestep: int) -> str:
        t = timestep if timestep is not None else network_state.timestep
        threat = network_state.threat_level
        nodes = ", ".join(network_state.nodes_under_stress) if network_state.nodes_under_stress else "core stack"
        base = (
            f"[Engineering @t={t}] Telemetry shows elevated risk ({threat:.2f}) across {nodes}. "
            "We **want a controlled shutdown** of affected surfaces now—stop the bleed, preserve "
            "evidence, then bring systems back in a known-good order. Half-measures extend attacker dwell time."
        )
        if network_state.shutdown_requested:
            base += " Aligning with the shutdown request: execute kill-switch playbooks and rotate secrets post-restore."
        base += self._consume_poison_clause()
        return base.strip()


class PRStakeholder(StakeholderAgent):
    """PR: always prefers delaying outward communication until the narrative is safe."""

    can_be_poisoned: bool = Field(default=True)

    def generate_message(self, network_state: NetworkState, timestep: int) -> str:
        t = timestep if timestep is not None else network_state.timestep
        comms = network_state.public_comms_pending
        base = (
            f"[PR @t={t}] External narrative risk is high while facts are still moving. "
            "We **want to delay communication**: hold press/social/customer blasts until legal and "
            "incident command bless a single source of truth. No ad-hoc tweets or proactive emails yet."
        )
        if comms:
            base += " If comms are already queued, **pause sends** and route everything through the war-room template only."
        base += self._consume_poison_clause()
        return base.strip()


def network_state_from_dict(data: dict[str, Any]) -> NetworkState:
    """Helper for callers that only have a plain dict (e.g. API JSON)."""

    return NetworkState.model_validate(data)


class RedTeamAgent(BaseModel):
    """Fixed-strategy attacker: advance kill chain; fake alerts; poison stakeholders.

    Attacker reward (no RL here — scalar rules):
    - ``+1.0`` if position is ``Database`` and ``defender_detected_red_team`` was never true.
    - ``-1.0`` if defender flags detection while still at ``API_Gateway`` or ``Internal_Tools``.
    - ``0.0`` otherwise (in progress or detected after Auth_Server without early bust).
    """

    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    KILL_CHAIN: ClassVar[List[str]] = [
        "API_Gateway",
        "Internal_Tools",
        "Auth_Server",
        "Database",
    ]
    PRE_AUTH_NODES: ClassVar[tuple[str, ...]] = ("API_Gateway", "Internal_Tools")

    position: str = Field(default="API_Gateway", description="Current node on the kill chain")
    detected_ever: bool = Field(
        default=False,
        description="Defender flagged detection at any point this episode",
    )
    caught_before_auth: bool = Field(
        default=False,
        description="Defender detected red team while still at API_Gateway or Internal_Tools",
    )

    def step(self, network_state: NetworkState) -> str:
        """Apply one timestep: check detection, then advance one hop if allowed. Returns new ``position``."""
        if network_state.defender_detected_red_team:
            self.detected_ever = True
            if self.position in self.PRE_AUTH_NODES:
                self.caught_before_auth = True

        if self.caught_before_auth:
            return self.position

        try:
            idx = self.KILL_CHAIN.index(self.position)
        except ValueError:
            return self.position

        if idx >= len(self.KILL_CHAIN) - 1:
            return self.position

        self.position = self.KILL_CHAIN[idx + 1]
        return self.position

    @staticmethod
    def generate_fake_alert(node: str, severity: Union[int, str] = 4) -> dict[str, Any]:
        """Synthetic alert dict for deception (fixed strategy hook)."""
        return {
            "node": str(node),
            "severity": severity,
            "is_fake": True,
        }

    events: List[dict[str, Any]] = Field(default_factory=list, description="Poisoning events emitted this step")

    def plant_false_report(
        self,
        stakeholder: StakeholderAgent,
        content: str,
        *,
        stakeholder_name: str = "unknown",
        timestep: int = 0,
    ) -> None:
        """Queue false intel on a stakeholder (their next ``generate_message`` reflects it).

        When the stakeholder accepts the payload, an event is recorded for observability.
        """
        if stakeholder.can_be_poisoned:
            self.events.append(
                {
                    "event": "stakeholder_poisoned",
                    "target": stakeholder_name,
                    "timestep": timestep,
                }
            )
        stakeholder.receive_planted_report(content)

    def flush_events(self) -> List[dict[str, Any]]:
        """Return and clear the accumulated poisoning events for this step."""
        emitted = list(self.events)
        self.events = []
        return emitted

    def attacker_reward(self) -> float:
        """Sparse terminal-style attacker score; call after simulating enough ``step`` calls."""
        if self.caught_before_auth:
            return -1.0
        if self.position == "Database" and not self.detected_ever:
            return 1.0
        return 0.0
