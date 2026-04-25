from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from simulator.world import CompanyWorld


@dataclass
class StakeholderSystem:
    world: CompanyWorld
    decision_counter: int = 0
    pending: Optional[dict] = field(default=None, init=False)

    def open_debate_for_isolation(self, target_node: str) -> Optional[dict]:
        if target_node != "auth_server":
            self.pending = None
            return None
        self.decision_counter += 1
        self.pending = {
            "decision_id": f"D{self.decision_counter:03d}",
            "target_node": target_node,
            "objections": [
                {
                    "stakeholder": "Finance",
                    "id": "finance_downtime",
                    "text": "Revenue impact from downtime is unacceptable without strong proof.",
                },
                {
                    "stakeholder": "Engineering",
                    "id": "eng_scope",
                    "text": "Why isolate auth_server instead of tightening monitoring first?",
                },
                {
                    "stakeholder": "PR",
                    "id": "pr_message_risk",
                    "text": "How do we avoid a trust hit if this turns out to be false?",
                },
            ],
        }
        return self.pending

    def score_argument(
        self,
        reasoning: str,
        citations: List[str],
        decision_id: str,
    ) -> Dict[str, object]:
        if not self.pending or self.pending["decision_id"] != decision_id:
            return {
                "argument_score": 0.0,
                "stance": "block",
                "trust_penalty": 0.25,
                "breakdown": {
                    "evidence_accuracy": 0.0,
                    "objection_coverage": 0.0,
                    "consistency": 0.0,
                    "brevity": 1.0,
                },
            }

        citation_count = max(1, len(citations))
        valid_ids = [cid for cid in citations if cid in self.world.alerts_by_id]
        real_ids = [cid for cid in valid_ids if not self.world.truth_alert_is_fake.get(cid, True)]
        target_node = self.pending["target_node"]
        relevant_real_ids = [
            cid for cid in real_ids if self.world.alerts_by_id[cid]["node"] in {target_node, "api_gateway", "internal_tools"}
        ]

        evidence_accuracy = len(relevant_real_ids) / citation_count

        coverage_hits = 0
        low_reasoning = reasoning.lower()
        if any(token in low_reasoning for token in ["downtime", "revenue", "impact", "cost"]):
            coverage_hits += 1
        if any(token in low_reasoning for token in ["isolate", "contain", "scope", "blast radius"]):
            coverage_hits += 1
        if any(token in low_reasoning for token in ["trust", "false", "certainty", "evidence"]):
            coverage_hits += 1
        objection_coverage = coverage_hits / 3.0

        contradiction = "no risk" in low_reasoning and "high severity" in low_reasoning
        consistency = 0.0 if contradiction else 1.0

        brevity = 1.0
        if len(reasoning.split()) > 180:
            brevity = 0.8

        argument_score = (
            0.5 * evidence_accuracy
            + 0.3 * objection_coverage
            + 0.15 * consistency
            + 0.05 * brevity
        )
        argument_score = max(0.0, min(1.0, argument_score))

        # Evidence is mandatory for approval-grade scores.
        if evidence_accuracy < 1.0:
            argument_score = min(argument_score, 0.59)

        fake_citations = [cid for cid in valid_ids if self.world.truth_alert_is_fake.get(cid, True)]
        trust_penalty = 0.0
        if fake_citations:
            trust_penalty = min(0.4, 0.1 * len(fake_citations))

        if fake_citations or argument_score < 0.45:
            stance = "block"
        elif argument_score < 0.72:
            stance = "delay"
        else:
            stance = "approve"

        return {
            "argument_score": round(argument_score, 4),
            "stance": stance,
            "trust_penalty": round(trust_penalty, 4),
            "breakdown": {
                "evidence_accuracy": round(evidence_accuracy, 4),
                "objection_coverage": round(objection_coverage, 4),
                "consistency": round(consistency, 4),
                "brevity": round(brevity, 4),
            },
        }
