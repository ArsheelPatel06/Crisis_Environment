import unittest

from server.tasks import (
    grade_task1_alert_triage,
    grade_task2_stakeholder_argument,
    grade_task3_full_episode,
)


class TestTaskGraders(unittest.TestCase):
    def test_task1_perfect(self):
        truth = {"a1": False, "a2": True, "a3": False}
        preds = {"a1": "real", "a2": "fake", "a3": "real"}
        self.assertAlmostEqual(grade_task1_alert_triage(preds, truth), 1.0)

    def test_task1_half_correct(self):
        truth = {"a1": False, "a2": True, "a3": False, "a4": True}
        preds = {"a1": "fake", "a2": "fake", "a3": "real", "a4": "real"}
        self.assertAlmostEqual(grade_task1_alert_triage(preds, truth), 0.5)

    def test_task2_good_argument_approves(self):
        truth = {"a1": False, "a2": False, "a3": True}
        node_map = {"a1": "auth_server", "a2": "auth_server", "a3": "internal_tools"}
        objections = [
            "downtime risk for revenue",
            "insufficient proof that auth_server is compromised",
            "customer trust impact if this is false alarm",
        ]
        result = grade_task2_stakeholder_argument(
            target_node="auth_server",
            objections=objections,
            argument_text=(
                "To reduce downtime risk we isolate auth_server briefly and restore quickly. "
                "Proof is from auth_server indicators. This avoids larger customer trust impact."
            ),
            citations=["a1", "a2"],
            truth_alert_is_fake=truth,
            alert_node_map=node_map,
        )
        self.assertGreaterEqual(result["argument_score"], 0.6)
        self.assertEqual(result["stance"], "approve")

    def test_task2_theme_coverage_without_exact_phrase(self):
        truth = {"a1": False, "a2": False}
        node_map = {"a1": "auth_server", "a2": "auth_server"}
        objections = [
            "downtime risk for revenue",
            "insufficient proof that auth_server is compromised",
            "customer trust impact if this is false alarm",
        ]
        result = grade_task2_stakeholder_argument(
            target_node="auth_server",
            objections=objections,
            argument_text=(
                "This containment limits outage and protects availability. "
                "Our telemetry indicators from auth_server support compromise likelihood. "
                "Acting now protects customer confidence and avoids a reputational hit."
            ),
            citations=["a1", "a2"],
            truth_alert_is_fake=truth,
            alert_node_map=node_map,
        )
        self.assertGreaterEqual(result["subscores"]["objection_coverage"], 0.66)

    def test_task2_fake_evidence_blocks(self):
        truth = {"a1": True, "a2": True}
        node_map = {"a1": "auth_server", "a2": "auth_server"}
        objections = ["downtime risk", "lack of proof", "trust impact"]
        result = grade_task2_stakeholder_argument(
            target_node="auth_server",
            objections=objections,
            argument_text="Please approve this immediately.",
            citations=["a1", "a2"],
            truth_alert_is_fake=truth,
            alert_node_map=node_map,
        )
        self.assertLess(result["argument_score"], 0.6)
        self.assertIn(result["stance"], {"delay", "block"})

    def test_task3_normalized(self):
        score = grade_task3_full_episode(
            episode_reward_totals=[0.4, 0.5, 0.6],
            baseline_score=0.3,
            analytic_best_estimate=0.8,
        )
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

