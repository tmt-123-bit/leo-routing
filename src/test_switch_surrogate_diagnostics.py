"""Focused contracts for frozen-policy surrogate diagnostics."""

import json
import unittest

import torch

from mappo_design import SharedCandidateActor
from switch_surrogate_diagnostics import gradient_probe, summarize_decisions


class SwitchSurrogateDiagnosticTests(unittest.TestCase):
    def test_switch_mass_can_exceed_half_while_greedy_stays(self):
        candidates = torch.zeros(1, 4, 18)
        candidates[0, 2:, 17] = 1.0
        mask = torch.tensor([[False, True, True, True]])
        summary = summarize_decisions(torch.tensor([[-1e9, 0.1, 0.0, 0.0]]), candidates, mask)
        self.assertEqual(summary["greedy_switches"], 0)
        self.assertGreater(summary["conditional_probability_rate"], 0.5)
        self.assertEqual(summary["mass_above_half_with_greedy_stay_count"], 1)
        self.assertEqual(summary["by_switch_candidate_count"]["2"]["eligible_decisions"], 1)
        json.dumps(summary, allow_nan=False)

    def test_first_use_forced_and_inactive_are_excluded(self):
        candidates = torch.zeros(4, 3, 18)
        candidates[0, 2, 17] = 1.0
        candidates[1, 1:, 17] = 1.0
        mask = torch.tensor([[False, True, True]] * 3 + [[True, False, False]])
        logits = torch.tensor([[-1e9, 0.0, 1.0]] * 3 + [[0.0, -1e9, -1e9]])
        summary = summarize_decisions(logits, candidates, mask)
        self.assertEqual(summary["decisions"], 4)
        self.assertEqual(summary["active_decisions"], 3)
        self.assertEqual(summary["eligible_decisions"], 1)
        self.assertEqual(summary["non_opportunity_active_decisions"], 2)
        self.assertEqual(summary["greedy_switches"], 1)

    def test_tie_uses_the_actor_argmax_order(self):
        candidates = torch.zeros(1, 3, 18)
        candidates[0, 1, 17] = 1.0
        summary = summarize_decisions(
            torch.zeros(1, 3), candidates, torch.tensor([[False, True, True]])
        )
        self.assertEqual(summary["greedy_switches"], 1)
        self.assertEqual(summary["positive_switch_margin_count"], 0)
        self.assertEqual(summary["zero_switch_margin_count"], 1)

    def test_probe_preserves_original_and_reduces_local_probability(self):
        torch.manual_seed(718)
        actor = SharedCandidateActor(18, 8, route_switch_feature_index=17)
        actor.train()
        candidates = torch.randn(6, 3, 18)
        candidates[:, :, 17] = torch.tensor([0.0, 0.0, 1.0])
        mask = torch.tensor([[False, True, True]] * 6)
        before = {name: value.clone() for name, value in actor.state_dict().items()}
        result = gradient_probe(actor, candidates, mask)
        self.assertEqual(result["status"], "probed")
        self.assertLess(result["conditional_probability_change"], 0.0)
        self.assertGreater(result["gradient_norm"], 0.0)
        self.assertTrue(actor.training)
        for name, value in actor.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertTrue(all(p.grad is None for p in actor.parameters()))
        json.dumps(result, allow_nan=False)

    def test_no_opportunities_are_undefined_not_zero_rate(self):
        actor = SharedCandidateActor(18, 8, route_switch_feature_index=17)
        candidates = torch.zeros(2, 3, 18)
        mask = torch.tensor([[False, True, True], [True, False, False]])
        result = gradient_probe(actor, candidates, mask)
        self.assertEqual(result["status"], "no_opportunities")
        self.assertIsNone(result["gradient_norm"])
        self.assertIsNone(result["before"]["conditional_probability_rate"])
        self.assertIsNone(result["before"]["greedy_switch_rate"])
        self.assertIsNone(result["after"])
        json.dumps(result, allow_nan=False)

    def test_zero_multiplier_produces_no_parameter_step(self):
        actor = SharedCandidateActor(18, 8, route_switch_feature_index=17)
        candidates = torch.zeros(1, 3, 18)
        candidates[0, 2, 17] = 1.0
        mask = torch.tensor([[False, True, True]])
        result = gradient_probe(actor, candidates, mask, dual_multiplier=0.0)
        self.assertEqual(result["status"], "zero_step")
        self.assertEqual(result["parameter_step_norm"], 0.0)
        self.assertEqual(result["conditional_probability_change"], 0.0)
        self.assertEqual(result["weighted_gradient_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
