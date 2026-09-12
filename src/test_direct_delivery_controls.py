import unittest
from unittest.mock import patch
import numpy as np

from mappo_evaluation import QRoutingPolicy
from run_development_load_diagnostics import PersistentDijkstraPolicy, make_wrapper, run_episode
from run_direct_delivery_probe import DirectDeliveryPolicy
from run_direct_delivery_controls import (
    BASES, DirectDeliveryControlPolicy, analyze, exact_sign_flip, matrix, row_index,
)


class DirectDeliveryControlTests(unittest.TestCase):
    def test_rule_is_inherited_unchanged(self):
        self.assertIs(DirectDeliveryControlPolicy.__call__, DirectDeliveryPolicy.__call__)

    def test_frozen_baseline_binding_and_reset(self):
        baseline = QRoutingPolicy(seed=13)
        baseline.freeze()
        before = baseline.q.copy()
        policy = DirectDeliveryControlPolicy(baseline)
        first, packets, _ = run_episode("medium_load", 2, "direct_q", policy, 13, 910096)
        self.assertIs(policy.wrapper, baseline.wrapper)
        overrides = int(policy.overrides)
        repeated, repeated_packets, _ = run_episode("medium_load", 2, "direct_q", policy, 13, 910096)
        self.assertEqual(first, repeated)
        self.assertEqual(packets, repeated_packets)
        self.assertEqual(overrides, int(policy.overrides))
        np.testing.assert_array_equal(before, baseline.q)

    def test_reject_training_q(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            DirectDeliveryControlPolicy(QRoutingPolicy())

    def test_persistent_direct_action_obeys_mask(self):
        policy = DirectDeliveryControlPolicy(PersistentDijkstraPolicy())
        wrapper = make_wrapper("hotspot_high_load", 16, 910097)
        policy.bind(wrapper)
        obs, _ = wrapper.reset(seed=910097)
        for _ in range(30):
            mask = wrapper.get_avail_actions()
            actions = policy(obs, mask)
            for i, action in enumerate(actions):
                self.assertTrue(mask[i, action])
                local = wrapper._obs[i]
                if local["hol_packet_id"] is None:
                    continue
                dst = wrapper.env.packets[local["hol_packet_id"]].dst
                if dst in local["neighbor_ids"]:
                    direct = local["neighbor_ids"].index(dst) + 1
                    if mask[i, direct]:
                        self.assertEqual(action, direct)
            obs, _, terminated, truncated, _ = wrapper.step(actions)
            if terminated or truncated:
                break

    def test_pairing_rejects_duplicates_and_missing_rows(self):
        row = dict(scenario="s", policy="p", policy_seed=1, workload_seed=2, delivered=3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            row_index([row, row])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            matrix(row_index([row]), "s", "p", [1], [2, 3], "delivered")

    def test_deterministic_reference_and_sign_flip(self):
        row = dict(scenario="s", policy="p", policy_seed=-1, workload_seed=2, delivered=3)
        np.testing.assert_array_equal(matrix(row_index([row]), "s", "p", [7, 8], [2], "delivered"), [[3], [3]])
        self.assertAlmostEqual(exact_sign_flip(np.ones(8)), 2 / 256)
        self.assertEqual(exact_sign_flip(np.zeros(8)), 1)
        self.assertIsNone(exact_sign_flip(np.ones(1)))

    def test_positive_mean_cannot_hide_negative_seed(self):
        rows = []
        for scenario in ("medium_load", "hotspot_high_load"):
            for base in BASES:
                for seed in [-1] if base == BASES[-1] else [7, 8]:
                    for direct in (False, True):
                        for workload in (1, 2):
                            gain = (-2 if seed == 7 else 4) if base == BASES[0] else 1
                            rows.append(dict(scenario=scenario, policy=("direct_" if direct else "") + base,
                                             policy_seed=seed, workload_seed=workload,
                                             generated=200, delivered=100 + (gain if direct else 0),
                                             decision_avoidable_switches=5, decision_switch_opportunities=100,
                                             average_delay_slots=5, mean_delivered_hops=3))
        with patch("run_direct_delivery_controls.WORKLOADS", (1, 2)), patch("run_direct_delivery_controls.BOOTSTRAPS", 100):
            effects, gates = analyze(rows, [7, 8])
        self.assertTrue(all(g["checks"]["mean_delivery_positive"] for g in gates))
        self.assertTrue(all(not g["checks"]["every_seed_delivery_nondecreasing"] for g in gates))
        self.assertTrue(all(not g["development_screen_pass"] for g in gates))
        effect = next(e for e in effects if e["contrast"] == "direct_minus_original_qos_only_constrained")
        self.assertEqual(effect["negative_seeds"], 1)
        self.assertAlmostEqual(effect["mean_delivery_effect_pp"], 0.5)


if __name__ == "__main__":
    unittest.main()
