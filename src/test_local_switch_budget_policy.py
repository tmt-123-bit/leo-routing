import unittest

import numpy as np

from local_switch_budget_policy import LocalSwitchBudgetPolicy
from mappo_evaluation import QRoutingPolicy
from run_development_load_diagnostics import make_wrapper, run_episode


class LocalBudgetTests(unittest.TestCase):
    def test_prefix_counts_match_environment_and_repeat_call_is_idempotent(self):
        base = QRoutingPolicy(seed=31)
        base.freeze()
        before = base.q.copy()
        policy = LocalSwitchBudgetPolicy(base)
        for scenario, load in (("medium_load", 6), ("hotspot_high_load", 16)):
            wrapper = make_wrapper(scenario, load, 929091)
            policy.bind(wrapper)
            obs, _ = wrapper.reset(seed=929091)
            opportunities = np.zeros(24, dtype=np.int64)
            switches = opportunities.copy()
            for _ in range(30):
                mask = wrapper.get_avail_actions()
                actions = policy(obs, mask)
                np.testing.assert_array_equal(actions, policy(obs, mask))
                for i, action in enumerate(actions):
                    self.assertTrue(mask[i, action])
                obs, _, terminated, truncated, info = wrapper.step(actions)
                opportunities += np.asarray(info["decision_switch_opportunities"], dtype=np.int64)
                switches += np.asarray(info["decision_avoidable_switch_costs"], dtype=np.int64)
                np.testing.assert_array_equal(policy.opportunities, opportunities)
                np.testing.assert_array_equal(policy.switches, switches)
                self.assertTrue(np.all(25 * switches <= 3 * opportunities))
                if terminated or truncated:
                    break
            self.assertGreater(int(opportunities.sum()), 0)
            self.assertGreater(policy.denied, 0)
        np.testing.assert_array_equal(before, base.q)

    def test_episode_reset_is_deterministic(self):
        base = QRoutingPolicy(seed=31)
        base.freeze()
        policy = LocalSwitchBudgetPolicy(base)
        first = run_episode("medium_load", 6, "budget", policy, 31, 929092)
        second = run_episode("medium_load", 6, "budget", policy, 31, 929092)
        self.assertEqual(first, second)

    def test_training_policy_rejected(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            LocalSwitchBudgetPolicy(QRoutingPolicy())


if __name__ == "__main__":
    unittest.main()
