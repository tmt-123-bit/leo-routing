import unittest
from unittest.mock import patch

from run_local_switch_budget_probe import assess


class ProbeGateTests(unittest.TestCase):
    def test_exact_packet_tie_is_not_a_negative_seed(self):
        rows = []
        for scenario in ("medium_load", "hotspot_high_load"):
            for method in ("budget_qos", "budget_q", "original_constrained", "direct_constrained", "direct_persistent"):
                for workload, difference in zip([1,2,3], [-2,1,1]):
                    rows.append(dict(scenario=scenario, policy=method, policy_seed=7, workload_seed=workload,
                                     generated=192, delivered=100+(difference if method.startswith("budget") else 0),
                                     decision_switch_opportunities=100, decision_avoidable_switches=1,
                                     average_delay_slots=5, mean_delivered_hops=3))
        with patch("run_local_switch_budget_probe.crossed_interval", return_value=[-1,1]):
            results = assess(rows, [7], [1,2,3])
        for result in results:
            self.assertTrue(result["checks"]["every_seed_delivery_nondecreasing"])
            self.assertFalse(result["checks"]["mean_delivery_positive"])
            self.assertEqual(result["per_seed_delivered_packet_difference"], [0])


if __name__ == "__main__":
    unittest.main()
