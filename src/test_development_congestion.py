import unittest

from analyze_development_congestion import decision_flags, hotspot_id
from run_development_load_diagnostics import make_wrapper
from dataclasses import asdict


class CongestionAnalysisTests(unittest.TestCase):
    def test_direct_delivery_respects_original_mask(self):
        from run_direct_delivery_probe import direct_action
        self.assertEqual(direct_action(3, [2, 3, 0], [0, 1, 1, 0], 1), 2)
        self.assertEqual(direct_action(3, [2, 3, 0], [0, 1, 0, 0], 1), 1)
        self.assertEqual(direct_action(4, [2, 3, 0], [0, 1, 1, 0], 1), 1)
        self.assertEqual(direct_action(3, [2, 3, 0], [1, 0, 0, 0], 0), 0)

    def test_hotspot_id(self):
        self.assertEqual(hotspot_id(asdict(make_wrapper("medium_load", 6, 1).env.cfg)), 9)

    def test_rising_and_lower_alternative(self):
        row = {"neighbors": [2, 3, 0], "mask": [False, True, True, False],
               "old_next_hop": 2, "neighbor_queues": [5, 2, None],
               "neighbor_previous_queues": [3, 2, None], "action": 1, "destination": 3}
        flags = decision_flags(row)
        self.assertTrue(flags["holding_rising_with_lower_alternative"])
        self.assertTrue(flags["direct_bypassed"])
        row["mask"][2] = False
        flags = decision_flags(row)
        self.assertFalse(flags["holding_rising_with_lower_alternative"])
        self.assertFalse(flags["direct_bypassed"])
        row["mask"][1] = False
        self.assertFalse(decision_flags(row)["old_feasible"])

    def test_no_cached_route_or_neighbor(self):
        row = {"neighbors": [2, 3, 0], "mask": [False, True, True, False],
               "old_next_hop": None, "neighbor_queues": [5, 2, None],
               "neighbor_previous_queues": [3, 2, None], "action": 2, "destination": 4}
        self.assertFalse(any(decision_flags(row).values()))


if __name__ == "__main__":
    unittest.main()
