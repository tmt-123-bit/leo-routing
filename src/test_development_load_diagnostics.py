import unittest
from dataclasses import asdict

from mappo_evaluation import GlobalDijkstraPolicy, evaluate_policy_with_constraint_metrics
from run_development_load_diagnostics import (
    PersistentDijkstraPolicy, aggregate, choose_persistent_action, make_wrapper,
    packet_accounting, run_episode,
)


class DevelopmentDiagnosticsTests(unittest.TestCase):
    def test_optimistic_capacity_bound(self):
        from audit_development_service_capacity import optimistic_delivery_bound
        self.assertEqual(optimistic_delivery_bound(500, 16, 400, 12, 100, 720), 196)
        self.assertEqual(optimistic_delivery_bound(20, 2, 10, 1, 100, 720), 18)
        self.assertEqual(optimistic_delivery_bound(500, 16, 400, 12, 100, 100), 100)

    def test_persistent_action_contract(self):
        neighbors = [2, 3, 0]
        self.assertEqual(choose_persistent_action(None, neighbors, [0, 1, 1, 0], 2), 2)
        self.assertEqual(choose_persistent_action(2, neighbors, [0, 1, 1, 0], 2), 1)
        self.assertEqual(choose_persistent_action(2, neighbors, [0, 0, 1, 0], 2), 2)
        self.assertEqual(choose_persistent_action(4, neighbors, [0, 1, 1, 0], 2), 2)
        self.assertEqual(choose_persistent_action(2, neighbors, [1, 0, 0, 0], 0), 0)

    def test_instrumentation_parity_and_determinism(self):
        row, packets, _ = run_episode("medium_load", 2, "global_dijkstra", GlobalDijkstraPolicy(), -1, 910091)
        original = evaluate_policy_with_constraint_metrics(
            "medium_load", "global_dijkstra", GlobalDijkstraPolicy(), -1, [910091],
            wrapper_factory=lambda seed: make_wrapper("medium_load", 2, seed, diagnostic=False),
        )[0]
        for field, value in asdict(original).items():
            if field == "decision_avoidable_switch_rate" and original.decision_switch_opportunities == 0:
                self.assertIsNone(row[field])
            else:
                self.assertEqual(row[field], value, field)
        repeated, repeated_packets, _ = run_episode("medium_load", 2, "global_dijkstra", GlobalDijkstraPolicy(), -1, 910091)
        self.assertEqual(row, repeated)
        self.assertEqual(packets, repeated_packets)
        self.assertEqual(len(packets), row["generated"])
        self.assertEqual(row["generated"], row["delivered"] + row["dropped"] + row["backlog"])
        self.assertEqual(row["backlog_final_slot_arrivals"], 2)

    def test_keep_policy_and_event_pairing(self):
        base, _, _ = run_episode("medium_load", 6, "global_dijkstra", GlobalDijkstraPolicy(), -1, 910092)
        kept, _, decisions = run_episode("medium_load", 6, "persistent_global_dijkstra", PersistentDijkstraPolicy(), -1, 910092)
        self.assertEqual(base["arrival_sha256"], kept["arrival_sha256"])
        self.assertEqual(base["physical_sha256"], kept["physical_sha256"])
        self.assertEqual(kept["decision_avoidable_switches"], 0)
        for row in decisions:
            old = row["old_next_hop"]
            if old in row["neighbors"]:
                action = row["neighbors"].index(old) + 1
                if row["mask"][action]:
                    self.assertEqual(row["action"], action)

    def test_accounting_rejects_missing_reason(self):
        wrapper = make_wrapper("medium_load", 1, 910093)
        wrapper.reset(seed=910093)
        env = wrapper.env
        packet_id = env.queues[1][0] if env.queues[1] else next(iter(env.generated))
        owner = env.packets[packet_id].owner
        env.queues[owner].remove(packet_id)
        env.dropped.add(packet_id)
        with self.assertRaisesRegex(AssertionError, "drop reason coverage"):
            packet_accounting(env)
        env.drop_reasons[packet_id] = "queue_overflow"
        extra, packets = packet_accounting(env)
        self.assertEqual(extra["drop_queue_overflow"], 1)
        self.assertEqual(sum(p["status"] == "dropped" for p in packets), 1)

    def test_aggregation_equal_seed_weights_and_undefined(self):
        row, _, _ = run_episode("medium_load", 2, "global_dijkstra", GlobalDijkstraPolicy(), -1, 910094)
        a = {**row, "policy_seed": 1, "decision_avoidable_switches": 1, "decision_switch_opportunities": 2}
        b = {**row, "policy_seed": 2, "decision_avoidable_switches": 1, "decision_switch_opportunities": 10}
        self.assertAlmostEqual(aggregate([a, b])[0]["decision_switch_rate"], 0.3)
        b.update(decision_avoidable_switches=0, decision_switch_opportunities=0)
        self.assertIsNone(aggregate([a, b])[0]["decision_switch_rate"])
        self.assertIsNone(aggregate([a, b])[0]["all_observed_seed_rates_within_budget"])


if __name__ == "__main__":
    unittest.main()
