from types import SimpleNamespace
import unittest

import numpy as np

from decision_split_policy import DecisionSplitPolicy, locally_dominates, QUEUE_INDEX, PROGRESS_INDEX
from mappo_evaluation import QRoutingPolicy
from run_development_load_diagnostics import run_episode


class SplitTests(unittest.TestCase):
    def test_score_guard_rejects_an_unsupported_override(self):
        constrained = lambda obs, mask: np.array([1])
        constrained.checkpoint_schema = {}
        constrained.action_logits = lambda obs, mask: np.array([[-1e9, 2., 0.]])
        policy = DecisionSplitPolicy(constrained, lambda obs, mask: np.array([2]),
                                     pareto_guard=True, max_logit_gap=np.log(2))
        features = np.zeros((2,26))
        features[0,QUEUE_INDEX] = 1
        wrapper = SimpleNamespace(external_to_internal=[1], env=SimpleNamespace(
            packets={1:SimpleNamespace(dst=99,traffic_class=0)}, route_cache={}),
            _obs=[dict(hol_packet_id=1,neighbor_ids=[10,11],candidate_features=features)])
        policy.bind(wrapper)
        mask = np.array([[False,True,True]])
        np.testing.assert_array_equal(policy(None,mask), [1])
        self.assertEqual(policy.counts["first_route_logit_rejected"],1)
        constrained.action_logits = lambda obs, mask: np.array([[-1e9,.5,0.]])
        np.testing.assert_array_equal(policy(None,mask), [2])

    def test_pareto_guard_rejects_tradeoffs_ties_and_noop(self):
        features = np.zeros((2, 26))
        features[0, [QUEUE_INDEX, PROGRESS_INDEX]] = [.2, .5]
        for queue, progress, expected in ((.1,.5,True), (.2,.6,True), (.1,.6,True),
                                          (.3,.6,False), (.1,.4,False), (.2,.5,False)):
            features[1, [QUEUE_INDEX, PROGRESS_INDEX]] = [queue, progress]
            self.assertEqual(locally_dominates(features, 2, 1), expected)
        self.assertFalse(locally_dominates(features, 0, 1))
        self.assertFalse(locally_dominates(features, 1, 1))

    def test_first_forced_opportunity_single_exit_and_empty(self):
        constrained = lambda obs, mask: np.array([1, 1, 1, 1, 0])
        routing = lambda obs, mask: np.array([2, 2, 2, 1, 0])
        constrained.checkpoint_schema = {}
        policy = DecisionSplitPolicy(constrained, routing)
        packet = SimpleNamespace(dst=99, traffic_class=0)
        wrapper = SimpleNamespace(external_to_internal=[1, 2, 3, 4, 5],
                                  env=SimpleNamespace(packets={1: packet}, route_cache={(2,99,0): 12, (3,99,0):10, (4,99,0):10}),
                                  _obs=[dict(hol_packet_id=1 if i < 4 else None, neighbor_ids=[10,11]) for i in range(5)])
        policy.bind(wrapper)
        mask = np.array([[False, True, True]]*3 + [[False, True, False], [True, False, False]])
        np.testing.assert_array_equal(policy(None, mask), [2,2,1,1,0])
        self.assertEqual(policy.counts["opportunity"], 1)
        self.assertEqual(policy.counts["first_route_changed"], 1)
        self.assertEqual(policy.counts["forced_recovery_changed"], 1)
        self.assertEqual(policy.counts["single_exit_changed"], 0)
        for obs in wrapper._obs:
            obs["candidate_features"] = np.zeros((2, 26))
            obs["candidate_features"][1, QUEUE_INDEX] = 1
        policy.pareto_guard = True
        np.testing.assert_array_equal(policy(None, mask), [1,1,1,1,0])
        wrapper._obs[0]["candidate_features"][0, QUEUE_INDEX] = 2
        np.testing.assert_array_equal(policy(None, mask), [2,1,1,1,0])

    def test_direct_delivery_shared_and_counted_as_opportunity(self):
        constrained = lambda obs, mask: np.array([1])
        constrained.checkpoint_schema = {}
        policy = DecisionSplitPolicy(constrained, lambda obs, mask: np.array([1]))
        wrapper = SimpleNamespace(external_to_internal=[1], env=SimpleNamespace(
            packets={1: SimpleNamespace(dst=11, traffic_class=0)}, route_cache={(1,11,0):10}),
            _obs=[dict(hol_packet_id=1, neighbor_ids=[10,11])])
        policy.bind(wrapper)
        np.testing.assert_array_equal(policy(None, np.array([[False,True,True]])), [2])
        self.assertEqual(policy.counts["opportunity"], 1)

    def test_episode_replay_and_frozen_q(self):
        constrained, routing = QRoutingPolicy(seed=41), QRoutingPolicy(seed=42)
        constrained.freeze()
        routing.freeze()
        constrained.checkpoint_schema = {}
        policy = DecisionSplitPolicy(constrained, routing)
        q_before = routing.q.copy()
        first = run_episode("hotspot_high_load", 16, "split", policy, 41, 929093)
        self.assertEqual(policy.counts["opportunity"], first[0]["decision_switch_opportunities"])
        second = run_episode("hotspot_high_load", 16, "split", policy, 41, 929093)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(q_before, routing.q)


if __name__ == "__main__":
    unittest.main()
