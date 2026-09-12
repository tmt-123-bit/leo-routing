"""Development-only separation of routing and optional-switch decisions."""

from collections import Counter

from leo_multiagent_env import BASE_CANDIDATE_FEATURE_NAMES
from run_direct_delivery_controls import DirectDeliveryControlPolicy


QUEUE_INDEX = BASE_CANDIDATE_FEATURE_NAMES.index("candidate_queue_ratio")
PROGRESS_INDEX = BASE_CANDIDATE_FEATURE_NAMES.index("destination_progress")


def locally_dominates(features, proposed, original):
    if proposed <= 0 or original <= 0 or proposed == original:
        return False
    new, old = features[proposed - 1], features[original - 1]
    return (new[QUEUE_INDEX] <= old[QUEUE_INDEX] and new[PROGRESS_INDEX] >= old[PROGRESS_INDEX]
            and (new[QUEUE_INDEX] < old[QUEUE_INDEX] or new[PROGRESS_INDEX] > old[PROGRESS_INDEX]))


class DecisionSplitPolicy(DirectDeliveryControlPolicy):
    """Use the constrained actor only when a cached feasible route has alternatives.

    Both actors see the unchanged local observation. This frozen two-policy
    diagnostic is not a newly trained MAPPO model and has no hard budget guarantee.
    """

    def __init__(self, constrained, routing, pareto_guard=False, max_logit_gap=None):
        super().__init__(constrained)
        self.routing = DirectDeliveryControlPolicy(routing)
        self.pareto_guard = pareto_guard
        self.max_logit_gap = max_logit_gap
        if max_logit_gap is not None and (max_logit_gap < 0 or not hasattr(constrained, "action_logits")):
            raise ValueError("logit guard requires a nonnegative gap and an actor score accessor")
        other_schema = getattr(routing, "checkpoint_schema", None)
        if other_schema is not None:
            for field in ("candidate_feature_dim", "action_size", "obs_size", "n_agents", "variant"):
                if other_schema[field] != self.checkpoint_schema[field]:
                    raise ValueError("incompatible routing actor schema: " + field)

    def bind(self, wrapper):
        super().bind(wrapper)
        self.routing.bind(wrapper)
        self.counts = Counter()

    def __call__(self, observation, mask):
        actions = super().__call__(observation, mask)
        routing_actions = self.routing(observation, mask)
        logits = self.policy.action_logits(observation, mask) if self.max_logit_gap is not None else None
        for external, sat in enumerate(self.wrapper.external_to_internal):
            obs = self.wrapper._obs[sat - 1]
            if obs["hol_packet_id"] is None:
                continue
            packet = self.wrapper.env.packets[obs["hol_packet_id"]]
            old = self.wrapper.env.route_cache.get((sat, packet.dst, packet.traffic_class))
            feasible = {obs["neighbor_ids"][a - 1] for a in range(1, len(mask[external]))
                        if mask[external, a]}
            opportunity = old in feasible and len(feasible) > 1
            kind = "opportunity" if opportunity else "first_route" if old is None else "forced_recovery" if old not in feasible else "single_exit"
            self.counts[kind] += 1
            if not opportunity:
                changed = actions[external] != routing_actions[external]
                if self.pareto_guard and changed and not locally_dominates(
                        obs["candidate_features"], routing_actions[external], actions[external]):
                    self.counts[kind + "_guard_rejected"] += 1
                    continue
                if changed and logits is not None and (
                        logits[external, actions[external]] - logits[external, routing_actions[external]] > self.max_logit_gap):
                    self.counts[kind + "_logit_rejected"] += 1
                    continue
                self.counts[kind + "_changed"] += int(changed)
                actions[external] = routing_actions[external]
        return actions
