"""Development-only, per-satellite prefix budget on decision-level switches."""

import numpy as np

from run_direct_delivery_controls import DirectDeliveryControlPolicy


class LocalSwitchBudgetPolicy(DirectDeliveryControlPolicy):
    """Apply direct delivery, then spend at most 3 switches per 25 opportunities.

    No initial credit, borrowing, global counters, or exemptions for useful
    switches. This is stronger than the original aggregate evaluation budget.
    Counters are local to each satellite and reset at the episode boundary.
    """

    def bind(self, wrapper):
        super().bind(wrapper)
        self.opportunities = np.zeros(len(wrapper.external_to_internal), dtype=np.int64)
        self.switches = np.zeros_like(self.opportunities)
        self.denied = 0
        self.last_slot = None
        self.last_actions = None

    def __call__(self, observation, mask):
        slot = self.wrapper.env.slot
        if self.last_slot == slot:
            return self.last_actions.copy()
        actions = super().__call__(observation, mask)
        for external, sat in enumerate(self.wrapper.external_to_internal):
            obs = self.wrapper._obs[sat - 1]
            action = int(actions[external])
            if obs["hol_packet_id"] is None or action <= 0 or not mask[external, action]:
                continue
            packet = self.wrapper.env.packets[obs["hol_packet_id"]]
            old = self.wrapper.env.route_cache.get((sat, packet.dst, packet.traffic_class))
            feasible = {obs["neighbor_ids"][a - 1]: a for a in range(1, len(mask[external]))
                        if mask[external, a]}
            if old not in feasible or len(feasible) < 2:
                continue
            self.opportunities[external] += 1
            if action == feasible[old]:
                continue
            if 25 * (self.switches[external] + 1) <= 3 * self.opportunities[external]:
                self.switches[external] += 1
            else:
                actions[external] = feasible[old]
                self.denied += 1
        self.last_slot = slot
        self.last_actions = actions.copy()
        return actions
