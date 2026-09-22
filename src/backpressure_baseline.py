"""Backpressure (max-weight) routing baseline with shortest-path bias.

Classic backpressure forwards to the neighbor maximizing the differential
backlog Q_i - Q_j, which is throughput-optimal but routes terribly without a
bias. The practical variant (EDR-BP style) adds gamma * (D_i - D_j) where D
is the hop distance to the packet destination; gamma trades backlog pressure
against path quality. gamma is tuned on a development sweep so the baseline
gets its best shot; distances are BFS over currently available links.
"""

from collections import deque

import numpy as np


class BackpressurePolicy:
    def __init__(self, gamma: float = 2.0):
        self.gamma = gamma
        self.wrapper = None
        self._cache = {}

    def bind(self, wrapper):
        self.wrapper = wrapper
        self._cache = {}

    def _distances(self, dst):
        env = self.wrapper.env
        key = (env.slot, dst)
        if key not in self._cache:
            reverse = {}
            for (u, v), edge in env.graph.items():
                if u >= 1 and v >= 1 and edge.available:
                    reverse.setdefault(v, []).append(u)
            dist = {dst: 0}
            queue = deque([dst])
            while queue:
                node = queue.popleft()
                for prev in reverse.get(node, ()):
                    if prev not in dist:
                        dist[prev] = dist[node] + 1
                        queue.append(prev)
            self._cache[key] = dist
        return self._cache[key]

    def __call__(self, observation, mask):
        env = self.wrapper.env
        actions = np.zeros(self.wrapper.n_agents, dtype=np.int64)
        for ext, sat in enumerate(self.wrapper.external_to_internal):
            obs = self.wrapper._obs[sat - 1]
            packet_id = obs["hol_packet_id"]
            if packet_id is None:
                continue
            packet = env.packets[packet_id]
            distances = self._distances(packet.dst)
            q_i = len(env.queues[sat])
            d_i = distances.get(sat, 10 ** 6)
            best_action, best_weight = 0, None
            for k, neighbor in enumerate(obs["neighbor_ids"]):
                if neighbor is None or not obs["action_mask"][k + 1]:
                    continue
                weight = (q_i - len(env.queues[neighbor])) + self.gamma * (
                    d_i - distances.get(neighbor, 10 ** 6)
                )
                if best_weight is None or weight > best_weight:
                    best_weight, best_action = weight, k + 1
            actions[ext] = best_action
        return actions
