"""Held-out evaluation helpers for the satellite-level MAPPO experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
from pathlib import Path
import random
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from mappo_design import SharedCandidateActor


Policy = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class EpisodeMetrics:
    scenario: str
    policy: str
    policy_seed: int
    workload_seed: int
    generated: int
    delivered: int
    dropped: int
    backlog: int
    delivery_ratio: float
    drop_rate: float
    throughput_packets_per_slot: float
    average_delay_slots: float
    p95_delay_slots: float
    mean_queue_packets: float
    max_queue_packets: int
    routing_switches: int
    episode_reward: float
    global_delay_cost: float
    global_queue_cost: float
    global_load_imbalance: float
    global_switch_cost: float
    global_throughput_reward: float
    global_control_overhead_ratio: float
    global_drop_cost: float
    class_0_delivery_ratio: float
    class_1_delivery_ratio: float
    class_2_delivery_ratio: float


def load_checkpoint_policy(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[Policy, Dict]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    feature_dim = int(checkpoint["candidate_feature_dim"])
    action_size = int(checkpoint["action_size"])
    actor_args = checkpoint["args"]
    actor = SharedCandidateActor(
        candidate_feature_dim=feature_dim,
        hidden_dim=int(actor_args["actor_hidden_dim"]),
        num_layers=int(actor_args["actor_num_layers"]),
    ).to(device)
    prefix = "shared_candidate_actor."
    actor_state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["actor"].items()
        if key.startswith(prefix)
    }
    if not actor_state:
        raise ValueError("checkpoint does not contain a shared candidate actor")
    actor.load_state_dict(actor_state)
    actor.eval()

    def policy(observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        candidates = torch.from_numpy(observation).float().to(device).reshape(
            observation.shape[0], action_size, feature_dim
        )
        action_mask = torch.from_numpy(mask).bool().to(device)
        with torch.no_grad():
            return actor(candidates, action_mask).argmax(dim=-1).cpu().numpy()

    return policy, checkpoint


def heuristic_policy(name: str, seed: int = 0) -> Policy:
    rng = random.Random(seed)

    def choose(observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n_agents, flat_dim = observation.shape
        action_size = mask.shape[1]
        feature_dim = flat_dim // action_size
        candidates = observation.reshape(n_agents, action_size, feature_dim)
        actions = np.zeros(n_agents, dtype=np.int64)
        for agent in range(n_agents):
            feasible = np.flatnonzero(mask[agent] > 0.5)
            if len(feasible) == 0:
                actions[agent] = 0
                continue
            if len(feasible) == 1:
                actions[agent] = int(feasible[0])
                continue
            if name == "random":
                actions[agent] = int(rng.choice(feasible.tolist()))
                continue
            rows = candidates[agent, feasible]
            if name == "delay_only":
                score = rows[:, 2]
            elif name == "full_heuristic":
                lifetime_cost = 1.0 / np.maximum(rows[:, 6], 1e-6)
                score = (
                    1.5 * rows[:, 2]
                    + rows[:, 1]
                    + rows[:, 4]
                    + 2.0 * (1.0 - rows[:, 5])
                    + lifetime_cost
                    + 0.2 * rows[:, 17]
                    - 0.5 * rows[:, 8]
                )
            else:
                raise ValueError(f"unknown heuristic policy: {name}")
            actions[agent] = int(feasible[int(np.argmin(score))])
        return actions

    return choose


class GlobalDijkstraPolicy:
    """Global link-state baseline using current propagation delay weights."""

    def __init__(self):
        self.wrapper: Optional[CleanMARLLeoMultiAgentWrapper] = None

    def bind(self, wrapper: CleanMARLLeoMultiAgentWrapper) -> None:
        self.wrapper = wrapper

    def _distance(self, source: int, destination: int) -> float:
        assert self.wrapper is not None
        graph = self.wrapper.env.graph
        distances = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if node == destination:
                return distance
            if distance != distances.get(node):
                continue
            for (edge_source, neighbor), edge in graph.items():
                if edge_source != node or not edge.available:
                    continue
                candidate = distance + edge.delay_ms
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        return float("inf")

    def __call__(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        assert self.wrapper is not None
        actions = np.zeros(self.wrapper.n_agents, dtype=np.int64)
        for external_index, internal_sat in enumerate(
            self.wrapper.external_to_internal
        ):
            feasible = np.flatnonzero(mask[external_index] > 0.5)
            if len(feasible) <= 1:
                actions[external_index] = int(feasible[0]) if len(feasible) else 0
                continue
            obs = self.wrapper._obs[internal_sat - 1]
            packet = self.wrapper.env.packets[obs["hol_packet_id"]]
            scores = []
            for action in feasible:
                if action == 0:
                    scores.append(float("inf"))
                    continue
                neighbor = obs["neighbor_ids"][int(action) - 1]
                edge = self.wrapper.env.graph[(internal_sat, neighbor)]
                scores.append(
                    edge.delay_ms + self._distance(neighbor, packet.dst)
                )
            actions[external_index] = int(feasible[int(np.argmin(scores))])
        return actions


class OspfEcmpPolicy:
    """OSPF/ECMP baseline — the realistic distributed incumbent.

    Each node forwards along the delay-weighted shortest path (SPF on its
    flooded link-state database), breaking ties across the equal-cost multi-path
    (ECMP) next-hop set by seeded random split. This is the canonical LEO/IP
    routing protocol and a required baseline: it tells the reviewer how much of
    MAPPO's gain is "learning" versus "having a sensible shortest-path policy at
    all". Uses full current link state (same information set as
    GlobalDijkstraPolicy), so it sits in the centralized-oracle column; the
    difference from Dijkstra is ECMP load splitting, which helps under
    multipath/hotspot traffic.
    """

    def __init__(self, seed: int = 0):
        self.wrapper: Optional[CleanMARLLeoMultiAgentWrapper] = None
        self.rng = random.Random(seed)
        self._cache_key: tuple = (-1, -1)
        self._cached_dist: Dict[int, float] = {}

    def bind(self, wrapper: CleanMARLLeoMultiAgentWrapper) -> None:
        self.wrapper = wrapper
        self._cache_key = (-1, -1)
        self._cached_dist = {}

    def _dist_to_destination(self, dst: int) -> Dict[int, float]:
        """Forward distance dist(node -> dst) for every node, via one reverse
        Dijkstra from dst. Cached per (slot, dst): the topology is frozen within
        a slot, so all agents in the same slot sharing a destination reuse it."""
        assert self.wrapper is not None
        slot = self.wrapper.env.slot
        key = (slot, dst)
        if key != self._cache_key:
            self._cache_key = key
            self._cached_dist = self._reverse_dijkstra(dst)
        return self._cached_dist

    def _reverse_dijkstra(self, dst: int) -> Dict[int, float]:
        graph = self.wrapper.env.graph  # type: ignore[union-attr]
        dist: Dict[int, float] = {dst: 0.0}
        queue = [(0.0, dst)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != dist.get(node):
                continue
            for (source, neighbor), edge in graph.items():
                # reversed edge: neighbor -> source, so this relaxes source
                if neighbor != node or not edge.available:
                    continue
                candidate = distance + edge.delay_ms
                if candidate < dist.get(source, float("inf")):
                    dist[source] = candidate
                    heapq.heappush(queue, (candidate, source))
        return dist

    def __call__(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        assert self.wrapper is not None
        actions = np.zeros(self.wrapper.n_agents, dtype=np.int64)
        for external_index, internal_sat in enumerate(
            self.wrapper.external_to_internal
        ):
            feasible = np.flatnonzero(mask[external_index] > 0.5)
            if len(feasible) <= 1:
                actions[external_index] = int(feasible[0]) if len(feasible) else 0
                continue
            obs = self.wrapper._obs[internal_sat - 1]
            packet = self.wrapper.env.packets[obs["hol_packet_id"]]
            dist_to_dst = self._dist_to_destination(packet.dst)
            best = float("inf")
            totals: Dict[int, float] = {}
            for action in feasible:
                if action == 0:  # NO_OP never helps a forwarder
                    totals[int(action)] = float("inf")
                    continue
                neighbor = obs["neighbor_ids"][int(action) - 1]
                edge = self.wrapper.env.graph[(internal_sat, neighbor)]
                total = edge.delay_ms + dist_to_dst.get(neighbor, float("inf"))
                totals[int(action)] = total
                if total < best:
                    best = total
            if best == float("inf"):
                actions[external_index] = 0
                continue
            # ECMP set: all equal-cost next-hops within a tight tolerance
            ecmp = [a for a, t in totals.items() if t <= best + 1e-6]
            actions[external_index] = int(self.rng.choice(ecmp))
        return actions


class QRoutingPolicy:
    """Tabular distributed Q-routing baseline indexed by node/destination/neighbor."""

    def __init__(self, n_nodes: int = 24, alpha: float = 0.3, epsilon: float = 0.1, seed: int = 0):
        self.n_nodes = n_nodes
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q = np.full((n_nodes + 1, n_nodes + 1, n_nodes + 1), 10.0, dtype=np.float32)
        self.wrapper: Optional[CleanMARLLeoMultiAgentWrapper] = None
        self.pending = []
        self.training = True

    def bind(self, wrapper: CleanMARLLeoMultiAgentWrapper) -> None:
        self.wrapper = wrapper
        self.pending = []

    def __call__(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        assert self.wrapper is not None
        actions = np.zeros(self.n_nodes, dtype=np.int64)
        self.pending = []
        for external_index, sat in enumerate(self.wrapper.external_to_internal):
            feasible = np.flatnonzero(mask[external_index] > 0.5)
            if len(feasible) <= 1:
                actions[external_index] = int(feasible[0]) if len(feasible) else 0
                continue
            obs = self.wrapper._obs[sat - 1]
            packet = self.wrapper.env.packets[obs["hol_packet_id"]]
            candidates = []
            for action in feasible:
                neighbor = obs["neighbor_ids"][int(action) - 1]
                feature = obs["candidate_features"][int(action) - 1]
                immediate = feature[2] + feature[1] + feature[4]
                candidates.append((int(action), neighbor, self.q[sat, packet.dst, neighbor] + immediate))
            if self.training and self.rng.random() < self.epsilon:
                action, neighbor, _ = self.rng.choice(candidates)
            else:
                action, neighbor, _ = min(candidates, key=lambda item: item[2])
            actions[external_index] = action
            self.pending.append((sat, packet.dst, neighbor))
        return actions

    def observe_transition(self, info: Dict) -> None:
        if not self.training or self.wrapper is None:
            return
        for sat, destination, neighbor in self.pending:
            packet_candidates = []
            for next_neighbor in self.wrapper.env.base._neighbors(neighbor):
                packet_candidates.append(self.q[neighbor, destination, next_neighbor])
            bootstrap = min(packet_candidates, default=0.0)
            edge = self.wrapper.env.graph.get((sat, neighbor))
            immediate = edge.delay_ms / self.wrapper.env.cfg.env.d_ref_ms if edge else 2.0
            immediate += len(self.wrapper.env.queues[neighbor]) / max(
                1, self.wrapper.env.cfg.max_queue_packets
            )
            target = immediate + bootstrap
            old = self.q[sat, destination, neighbor]
            self.q[sat, destination, neighbor] = (1.0 - self.alpha) * old + self.alpha * target

    def freeze(self) -> None:
        self.training = False
        self.epsilon = 0.0


def train_q_routing(
    scenario: str,
    seed: int,
    episodes: int = 200,
    workload_seed_start: int = 9001,
    workload_seed_count: int = 20,
    variant: str = "full",
) -> QRoutingPolicy:
    policy = QRoutingPolicy(seed=seed)
    for episode in range(episodes):
        workload_seed = workload_seed_start + episode % workload_seed_count
        wrapper = CleanMARLLeoMultiAgentWrapper(
            scenario=scenario, seed=workload_seed, variant=variant
        )
        observation, _ = wrapper.reset(seed=workload_seed)
        policy.bind(wrapper)
        terminated = truncated = False
        while not terminated and not truncated:
            actions = policy(observation, wrapper.get_avail_actions())
            observation, _, terminated, truncated, info = wrapper.step(actions)
            policy.observe_transition(info)
        wrapper.close()
    policy.freeze()
    return policy


def evaluate_policy(
    scenario: str,
    policy_name: str,
    policy: Policy,
    policy_seed: int,
    workload_seeds: Iterable[int],
    wrapper_factory: Optional[Callable[[int], CleanMARLLeoMultiAgentWrapper]] = None,
    variant: str = "full",
) -> list[EpisodeMetrics]:
    rows = []
    for workload_seed in workload_seeds:
        if wrapper_factory is None:
            wrapper = CleanMARLLeoMultiAgentWrapper(
                scenario=scenario, seed=int(workload_seed), variant=variant
            )
        else:
            wrapper = wrapper_factory(int(workload_seed))
        binder = getattr(policy, "bind", None)
        if binder is not None:
            binder(wrapper)
        observation, _ = wrapper.reset(seed=int(workload_seed))
        terminated = truncated = False
        episode_reward = 0.0
        queue_samples = []
        max_queue = 0
        info = {}
        component_samples = []
        while not terminated and not truncated:
            queue_lengths = [len(queue) for queue in wrapper.env.queues.values()]
            queue_samples.extend(queue_lengths)
            max_queue = max(max_queue, max(queue_lengths, default=0))
            action = policy(observation, wrapper.get_avail_actions())
            observation, reward, terminated, truncated, info = wrapper.step(action)
            observer = getattr(policy, "observe_transition", None)
            if observer is not None:
                observer(info)
            episode_reward += reward
            component_samples.append(info["reward_components"])

        env = wrapper.env
        delays = [
            env.delivery_slots[packet_id]
            - env.packets[packet_id].created_slot
            + 1
            for packet_id in env.delivered
        ]
        class_ratios = []
        for traffic_class in range(3):
            generated = sum(
                packet.traffic_class == traffic_class
                for packet in env.packets.values()
            )
            delivered = sum(
                env.packets[packet_id].traffic_class == traffic_class
                for packet_id in env.delivered
            )
            class_ratios.append(delivered / max(1, generated))
        slots = max(1, env.slot - 1)

        def component_mean(name: str) -> float:
            return float(
                np.mean([sample[name] for sample in component_samples])
            ) if component_samples else 0.0

        rows.append(
            EpisodeMetrics(
                scenario=scenario,
                policy=policy_name,
                policy_seed=int(policy_seed),
                workload_seed=int(workload_seed),
                generated=len(env.generated),
                delivered=len(env.delivered),
                dropped=len(env.dropped),
                backlog=len(env._backlog_ids()),
                delivery_ratio=len(env.delivered) / max(1, len(env.generated)),
                drop_rate=len(env.dropped) / max(1, len(env.generated)),
                throughput_packets_per_slot=len(env.delivered) / slots,
                average_delay_slots=float(np.mean(delays)) if delays else 0.0,
                p95_delay_slots=float(np.percentile(delays, 95)) if delays else 0.0,
                mean_queue_packets=float(np.mean(queue_samples)) if queue_samples else 0.0,
                max_queue_packets=max_queue,
                routing_switches=int(info.get("routing_switches", 0)),
                episode_reward=float(episode_reward),
                global_delay_cost=component_mean("delay_cost"),
                global_queue_cost=component_mean("queue_cost"),
                global_load_imbalance=component_mean("load_imbalance"),
                global_switch_cost=component_mean("switch_cost"),
                global_throughput_reward=component_mean("throughput_reward"),
                global_control_overhead_ratio=component_mean(
                    "control_overhead_ratio"
                ),
                global_drop_cost=component_mean("drop_cost"),
                class_0_delivery_ratio=class_ratios[0],
                class_1_delivery_ratio=class_ratios[1],
                class_2_delivery_ratio=class_ratios[2],
            )
        )
        wrapper.close()
    return rows


def metrics_as_dicts(rows: Iterable[EpisodeMetrics]) -> list[Dict]:
    return [asdict(row) for row in rows]
