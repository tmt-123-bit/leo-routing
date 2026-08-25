"""Synchronous satellite-level environment for shared-parameter MAPPO.

Every satellite is a logical agent.  All agents observe one frozen slot
snapshot, choose simultaneously, and the environment resolves transmissions in
batch.  This file does not implement MAPPO; it fixes the environment semantics
that must exist before a multi-agent training result is meaningful.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import json
import random
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from leo_marl_env import EnvConfig, LeoRoutingEnv, LinkState, SCENARIOS


NO_OP = 0
EVENT_ORDER = (
    "freeze_snapshot",
    "validate_joint_actions",
    "resolve_link_service",
    "admit_forwarded_packets",
    "expire_packet_deadlines",
    "admit_exogenous_arrivals",
    "advance_time",
)

MULTIAGENT_LOADS = {
    "low_load": (6, 2),
    "medium_load": (12, 6),
    "hotspot_high_load": (20, 16),
    "frequent_break": (12, 6),
    "fault_links": (12, 6),
}


@dataclass
class MultiAgentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    episode_slots: int = 30
    max_queue_packets: int = 45
    exogenous_packets_per_slot: int = 0
    initial_packets: int = 8
    link_capacity_packets: int = 3
    shared_reward: bool = True
    seed: int = 11
    packet_class_probabilities: Tuple[float, ...] = (0.50, 0.30, 0.20)
    packet_class_deadlines: Tuple[int, ...] = (30, 12, 20)
    packet_class_delay_weights: Tuple[float, ...] = (1.0, 1.8, 1.2)
    packet_class_reliability_floor: Tuple[float, ...] = (0.86, 0.88, 0.94)
    variant: str = "full"
    credit_weight: float = 0.25
    global_delay_weight: float = 1.0
    global_queue_weight: float = 0.8
    global_imbalance_weight: float = 0.1
    global_switch_weight: float = 0.2
    global_throughput_weight: float = 2.0
    global_control_weight: float = 0.1
    global_drop_weight: float = 2.0
    data_packet_bytes: int = 1500


@dataclass
class RoutedPacket:
    packet_id: int
    src: int
    dst: int
    owner: int
    created_slot: int
    traffic_class: int = 0
    hop_count: int = 0
    last_next_hop: Optional[int] = None
    previous_node: Optional[int] = None
    visited: Tuple[int, ...] = ()
    cumulative_link_delay_ms: float = 0.0


class SynchronousLeoMultiAgentEnv:
    """FIFO/HOL routing with simultaneous satellite actions."""

    def __init__(self, cfg: Optional[MultiAgentConfig] = None):
        self.cfg = cfg or MultiAgentConfig()
        self.rng = random.Random(self.cfg.seed)
        self.base = LeoRoutingEnv(self.cfg.env)
        self.n_agents = self.cfg.env.n_sats
        self.max_degree = self.base.max_degree
        self.candidate_feature_dim = self.base.candidate_feature_dim + 6
        self.action_size = self.max_degree + 1
        self.slot = 1
        self.next_packet_id = 1
        self.queues: Dict[int, Deque[int]] = {
            i: deque() for i in range(1, self.n_agents + 1)
        }
        self.packets: Dict[int, RoutedPacket] = {}
        self.generated: set[int] = set()
        self.delivered: set[int] = set()
        self.dropped: set[int] = set()
        self.drop_reasons: Dict[int, str] = {}
        self.delivery_slots: Dict[int, int] = {}
        self.drop_slots: Dict[int, int] = {}
        self.graph: Dict[Tuple[int, int], LinkState] = {}
        self.used_rate = [
            [0.0 for _ in range(self.n_agents + 1)]
            for _ in range(self.n_agents + 1)
        ]
        self.last_transition: Dict = {}
        self.trace: List[Dict] = []
        self.route_cache: Dict[Tuple[int, int, int], int] = {}
        self.previous_selected_edges: set[Tuple[int, int]] = set()
        self.previous_blocked_edges: set[Tuple[int, int]] = set()
        self.previous_contention = [0 for _ in range(self.n_agents + 1)]
        self.switch_count = 0

    @classmethod
    def from_scenario(cls, name: str, seed: int = 11):
        env_cfg = EnvConfig(seed=seed, scenario=SCENARIOS[name])
        initial_packets, exogenous_packets = MULTIAGENT_LOADS[name]
        return cls(
            MultiAgentConfig(
                env=env_cfg,
                initial_packets=initial_packets,
                exogenous_packets_per_slot=exogenous_packets,
                seed=seed,
            )
        )

    def reset(
        self,
        seed: Optional[int] = None,
        initial_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Tuple[List[Dict], Dict]:
        if seed is not None:
            self.cfg.seed = seed
            self.cfg.env.seed = seed
        self.rng = random.Random(self.cfg.seed)
        self.base = LeoRoutingEnv(self.cfg.env)
        self.base.rng = random.Random(self.cfg.seed)
        self.slot = 1
        self.next_packet_id = 1
        self.queues = {i: deque() for i in range(1, self.n_agents + 1)}
        self.packets = {}
        self.generated = set()
        self.delivered = set()
        self.dropped = set()
        self.drop_reasons = {}
        self.delivery_slots = {}
        self.drop_slots = {}
        self.used_rate = [
            [0.0 for _ in range(self.n_agents + 1)]
            for _ in range(self.n_agents + 1)
        ]
        self.last_transition = {}
        self.trace = []
        self.route_cache = {}
        self.previous_selected_edges = set()
        self.previous_blocked_edges = set()
        self.previous_contention = [0 for _ in range(self.n_agents + 1)]
        self.switch_count = 0

        if initial_pairs is None:
            initial_pairs = self._sample_initial_pairs(self.cfg.initial_packets)
        for pair in initial_pairs:
            if len(pair) == 2:
                src, dst = pair
                traffic_class = None
            elif len(pair) == 3:
                src, dst, traffic_class = pair
            else:
                raise ValueError("initial packet tuple must be (src, dst) or (src, dst, class)")
            self._create_packet(src, dst, traffic_class=traffic_class)
        self._refresh_graph()
        observations = self.observe()
        return observations, {
            "slot": self.slot,
            "active_agents": self._active_agents(),
            "global_state": self.global_state(),
        }

    def observe(self) -> List[Dict]:
        self._refresh_graph()
        return [self._agent_observation(i) for i in range(1, self.n_agents + 1)]

    def step(
        self,
        actions: Sequence[int],
        agent_order: Optional[Sequence[int]] = None,
    ) -> Tuple[List[Dict], List[float], bool, bool, Dict]:
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions, got {len(actions)}")
        self._refresh_graph()
        frozen_obs = self.observe()
        frozen_graph = dict(self.graph)
        delivered_before = set(self.delivered)
        dropped_before = set(self.dropped)
        order = list(agent_order or range(1, self.n_agents + 1))
        if sorted(order) != list(range(1, self.n_agents + 1)):
            raise ValueError("agent_order must be a permutation of satellite IDs")

        local_rewards = [0.0 for _ in range(self.n_agents)]
        proposals: List[Dict] = []
        mask_ledger: Dict[int, Dict] = {}
        immediate_drops: List[Tuple[int, int, str]] = []

        for sat in order:
            obs = frozen_obs[sat - 1]
            action = int(actions[sat - 1])
            mask_ledger[sat] = {
                "packet_id": obs["hol_packet_id"],
                "mask": list(obs["action_mask"]),
                "reasons": list(obs["mask_reasons"]),
            }
            if obs["hol_packet_id"] is None:
                if action != NO_OP:
                    local_rewards[sat - 1] -= self.cfg.env.w_invalid
                continue

            feasible_count = sum(obs["action_mask"][1:])
            if feasible_count == 0:
                immediate_drops.append((sat, obs["hol_packet_id"], "no_route"))
                local_rewards[sat - 1] -= self.cfg.env.w_invalid
                continue
            if action <= 0 or action >= self.action_size or not obs["action_mask"][action]:
                immediate_drops.append((sat, obs["hol_packet_id"], "invalid_action"))
                local_rewards[sat - 1] -= self.cfg.env.w_invalid
                continue

            next_hop = obs["neighbor_ids"][action - 1]
            packet = self.packets[obs["hol_packet_id"]]
            edge = frozen_graph[(sat, next_hop)]
            reward = self._forward_reward(packet, sat, next_hop, edge)
            is_switch = self._is_route_switch(packet, sat, next_hop)
            proposals.append(
                {
                    "sat": sat,
                    "packet_id": packet.packet_id,
                    "next_hop": next_hop,
                    "reward": reward,
                    "is_switch": is_switch,
                    "edge_delay_ms": edge.delay_ms,
                }
            )

        # Invalid/no-route outcomes are applied before accepted transmissions,
        # but only after every action has been read from the frozen snapshot.
        for sat, packet_id, reason in sorted(immediate_drops, key=lambda x: x[1]):
            self._remove_hol(sat, packet_id)
            self._drop_packet(packet_id, reason)

        accepted, blocked = self._resolve_link_capacity(proposals)
        transmitted_ids: List[int] = []
        link_counts: Dict[str, int] = defaultdict(int)
        incoming: Dict[int, List[Dict]] = defaultdict(list)

        for item in accepted:
            sat = item["sat"]
            packet_id = item["packet_id"]
            next_hop = item["next_hop"]
            self._remove_hol(sat, packet_id)
            incoming[next_hop].append(item)
            transmitted_ids.append(packet_id)
            link_counts[f"{sat}->{next_hop}"] += 1
            local_rewards[sat - 1] += item["reward"]
            self.used_rate[sat][next_hop] += self.cfg.env.packet_demand_mbps
            packet = self.packets[packet_id]
            self.route_cache[(sat, packet.dst, packet.traffic_class)] = next_hop
            self.switch_count += int(item["is_switch"])

        for item in blocked:
            local_rewards[item["sat"] - 1] -= self.cfg.env.w_load

        # Queue admission is deterministic by packet ID, not by satellite loop order.
        for next_hop in sorted(incoming):
            for item in sorted(incoming[next_hop], key=lambda x: x["packet_id"]):
                packet = self.packets[item["packet_id"]]
                packet.hop_count += 1
                packet.cumulative_link_delay_ms += item["edge_delay_ms"]
                packet.previous_node = item["sat"]
                packet.last_next_hop = next_hop
                packet.owner = next_hop
                packet.visited = packet.visited + (next_hop,)
                if next_hop == packet.dst:
                    self.delivered.add(packet.packet_id)
                    self.delivery_slots[packet.packet_id] = self.slot
                    local_rewards[item["sat"] - 1] += self.cfg.env.w_deliver
                elif packet.hop_count >= self.cfg.env.max_local_hops:
                    self._drop_packet(packet.packet_id, "ttl_exceeded")
                    local_rewards[item["sat"] - 1] -= self.cfg.env.w_loop
                elif len(self.queues[next_hop]) >= self.cfg.max_queue_packets:
                    self._drop_packet(packet.packet_id, "queue_overflow")
                    local_rewards[item["sat"] - 1] -= self.cfg.env.w_invalid
                else:
                    self.queues[next_hop].append(packet.packet_id)

        deadline_dropped = self._expire_deadline_packets()
        for packet_id in deadline_dropped:
            packet = self.packets[packet_id]
            # Attribute a deadline drop to the agent that made the final routing
            # decision (previous_node), not the current owner, which may have
            # just received the packet this slot with no chance to act. Fall
            # back to owner for packets never forwarded (still at source).
            blame = packet.previous_node if packet.previous_node is not None else packet.owner
            local_rewards[blame - 1] -= self.cfg.env.w_invalid

        exogenous_admitted: List[int] = []
        exogenous_dropped: List[int] = []
        if self.cfg.exogenous_packets_per_slot > 0:
            for src, dst in self._sample_initial_pairs(
                self.cfg.exogenous_packets_per_slot
            ):
                if len(self.queues[src]) < self.cfg.max_queue_packets:
                    exogenous_admitted.append(self._create_packet(src, dst))
                else:
                    packet_id = self._create_packet(src, dst)
                    if self.queues[src][-1] != packet_id:
                        raise AssertionError(
                            "new exogenous packet was not appended at queue tail"
                        )
                    self.queues[src].pop()
                    self._drop_packet(packet_id, "source_queue_overflow")
                    exogenous_dropped.append(packet_id)

        delivered_this_slot = self.delivered - delivered_before
        dropped_this_slot = self.dropped - dropped_before
        reward_components = self._global_reward_components(
            accepted=accepted,
            frozen_graph=frozen_graph,
            delivered_this_slot=delivered_this_slot,
            dropped_this_slot=dropped_this_slot,
            active_count=sum(obs["hol_packet_id"] is not None for obs in frozen_obs),
        )
        global_reward = reward_components["team_reward"]
        policy_active = [
            # Active = holding a HOL packet. Agents in a no_route state (HOL
            # packet but no feasible candidate) are included so their no_route
            # penalty receives centered credit; previously the any(...) guard
            # zeroed their credit via the float(policy_active) factor, so they
            # faced no individual consequence for dead-end routing.
            obs["hol_packet_id"] is not None
            for obs in frozen_obs
        ]
        active_local = [
            local_rewards[index]
            for index, active in enumerate(policy_active)
            if active
        ]
        local_mean = sum(active_local) / max(1, len(active_local))
        credit_weight = 0.0 if self.cfg.variant == "no_credit" else self.cfg.credit_weight
        if self.cfg.shared_reward:
            agent_rewards = [
                global_reward
                + credit_weight
                * (local_rewards[index] - local_mean)
                * float(policy_active[index])
                for index in range(self.n_agents)
            ]
        else:
            agent_rewards = list(local_rewards)

        concurrent_actions = len(accepted)

        transition = {
            "slot": self.slot,
            "event_order": list(EVENT_ORDER),
            "actions": [int(a) for a in actions],
            "accepted": sorted(transmitted_ids),
            "blocked": sorted(x["packet_id"] for x in blocked),
            "link_counts": dict(sorted(link_counts.items())),
            "mask_ledger": mask_ledger,
            "concurrent_non_noop": concurrent_actions,
            "global_reward": global_reward,
            "agent_rewards": list(agent_rewards),
            "local_rewards": list(local_rewards),
            "reward_components": reward_components,
            "exogenous_admitted": sorted(exogenous_admitted),
            "exogenous_dropped": sorted(exogenous_dropped),
            "deadline_dropped": sorted(deadline_dropped),
            "incoming_attempts": {
                str(node): len(items) for node, items in sorted(incoming.items())
            },
        }
        self.previous_selected_edges = {
            (item["sat"], item["next_hop"]) for item in accepted
        }
        self.previous_blocked_edges = {
            (item["sat"], item["next_hop"]) for item in blocked
        }
        self.previous_contention = [0 for _ in range(self.n_agents + 1)]
        for node, items in incoming.items():
            self.previous_contention[node] = max(0, len(items) - 1)
        self.last_transition = transition
        self.trace.append(transition)
        self._decay_load()
        self.slot += 1
        self.validate_invariants()

        terminated = len(self._backlog_ids()) == 0 and self.cfg.exogenous_packets_per_slot == 0
        truncated = self.slot > self.cfg.episode_slots
        next_obs = self.observe()
        info = {
            **transition,
            "delivered": len(self.delivered),
            "dropped": len(self.dropped),
            "backlog": len(self._backlog_ids()),
            "generated": len(self.generated),
            "delivery_ratio": len(self.delivered) / max(1, len(self.generated)),
            "drop_rate": len(self.dropped) / max(1, len(self.generated)),
            "average_delay_slots": self.average_delivery_delay_slots(),
            "routing_switches": self.switch_count,
            "trace_hash": self.trace_hash(),
            "global_state": self.global_state(),
        }
        return next_obs, agent_rewards, terminated, truncated, info

    def global_state(self) -> Dict:
        self._refresh_graph()
        observations = [
            self._agent_observation(sat) for sat in range(1, self.n_agents + 1)
        ]
        node_features = []
        for sat in range(1, self.n_agents + 1):
            u, w = self.base.orbital_geodetic_coord(sat, self.slot)
            queue_ratio = (
                0.0
                if self.cfg.variant == "no_queue"
                else len(self.queues[sat]) / max(1, self.cfg.max_queue_packets)
            )
            if self.queues[sat]:
                packet = self.packets[self.queues[sat][0]]
                su, sw = self.base.orbital_geodetic_coord(packet.src, self.slot)
                du, dw = self.base.orbital_geodetic_coord(packet.dst, self.slot)
                class_one_hot = [
                    float(packet.traffic_class == idx)
                    for idx in range(len(self.cfg.packet_class_probabilities))
                ]
                wait_ratio = self._waiting_ratio(packet)
                hop_ratio = packet.hop_count / max(
                    1, self.cfg.env.max_local_hops
                )
                packet_features = [
                    1.0,
                    su,
                    sw,
                    du,
                    dw,
                    *class_one_hot,
                    wait_ratio,
                    hop_ratio,
                    max(0.0, 1.0 - hop_ratio),
                    len(set(packet.visited)) / max(1, self.n_agents),
                    min(1.0, wait_ratio),
                ]
                if self.cfg.variant == "no_packet_context":
                    packet_features = [1.0] + [0.0] * 12
            else:
                packet_features = [0.0] * 13
            node_features.append(
                [
                    queue_ratio,
                    u,
                    w,
                    *packet_features,
                    *[
                        float(x)
                        for x in observations[sat - 1]["action_mask"]
                    ],
                    min(
                        1.0,
                        self.previous_contention[sat]
                        / max(1, self.max_degree),
                    ),
                    float(
                        any(
                            src == sat
                            for src, _ in self.previous_selected_edges
                        )
                    ),
                ]
            )

        edge_feature_dim = 11
        edge_features = [
            [[0.0] * edge_feature_dim for _ in range(self.n_agents)]
            for _ in range(self.n_agents)
        ]
        feasible_edges = set()
        for sat, obs in enumerate(observations, start=1):
            for index, neighbor in enumerate(obs["neighbor_ids"], start=1):
                if neighbor and obs["action_mask"][index]:
                    feasible_edges.add((sat, neighbor))
        for (src, dst), edge in self.graph.items():
            edge_features[src - 1][dst - 1] = [
                1.0,
                edge.delay_ms / self.cfg.env.d_ref_ms,
                self._remaining_bandwidth(src, dst)
                / self.cfg.env.capacity_mbps,
                edge.rho,
                edge.reliability,
                min(1.0, edge.t_rem / self.cfg.env.t_safe)
                * float(self.cfg.variant != "no_lifetime"),
                float(edge.is_cross),
                float((src, dst) in feasible_edges),
                float((src, dst) in self.previous_selected_edges),
                float((src, dst) in self.previous_blocked_edges),
                min(
                    1.0,
                    self.previous_contention[dst]
                    / max(1, self.max_degree),
                ),
            ]
        generated = max(1, len(self.generated))
        return {
            "schema_version": 2,
            "global_features": [
                (self.slot % self.cfg.env.max_steps)
                / max(1, self.cfg.env.max_steps),
                len(self._backlog_ids())
                / max(1, self.n_agents * self.cfg.max_queue_packets)
                * float(self.cfg.variant != "no_queue"),
                len(self.delivered) / generated,
                len(self.dropped) / generated,
                sum(self.previous_contention)
                / max(1, self.n_agents * self.max_degree),
            ],
            "node_features": node_features,
            "edge_features": edge_features,
            "schema": {
                "global_feature_dim": 5,
                "node_feature_dim": len(node_features[0]),
                "edge_feature_dim": edge_feature_dim,
                "n_nodes": self.n_agents,
            },
        }

    def validate_invariants(self) -> None:
        backlog = self._backlog_ids()
        if len(backlog) != sum(len(q) for q in self.queues.values()):
            raise AssertionError("single ownership violated: duplicate packet in queues")
        terminal = self.delivered | self.dropped
        if self.delivered & self.dropped:
            raise AssertionError("delivered and dropped sets overlap")
        if backlog & terminal:
            raise AssertionError("terminal packet still has queue ownership")
        if self.generated != backlog | terminal:
            raise AssertionError("packet identity conservation failed")
        accepted = self.last_transition.get("accepted", [])
        if len(accepted) != len(set(accepted)):
            raise AssertionError("packet transmitted more than once in one slot")
        for count in self.last_transition.get("link_counts", {}).values():
            if count > self.cfg.link_capacity_packets:
                raise AssertionError("directed link capacity exceeded")

    def average_delivery_delay_slots(self) -> float:
        if not self.delivery_slots:
            return 0.0
        delays = [
            delivered_slot - self.packets[packet_id].created_slot + 1
            for packet_id, delivered_slot in self.delivery_slots.items()
        ]
        return sum(delays) / len(delays)

    def trace_hash(self) -> str:
        payload = json.dumps(self.trace, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def state_digest(self) -> str:
        state = {
            "slot": self.slot,
            "queues": {str(k): list(v) for k, v in sorted(self.queues.items())},
            "delivered": sorted(self.delivered),
            "dropped": sorted(self.dropped),
            "delivery_slots": dict(sorted(self.delivery_slots.items())),
            "drop_slots": dict(sorted(self.drop_slots.items())),
            "route_cache": {
                f"{sat}:{dst}:{traffic_class}": next_hop
                for (sat, dst, traffic_class), next_hop in sorted(
                    self.route_cache.items()
                )
            },
            "previous_selected_edges": sorted(self.previous_selected_edges),
            "previous_blocked_edges": sorted(self.previous_blocked_edges),
            "previous_contention": self.previous_contention,
            "switch_count": self.switch_count,
            "packets": {
                str(pid): {
                    "owner": p.owner,
                    "hop": p.hop_count,
                    "last": p.last_next_hop,
                    "previous_node": p.previous_node,
                    "class": p.traffic_class,
                    "created": p.created_slot,
                    "cumulative_link_delay_ms": p.cumulative_link_delay_ms,
                    "visited": list(p.visited),
                }
                for pid, p in sorted(self.packets.items())
            },
        }
        payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _agent_observation(self, sat: int) -> Dict:
        queue = self.queues[sat]
        if not queue:
            return {
                "agent_id": sat,
                "hol_packet_id": None,
                "neighbor_ids": [0] * self.max_degree,
                "candidate_features": [[0.0] * self.candidate_feature_dim for _ in range(self.max_degree)],
                "action_mask": [True] + [False] * self.max_degree,
                "mask_reasons": ["inactive_no_op"] + ["inactive"] * self.max_degree,
                "packet_class": None,
                "hol_waiting_ratio": 0.0,
                "visited_summary": [0.0, 0.0],
            }

        packet = self.packets[queue[0]]
        neighbors = self.base._neighbors(sat)[: self.max_degree]
        padded_ids = neighbors + [0] * (self.max_degree - len(neighbors))
        features: List[List[float]] = []
        candidate_mask: List[bool] = []
        reasons: List[str] = []
        for neighbor in padded_ids:
            if neighbor == 0:
                features.append([0.0] * self.candidate_feature_dim)
                candidate_mask.append(False)
                reasons.append("padding")
                continue
            features.append(self._candidate_features(packet, sat, neighbor))
            reason = self._mask_reason(packet, sat, neighbor)
            candidate_mask.append(reason == "feasible")
            reasons.append(reason)
        no_route = not any(candidate_mask)
        return {
            "agent_id": sat,
            "hol_packet_id": packet.packet_id,
            "neighbor_ids": padded_ids,
            "candidate_features": features,
            "action_mask": [no_route] + candidate_mask,
            "mask_reasons": [
                "no_route_forced_no_op" if no_route else "active_no_op_forbidden"
            ]
            + reasons,
            "packet_class": packet.traffic_class,
            "hol_waiting_ratio": self._waiting_ratio(packet),
            "visited_summary": [
                len(set(packet.visited)) / max(1, self.n_agents),
                float(packet.previous_node is not None),
            ],
            "remaining_hop_ratio": max(
                0.0,
                (self.cfg.env.max_local_hops - packet.hop_count)
                / max(1, self.cfg.env.max_local_hops),
            ),
        }

    def _candidate_features(self, packet: RoutedPacket, u: int, v: int) -> List[float]:
        edge = self.graph[(u, v)]
        vu, vw = self.base.orbital_geodetic_coord(v, self.slot)
        uu, uw = self.base.orbital_geodetic_coord(u, self.slot)
        du, dw = self.base.orbital_geodetic_coord(packet.dst, self.slot)
        base_features = [
            len(self.queues[u])
            / max(1, self.cfg.max_queue_packets)
            * float(self.cfg.variant != "no_queue"),
            len(self.queues[v])
            / max(1, self.cfg.max_queue_packets)
            * float(self.cfg.variant != "no_queue"),
            edge.delay_ms / self.cfg.env.d_ref_ms,
            self._remaining_bandwidth(u, v) / self.cfg.env.capacity_mbps,
            edge.rho,
            edge.reliability,
            min(1.0, edge.t_rem / self.cfg.env.t_safe)
            * float(self.cfg.variant != "no_lifetime"),
            self.previous_contention[v] / max(1, self.max_degree),
            self.base._progress_value(u, v, packet.dst),
            vu,
            vw,
            uu,
            uw,
            du,
            dw,
            max(0.0, (self.cfg.env.max_local_hops - packet.hop_count) / max(1, self.cfg.env.max_local_hops)),
            packet.hop_count / max(1, self.cfg.env.max_local_hops),
            float(self._is_route_switch(packet, u, v)),
            len(set(packet.visited)) / max(1, self.n_agents),
            (self.slot % self.cfg.env.max_steps) / max(1, self.cfg.env.max_steps),
        ]
        class_one_hot = [
            float(packet.traffic_class == idx)
            for idx in range(len(self.cfg.packet_class_probabilities))
        ]
        features = [
            *base_features,
            self._waiting_ratio(packet),
            *class_one_hot,
            float(v in packet.visited),
            float(packet.previous_node == v),
        ]
        if self.cfg.variant == "no_packet_context":
            for index in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
                features[index] = 0.0
        return features

    def _mask_reason(self, packet: RoutedPacket, u: int, v: int) -> str:
        if v in packet.visited:
            return "visited"
        edge = self.graph.get((u, v))
        if edge is None or not edge.available:
            return "unavailable"
        if self._remaining_bandwidth(u, v) < self.cfg.env.b_min_mbps:
            return "bandwidth"
        if len(self.queues[v]) >= self.cfg.max_queue_packets:
            return "queue_full"
        reliability_floor = self.cfg.packet_class_reliability_floor[
            packet.traffic_class
        ]
        if edge.reliability < reliability_floor:
            return "reliability"
        if self.cfg.variant != "no_lifetime" and edge.t_rem < self.cfg.env.t_safe:
            return "lifetime"
        return "feasible"

    def _resolve_link_capacity(self, proposals: Iterable[Dict]):
        by_link: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
        for item in proposals:
            by_link[(item["sat"], item["next_hop"])].append(item)
        accepted: List[Dict] = []
        blocked: List[Dict] = []
        for link in sorted(by_link):
            items = sorted(by_link[link], key=lambda x: x["packet_id"])
            accepted.extend(items[: self.cfg.link_capacity_packets])
            blocked.extend(items[self.cfg.link_capacity_packets :])
        return accepted, blocked

    def _global_reward_components(
        self,
        *,
        accepted: Sequence[Dict],
        frozen_graph: Dict[Tuple[int, int], LinkState],
        delivered_this_slot: set[int],
        dropped_this_slot: set[int],
        active_count: int,
    ) -> Dict[str, float]:
        if delivered_this_slot:
            delivered_delay = []
            for packet_id in delivered_this_slot:
                packet = self.packets[packet_id]
                path_delay = packet.cumulative_link_delay_ms / max(
                    self.cfg.env.d_ref_ms,
                    self.cfg.env.d_ref_ms * self.cfg.env.max_local_hops,
                )
                waiting_delay = (
                    self.slot - packet.created_slot + 1
                ) / max(1, self.cfg.packet_class_deadlines[packet.traffic_class])
                delivered_delay.append(min(2.0, 0.5 * path_delay + 0.5 * waiting_delay))
            delay_cost = sum(delivered_delay) / len(delivered_delay)
        elif accepted:
            delay_cost = sum(
                item["edge_delay_ms"] / self.cfg.env.d_ref_ms
                for item in accepted
            ) / len(accepted)
        else:
            # No traffic this slot: no delay penalty (was 1.0, which spuriously
            # penalized idle slots and caused a reward discontinuity vs the
            # delivered/accepted branches).
            delay_cost = 0.0

        queue_cost = sum(len(queue) for queue in self.queues.values()) / max(
            1, self.n_agents * self.cfg.max_queue_packets
        )
        utilizations = [
            min(
                1.0,
                self.used_rate[src][dst]
                / max(1e-6, edge.capacity_mbps),
            )
            for (src, dst), edge in frozen_graph.items()
        ]
        load_imbalance = 1.0 - self._jain_index(utilizations)
        switch_cost = sum(bool(item["is_switch"]) for item in accepted) / max(
            1, active_count
        )
        throughput_reward = len(delivered_this_slot) / max(1, active_count)
        hello_payload_bytes = (
            self.cfg.env.bytes_node_id
            + self.cfg.env.bytes_timestamp
            + self.cfg.env.bytes_queue
            + self.cfg.env.bytes_load
            + self.cfg.env.bytes_reliability
            + self.cfg.env.bytes_t_rem
        )
        hello_bytes = 0.0
        if self.slot % max(1, self.cfg.env.hello_period_slots) == 0:
            hello_bytes = len(frozen_graph) * hello_payload_bytes
        data_bytes = len(accepted) * self.cfg.data_packet_bytes
        control_ratio = hello_bytes / max(1.0, hello_bytes + data_bytes)
        drop_cost = len(dropped_this_slot) / max(1, active_count)

        queue_weight = (
            0.0
            if self.cfg.variant == "no_queue"
            else self.cfg.global_queue_weight
        )
        team_reward = (
            -self.cfg.global_delay_weight * delay_cost
            -queue_weight * queue_cost
            -self.cfg.global_imbalance_weight * load_imbalance
            -self.cfg.global_switch_weight * switch_cost
            +self.cfg.global_throughput_weight * throughput_reward
            -self.cfg.global_control_weight * control_ratio
            -self.cfg.global_drop_weight * drop_cost
        )
        return {
            "team_reward": float(team_reward),
            "delay_cost": float(delay_cost),
            "queue_cost": float(queue_cost),
            "load_imbalance": float(load_imbalance),
            "switch_cost": float(switch_cost),
            "throughput_reward": float(throughput_reward),
            "control_overhead_ratio": float(control_ratio),
            "drop_cost": float(drop_cost),
        }

    @staticmethod
    def _jain_index(values: Sequence[float]) -> float:
        if not values:
            return 1.0
        total = sum(values)
        squares = sum(value * value for value in values)
        if squares <= 1e-12:
            return 1.0
        return total * total / (len(values) * squares)

    def _forward_reward(self, packet: RoutedPacket, u: int, v: int, edge: LinkState) -> float:
        delay_weight = self.cfg.env.w_delay * self.cfg.packet_class_delay_weights[
            packet.traffic_class
        ]
        return (
            -delay_weight * edge.delay_ms / self.cfg.env.d_ref_ms
            -self.cfg.env.w_queue
            * len(self.queues[v])
            / max(1, self.cfg.max_queue_packets)
            * float(self.cfg.variant != "no_queue")
            -self.cfg.env.w_load * edge.rho
            -self.cfg.env.w_risk * (1.0 - edge.reliability)
            -self.cfg.env.w_lifetime
            * self.cfg.env.t_safe
            / max(self.cfg.env.t_safe, edge.t_rem)
            * float(self.cfg.variant != "no_lifetime")
            +self.cfg.env.w_progress * self.base._progress_value(u, v, packet.dst)
            -self.cfg.env.w_switch * float(self._is_route_switch(packet, u, v))
        )

    def _is_route_switch(self, packet: RoutedPacket, sat: int, next_hop: int) -> bool:
        previous = self.route_cache.get((sat, packet.dst, packet.traffic_class))
        return previous is not None and previous != next_hop

    def _waiting_ratio(self, packet: RoutedPacket) -> float:
        deadline = self.cfg.packet_class_deadlines[packet.traffic_class]
        return min(
            2.0,
            max(0, self.slot - packet.created_slot) / max(1, deadline),
        )

    def _refresh_graph(self) -> None:
        self.base.time_slot = self.slot
        self.base.used_rate = self.used_rate
        self.base.queues = [0.0] + [
            float(len(self.queues[i])) for i in range(1, self.n_agents + 1)
        ]
        self.graph = self.base._build_topology(self.slot)
        self.base.graph = self.graph

    def _decay_load(self) -> None:
        for u in range(1, self.n_agents + 1):
            for v in range(1, self.n_agents + 1):
                self.used_rate[u][v] *= self.cfg.env.load_decay

    def _remaining_bandwidth(self, u: int, v: int) -> float:
        edge = self.graph[(u, v)]
        return edge.capacity_mbps - edge.used_rate_mbps

    def _remove_hol(self, sat: int, packet_id: int) -> None:
        if not self.queues[sat] or self.queues[sat][0] != packet_id:
            raise AssertionError("HOL ownership changed during frozen-slot resolution")
        self.queues[sat].popleft()

    def _drop_packet(self, packet_id: int, reason: str) -> None:
        self.dropped.add(packet_id)
        self.drop_reasons[packet_id] = reason
        self.drop_slots[packet_id] = self.slot

    def _expire_deadline_packets(self) -> List[int]:
        expired = []
        for sat in range(1, self.n_agents + 1):
            kept = deque()
            while self.queues[sat]:
                packet_id = self.queues[sat].popleft()
                packet = self.packets[packet_id]
                deadline = self.cfg.packet_class_deadlines[packet.traffic_class]
                age_after_slot = self.slot - packet.created_slot + 1
                if age_after_slot >= deadline:
                    self._drop_packet(packet_id, "deadline_exceeded")
                    expired.append(packet_id)
                else:
                    kept.append(packet_id)
            self.queues[sat] = kept
        return expired

    def _create_packet(
        self, src: int, dst: int, traffic_class: Optional[int] = None
    ) -> int:
        if src == dst:
            raise ValueError("packet source and destination must differ")
        packet_id = self.next_packet_id
        self.next_packet_id += 1
        if traffic_class is None:
            traffic_class = self.rng.choices(
                range(len(self.cfg.packet_class_probabilities)),
                weights=self.cfg.packet_class_probabilities,
                k=1,
            )[0]
        if not 0 <= traffic_class < len(self.cfg.packet_class_probabilities):
            raise ValueError("traffic_class is outside the configured class range")
        self.packets[packet_id] = RoutedPacket(
            packet_id=packet_id,
            src=src,
            dst=dst,
            owner=src,
            created_slot=self.slot,
            traffic_class=traffic_class,
            visited=(src,),
        )
        self.generated.add(packet_id)
        self.queues[src].append(packet_id)
        return packet_id

    def _sample_initial_pairs(self, count: int) -> List[Tuple[int, int]]:
        sources = list(range(1, self.n_agents + 1))
        self.rng.shuffle(sources)
        pairs = []
        for src in sources[: min(count, self.n_agents)]:
            dst = self._sample_destination(src)
            pairs.append((src, dst))
        return pairs

    def _sample_destination(self, src: int) -> int:
        scenario = self.cfg.env.scenario
        hotspot = scenario.force_hotspot_dst or self.base._sat_id(
            (self.cfg.env.n_planes + 1) // 2,
            (self.cfg.env.sats_per_plane + 1) // 2,
        )
        if src != hotspot and self.rng.random() < scenario.hotspot_ratio:
            return hotspot
        dst = self.rng.randint(1, self.n_agents)
        while dst == src:
            dst = self.rng.randint(1, self.n_agents)
        return dst

    def _active_agents(self) -> List[int]:
        return [sat for sat, queue in self.queues.items() if queue]

    def _backlog_ids(self) -> set[int]:
        ids: List[int] = []
        for queue in self.queues.values():
            ids.extend(queue)
        return set(ids)


def first_feasible_actions(observations: Sequence[Dict]) -> List[int]:
    actions = []
    for obs in observations:
        valid = [i for i, allowed in enumerate(obs["action_mask"]) if allowed]
        actions.append(valid[0] if valid else NO_OP)
    return actions


if __name__ == "__main__":
    env = SynchronousLeoMultiAgentEnv.from_scenario("medium_load")
    obs, _ = env.reset(seed=11)
    for _ in range(5):
        obs, rewards, terminated, truncated, info = env.step(
            first_feasible_actions(obs)
        )
        print(
            {
                "slot": info["slot"],
                "concurrent": info["concurrent_non_noop"],
                "delivered": info["delivered"],
                "dropped": info["dropped"],
                "backlog": info["backlog"],
                "reward": round(sum(rewards), 4),
            }
        )
        if terminated or truncated:
            break
