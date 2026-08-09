"""
LEO routing MARL environment interface.

This file follows the research manuscript:
- Time-varying LEO graph G(t) = (V, E(t), X(t))
- Real-time Dec-POMDP style local next-hop decision
- Queue/load/reliability/link-lifetime aware observation
- Action mask for infeasible, risky, soon-broken, congested, or loop-forming links
- AoI and Hello-message control-overhead budget
- Scenario presets for low/mid/high load, hotspot, frequent link break, and fault cases
- Placeholder topology-provider interface for real constellation/TLE-derived topologies
- MAPPO/CleanMARL friendly observation/action-mask/state output

It is intentionally an environment/interface file, not an algorithm demo.
Later it can be wrapped for MAPPO official on-policy, CleanMARL, BenchMARL,
or another mature MARL implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import math
import random

TopologyProvider = Callable[[int, "LeoRoutingEnv"], Dict[Tuple[int, int], "LinkState"]]


@dataclass
class ScenarioConfig:
    name: str = "medium_load"
    background_packets_per_slot: int = 6
    hotspot_packets_per_slot: int = 12
    hotspot_ratio: float = 0.65
    fault_link_ratio: float = 0.0
    reliability_penalty: float = 0.0
    shorten_trem_ratio: float = 0.0
    force_hotspot_dst: Optional[int] = None


SCENARIOS: Dict[str, ScenarioConfig] = {
    "low_load": ScenarioConfig("low_load", background_packets_per_slot=2, hotspot_packets_per_slot=2, hotspot_ratio=0.2),
    "medium_load": ScenarioConfig("medium_load", background_packets_per_slot=6, hotspot_packets_per_slot=12, hotspot_ratio=0.65),
    "hotspot_high_load": ScenarioConfig("hotspot_high_load", background_packets_per_slot=8, hotspot_packets_per_slot=24, hotspot_ratio=0.8),
    "frequent_break": ScenarioConfig("frequent_break", background_packets_per_slot=6, hotspot_packets_per_slot=12, shorten_trem_ratio=0.35),
    "fault_links": ScenarioConfig("fault_links", background_packets_per_slot=6, hotspot_packets_per_slot=12, fault_link_ratio=0.08, reliability_penalty=0.25),
}


@dataclass
class EnvConfig:
    n_planes: int = 4
    sats_per_plane: int = 6
    max_steps: int = 30
    max_local_hops: int = 12
    shell_id: int = 1

    q_max: int = 45
    service_packets_per_slot: int = 3
    capacity_mbps: float = 100.0
    packet_demand_mbps: float = 1.0
    d_ref_ms: float = 20.0

    t_safe: float = 3.0
    r_min: float = 0.86
    b_min_mbps: float = 5.0
    load_decay: float = 0.55

    hello_period_slots: int = 1
    hello_fields: Tuple[str, ...] = ("queue", "load", "reliability", "t_rem")
    bytes_node_id: int = 2
    bytes_timestamp: int = 4
    bytes_queue: int = 2
    bytes_load: int = 2
    bytes_reliability: int = 2
    bytes_t_rem: int = 2
    control_budget_ratio: float = 0.05

    w_deliver: float = 2.0
    w_delay: float = 1.5
    w_queue: float = 1.0
    w_load: float = 1.0
    w_risk: float = 2.0
    w_lifetime: float = 1.0
    w_loop: float = 3.0
    w_invalid: float = 2.0
    w_progress: float = 0.5
    w_overhead: float = 0.2
    w_switch: float = 0.2

    seed: int = 11
    scenario: ScenarioConfig = field(default_factory=lambda: SCENARIOS["medium_load"])
    topology_provider: Optional[TopologyProvider] = None

    @property
    def n_sats(self) -> int:
        return self.n_planes * self.sats_per_plane


@dataclass
class PacketState:
    src: int
    dst: int
    current: int
    visited: List[int]
    hop_count: int = 0
    delivered: bool = False
    dropped: bool = False
    last_next_hop: Optional[int] = None


@dataclass
class LinkState:
    delay_ms: float
    capacity_mbps: float
    used_rate_mbps: float
    rho: float
    reliability: float
    p_out: float
    t_rem: float
    available: bool
    is_cross: bool = False
    shell_src: int = 1
    shell_dst: int = 1


class LeoRoutingEnv:
    """Single-packet local next-hop environment for later MARL wrapping.

    reset(src=None, dst=None) -> dict observation
    step(action_index) -> obs, reward, terminated, truncated, info

    Action is an index over the current node's neighbor list, not a global node id.
    For PPO/MAPPO-style code, use as_mappo_inputs(obs). Candidate rows keep a
    fixed feature schema so a shared candidate scorer can process them without
    depending on neighbor ordering.
    """

    max_degree = 6
    candidate_feature_dim = 20

    def __init__(self, cfg: Optional[EnvConfig] = None):
        self.cfg = cfg or EnvConfig()
        self.rng = random.Random(self.cfg.seed)
        self.time_slot = 1
        self.queues = [0.0 for _ in range(self.cfg.n_sats + 1)]
        self.used_rate = [[0.0 for _ in range(self.cfg.n_sats + 1)] for _ in range(self.cfg.n_sats + 1)]
        self.age = [0 for _ in range(self.cfg.n_sats + 1)]
        self.packet: Optional[PacketState] = None
        self.graph: Dict[Tuple[int, int], LinkState] = {}
        self.generated_packets = 0
        self.delivered_packets = 0
        self.dropped_packets = 0
        self.control_bytes_acc = 0
        self.data_bytes_acc = 0
        self.switch_count = 0
        self.fault_links = set()
        self.short_trem_links = set()
        self._init_scenario_faults()

    @classmethod
    def from_scenario(cls, name: str) -> "LeoRoutingEnv":
        cfg = EnvConfig(scenario=SCENARIOS[name])
        return cls(cfg)

    def reset(self, src: Optional[int] = None, dst: Optional[int] = None) -> Dict:
        self.time_slot = 1
        self.queues = [0.0 for _ in range(self.cfg.n_sats + 1)]
        self.used_rate = [[0.0 for _ in range(self.cfg.n_sats + 1)] for _ in range(self.cfg.n_sats + 1)]
        self.age = [0 for _ in range(self.cfg.n_sats + 1)]
        self.generated_packets = 1
        self.delivered_packets = 0
        self.dropped_packets = 0
        self.control_bytes_acc = 0
        self.data_bytes_acc = 0
        self.switch_count = 0
        self._init_scenario_faults()
        src = src or self.rng.randint(1, self.cfg.n_sats)
        dst = dst or self._sample_destination(src)
        self.packet = PacketState(src=src, dst=dst, current=src, visited=[src])
        self.graph = self._build_topology(self.time_slot)
        return self._make_obs()

    def step(self, action_index: int):
        if self.packet is None:
            raise RuntimeError("Call reset() before step().")
        if self.packet.delivered or self.packet.dropped:
            raise RuntimeError("Episode already ended. Call reset().")

        obs = self._make_obs()
        neighbors = obs["neighbor_ids"]
        mask = obs["action_mask"]
        invalid = action_index < 0 or action_index >= len(neighbors) or not mask[action_index]
        if invalid:
            self.packet.dropped = True
            self.dropped_packets += 1
            reward = -self.cfg.w_invalid
            return self._make_obs(), reward, True, False, {
                "status": "invalid_action",
                "current": self.packet.current,
                "action_mask": mask,
                "control_overhead_ratio": self.control_overhead_ratio(),
            }

        next_hop = neighbors[action_index]
        edge = self.graph[(self.packet.current, next_hop)]
        reward = self._local_reward(self.packet.current, next_hop, edge)
        if self.packet.last_next_hop is not None and next_hop != self.packet.last_next_hop:
            self.switch_count += 1
            reward -= self.cfg.w_switch

        self._apply_forwarding_load(self.packet.current, next_hop)
        self.packet.last_next_hop = next_hop
        self.packet.current = next_hop
        self.packet.visited.append(next_hop)
        self.packet.hop_count += 1

        terminated = False
        truncated = False
        status = "forwarded"
        if next_hop == self.packet.dst:
            self.packet.delivered = True
            self.delivered_packets += 1
            reward += self.cfg.w_deliver
            terminated = True
            status = "delivered"
        elif self.packet.hop_count >= self.cfg.max_local_hops:
            self.packet.dropped = True
            self.dropped_packets += 1
            reward -= self.cfg.w_loop
            # TTL exhaustion is an absorbing packet drop, not a time-limit cut.
            terminated = True
            status = "ttl_exceeded"

        self._advance_time()
        overhead_ratio = self.control_overhead_ratio()
        if overhead_ratio > self.cfg.control_budget_ratio:
            reward -= self.cfg.w_overhead * (overhead_ratio / self.cfg.control_budget_ratio - 1.0)

        next_obs = self._make_obs()
        info = {
            "status": status,
            "selected_next_hop": next_hop,
            "current": self.packet.current,
            "dst": self.packet.dst,
            "hop_count": self.packet.hop_count,
            "visited": list(self.packet.visited),
            "action_mask": next_obs["action_mask"],
            "control_overhead_bytes": self.control_bytes_acc,
            "control_overhead_ratio": overhead_ratio,
            "decision_flops": self.estimate_decision_flops(len(next_obs["neighbor_ids"])),
            "scenario": self.cfg.scenario.name,
        }
        return next_obs, reward, terminated, truncated, info

    def _make_obs(self) -> Dict:
        assert self.packet is not None
        self.graph = self._build_topology(self.time_slot)
        current = self.packet.current
        dst = self.packet.dst
        neighbors = self._neighbors(current)
        current_u, current_w = self.orbital_geodetic_coord(current, self.time_slot)
        dst_u, dst_w = self.orbital_geodetic_coord(dst, self.time_slot)
        hop_ratio = self.packet.hop_count / max(1, self.cfg.max_local_hops)
        remaining_hop_ratio = max(
            0.0,
            (self.cfg.max_local_hops - self.packet.hop_count)
            / max(1, self.cfg.max_local_hops),
        )
        visited_ratio = len(set(self.packet.visited)) / max(1, self.cfg.n_sats)
        topology_phase = (self.time_slot % max(1, self.cfg.max_steps)) / max(
            1, self.cfg.max_steps
        )
        features = []
        mask = []
        for j in neighbors:
            edge = self.graph[(current, j)]
            u_coord, w_coord = self.orbital_geodetic_coord(j, self.time_slot)
            switch_if_selected = float(
                self.packet.last_next_hop is not None
                and j != self.packet.last_next_hop
            )
            features.append([
                self.queues[current] / self.cfg.q_max,
                self.queues[j] / self.cfg.q_max,
                edge.delay_ms / self.cfg.d_ref_ms,
                self._remaining_bandwidth(current, j) / self.cfg.capacity_mbps,
                edge.rho,
                edge.reliability,
                min(1.0, edge.t_rem / self.cfg.t_safe),
                self.age[j] / max(1, self.cfg.hello_period_slots),
                self._progress_value(current, j, dst),
                u_coord,
                w_coord,
                current_u,
                current_w,
                dst_u,
                dst_w,
                remaining_hop_ratio,
                hop_ratio,
                switch_if_selected,
                visited_ratio,
                topology_phase,
            ])
            mask.append(self._action_feasible(current, j))
        return {
            "time_slot": self.time_slot,
            "current": current,
            "dst": dst,
            "visited": list(self.packet.visited),
            "neighbor_ids": neighbors,
            "neighbor_features": features,
            "action_mask": mask,
            "packet_context": {
                "hop_ratio": hop_ratio,
                "remaining_hop_ratio": remaining_hop_ratio,
                "visited_ratio": visited_ratio,
                "last_next_hop": self.packet.last_next_hop,
                "current_coord": [current_u, current_w],
                "destination_coord": [dst_u, dst_w],
                "topology_phase": topology_phase,
            },
            "global_state_for_critic": self._global_state_summary(),
        }

    def as_mappo_inputs(self, obs: Dict) -> Dict:
        """Return a simple MAPPO/CleanMARL friendly representation.

        actor_obs is flattened only for API compatibility. The revised actor
        reshapes it back to [candidate, feature] and scores every candidate with
        shared parameters. critic_state is packet-conditioned and normalized.
        """
        max_degree = self.max_degree
        feat_dim = self.candidate_feature_dim
        padded = [[0.0] * feat_dim for _ in range(max_degree)]
        mask = [False] * max_degree
        for i, feat in enumerate(obs["neighbor_features"][:max_degree]):
            if len(feat) != feat_dim:
                raise ValueError(
                    f"candidate feature length {len(feat)} != expected {feat_dim}"
                )
            padded[i] = feat
            mask[i] = bool(obs["action_mask"][i])
        actor_obs = [x for row in padded for x in row]
        critic_state = self._critic_state_vector(obs, padded, mask)
        return {
            "actor_obs": actor_obs,
            "candidate_obs": padded,
            "critic_state": critic_state,
            "action_mask": mask,
        }

    def _critic_state_vector(
        self, obs: Dict, padded_candidates: List[List[float]], mask: List[bool]
    ) -> List[float]:
        """Normalized, packet-conditioned state for the centralized critic."""
        assert self.packet is not None
        state = obs["global_state_for_critic"]
        ctx = obs["packet_context"]
        last_u, last_w = (0.0, 0.0)
        has_last_hop = float(self.packet.last_next_hop is not None)
        if self.packet.last_next_hop is not None:
            last_u, last_w = self.orbital_geodetic_coord(
                self.packet.last_next_hop, self.time_slot
            )

        global_part = [
            state["avg_queue"] / max(1, self.cfg.q_max),
            state["max_queue"] / max(1, self.cfg.q_max),
            state["avg_rho"],
            state["jain_load"],
            state["num_available_links"] / max(1, self.cfg.n_sats * 4),
            min(2.0, state["control_overhead_ratio"] / max(1e-6, self.cfg.control_budget_ratio)),
            state["delivery_ratio"],
            state["drop_rate"],
            state.get("switch_count", 0) / max(1, self.cfg.max_local_hops),
        ]
        packet_part = [
            *ctx["current_coord"],
            *ctx["destination_coord"],
            ctx["hop_ratio"],
            ctx["remaining_hop_ratio"],
            ctx["visited_ratio"],
            has_last_hop,
            last_u,
            last_w,
            ctx["topology_phase"],
        ]
        candidate_part = [x for row in padded_candidates for x in row]
        mask_part = [float(x) for x in mask]
        return global_part + packet_part + candidate_part + mask_part

    def _action_feasible(self, u: int, v: int) -> bool:
        assert self.packet is not None
        if v in self.packet.visited:
            return False
        edge = self.graph.get((u, v))
        if edge is None or not edge.available:
            return False
        if self._remaining_bandwidth(u, v) < self.cfg.b_min_mbps:
            return False
        if self.queues[v] >= self.cfg.q_max:
            return False
        if edge.reliability < self.cfg.r_min:
            return False
        if edge.t_rem < self.cfg.t_safe:
            return False
        return True

    def _local_reward(self, u: int, v: int, edge: LinkState) -> float:
        delay_cost = edge.delay_ms / self.cfg.d_ref_ms
        queue_cost = self.queues[v] / self.cfg.q_max
        load_cost = edge.rho
        risk_cost = 1.0 - edge.reliability
        lifetime_cost = self.cfg.t_safe / max(self.cfg.t_safe, edge.t_rem)
        progress_reward = self._progress_value(u, v, self.packet.dst if self.packet else v)
        return (
            - self.cfg.w_delay * delay_cost
            - self.cfg.w_queue * queue_cost
            - self.cfg.w_load * load_cost
            - self.cfg.w_risk * risk_cost
            - self.cfg.w_lifetime * lifetime_cost
            + self.cfg.w_progress * progress_reward
        )

    def _apply_forwarding_load(self, u: int, v: int) -> None:
        self.queues[v] = min(self.cfg.q_max + 1, self.queues[v] + 1)
        self.used_rate[u][v] += self.cfg.packet_demand_mbps
        self.used_rate[v][u] += self.cfg.packet_demand_mbps
        self.data_bytes_acc += int(self.cfg.packet_demand_mbps * 125000)

    def _advance_time(self) -> None:
        self.time_slot += 1
        self.queues = [max(0.0, q - self.cfg.service_packets_per_slot) for q in self.queues]
        for u in range(1, self.cfg.n_sats + 1):
            for v in range(1, self.cfg.n_sats + 1):
                self.used_rate[u][v] *= self.cfg.load_decay
        self.age = [a + 1 for a in self.age]
        if self.time_slot % self.cfg.hello_period_slots == 0:
            self.control_bytes_acc += self.estimate_control_overhead_bytes()
            self.age = [0 for _ in self.age]

    def estimate_control_overhead_bytes(self) -> int:
        fields = self.cfg.bytes_node_id + self.cfg.bytes_timestamp
        if "queue" in self.cfg.hello_fields:
            fields += self.cfg.bytes_queue
        if "load" in self.cfg.hello_fields:
            fields += self.cfg.bytes_load
        if "reliability" in self.cfg.hello_fields:
            fields += self.cfg.bytes_reliability
        if "t_rem" in self.cfg.hello_fields:
            fields += self.cfg.bytes_t_rem
        return len(self.graph) * fields

    def control_overhead_ratio(self) -> float:
        return self.control_bytes_acc / max(1, self.data_bytes_acc)

    def estimate_decision_flops(self, neighbor_count: int) -> int:
        z_dim, h1, h2 = self.candidate_feature_dim, 32, 16
        macs_per_neighbor = z_dim * h1 + h1 * h2 + h2
        return 2 * neighbor_count * macs_per_neighbor

    def _global_state_summary(self) -> Dict:
        rhos = [edge.rho for edge in self.graph.values() if edge.available]
        generated = max(1, self.generated_packets)
        return {
            "avg_queue": sum(self.queues[1:]) / self.cfg.n_sats,
            "max_queue": max(self.queues[1:]),
            "avg_rho": sum(rhos) / max(1, len(rhos)),
            "jain_load": self._jain_index(rhos),
            "num_available_links": len(rhos),
            "control_overhead_ratio": self.control_overhead_ratio(),
            "delivery_ratio": self.delivered_packets / generated,
            "drop_rate": self.dropped_packets / generated,
            "switch_count": self.switch_count,
        }

    def _build_topology(self, t: int) -> Dict[Tuple[int, int], LinkState]:
        if self.cfg.topology_provider is not None:
            return self.cfg.topology_provider(t, self)
        graph: Dict[Tuple[int, int], LinkState] = {}
        for p in range(1, self.cfg.n_planes + 1):
            for s in range(1, self.cfg.sats_per_plane + 1):
                u = self._sat_id(p, s)
                v_same = self._sat_id(p, s % self.cfg.sats_per_plane + 1)
                self._add_undirected_link(graph, u, v_same, self._same_orbit_delay_ms(t, p, s), False, p, s)
                next_plane = p % self.cfg.n_planes + 1
                if self._cross_plane_available(t, p, s):
                    v_cross = self._sat_id(next_plane, s)
                    self._add_undirected_link(graph, u, v_cross, self._cross_plane_delay_ms(t, p, s), True, p, s)
        return graph

    def _add_undirected_link(self, graph: Dict[Tuple[int, int], LinkState], u: int, v: int, delay: float, is_cross: bool, plane: int, pos: int) -> None:
        self._add_directed_link(graph, u, v, delay, is_cross, plane, pos)
        self._add_directed_link(graph, v, u, delay, is_cross, plane, pos)

    def _add_directed_link(self, graph: Dict[Tuple[int, int], LinkState], u: int, v: int, delay: float, is_cross: bool, plane: int, pos: int) -> None:
        used = self.used_rate[u][v]
        rho = min(1.0, used / self.cfg.capacity_mbps)
        reliability = max(0.80, 0.995 - 0.12 * rho)
        key = (u, v)
        if key in self.fault_links:
            reliability = max(0.0, reliability - self.cfg.scenario.reliability_penalty)
        t_rem = self._remaining_time_seconds_for_link(is_cross, plane, pos)
        if key in self.short_trem_links:
            t_rem = min(t_rem, max(1.0, self.cfg.t_safe - 0.5))
        graph[key] = LinkState(
            delay_ms=delay,
            capacity_mbps=self.cfg.capacity_mbps,
            used_rate_mbps=used,
            rho=rho,
            reliability=reliability,
            p_out=1.0 - reliability,
            t_rem=t_rem,
            available=True,
            is_cross=is_cross,
            shell_src=self.cfg.shell_id,
            shell_dst=self.cfg.shell_id,
        )

    def orbital_geodetic_coord(self, node_id: int, t: int) -> Tuple[float, float]:
        """SHORT-inspired orbital-geodetic coordinate placeholder.

        This is not a full TLE/SGP4 implementation. It gives the environment a
        stable (U,W) feature slot. A real topology provider can replace it with
        TLE/Hypatia/StarryNet-derived orbital-geodetic coordinates.
        """
        plane, pos = self._decode_sat(node_id)
        u0 = (plane - 1) / max(1, self.cfg.n_planes)
        w0 = (pos - 1) / max(1, self.cfg.sats_per_plane)
        u = (u0 + t / max(1, self.cfg.max_steps * self.cfg.n_planes)) % 1.0
        w = (w0 + t / max(1, self.cfg.max_steps)) % 1.0
        return u, w

    def _init_scenario_faults(self) -> None:
        self.fault_links = set()
        self.short_trem_links = set()
        all_edges = []
        for p in range(1, self.cfg.n_planes + 1):
            for s in range(1, self.cfg.sats_per_plane + 1):
                u = self._sat_id(p, s)
                all_edges.append((u, self._sat_id(p, s % self.cfg.sats_per_plane + 1)))
                all_edges.append((u, self._sat_id(p % self.cfg.n_planes + 1, s)))
        for u, v in all_edges:
            if self.rng.random() < self.cfg.scenario.fault_link_ratio:
                self.fault_links.add((u, v))
                self.fault_links.add((v, u))
            if self.rng.random() < self.cfg.scenario.shorten_trem_ratio:
                self.short_trem_links.add((u, v))
                self.short_trem_links.add((v, u))

    def sample_traffic_batch(self) -> List[Tuple[int, int]]:
        pairs = []
        hotspot_dst = self.cfg.scenario.force_hotspot_dst or self._sat_id(math.ceil(self.cfg.n_planes / 2), math.ceil(self.cfg.sats_per_plane / 2))
        for _ in range(self.cfg.scenario.background_packets_per_slot):
            src = self.rng.randint(1, self.cfg.n_sats)
            dst = self._sample_destination(src)
            pairs.append((src, dst))
        for _ in range(self.cfg.scenario.hotspot_packets_per_slot):
            src = self.rng.randint(1, self.cfg.n_sats)
            if src != hotspot_dst:
                pairs.append((src, hotspot_dst))
        return pairs

    def _sample_destination(self, src: int) -> int:
        if self.rng.random() < self.cfg.scenario.hotspot_ratio:
            dst = self.cfg.scenario.force_hotspot_dst or self._sat_id(math.ceil(self.cfg.n_planes / 2), math.ceil(self.cfg.sats_per_plane / 2))
            if dst != src:
                return dst
        dst = self.rng.randint(1, self.cfg.n_sats)
        while dst == src:
            dst = self.rng.randint(1, self.cfg.n_sats)
        return dst

    def _neighbors(self, u: int) -> List[int]:
        return sorted(v for (src, v) in self.graph.keys() if src == u)

    def _remaining_bandwidth(self, u: int, v: int) -> float:
        edge = self.graph[(u, v)]
        return edge.capacity_mbps - edge.used_rate_mbps

    def _sat_id(self, plane: int, pos: int) -> int:
        return (plane - 1) * self.cfg.sats_per_plane + pos

    def _decode_sat(self, node_id: int) -> Tuple[int, int]:
        plane = (node_id - 1) // self.cfg.sats_per_plane + 1
        pos = (node_id - 1) % self.cfg.sats_per_plane + 1
        return plane, pos

    def _same_orbit_delay_ms(self, t: int, plane: int, pos: int) -> float:
        return 8.0 + 1.5 * math.sin(2 * math.pi * (t + pos) / self.cfg.max_steps)

    def _cross_plane_delay_ms(self, t: int, plane: int, pos: int) -> float:
        return 12.0 + 2.0 * math.cos(2 * math.pi * (t + plane + pos) / self.cfg.max_steps)

    def _cross_plane_available(self, t: int, plane: int, pos: int) -> bool:
        phase = ((t + pos - 2) % self.cfg.max_steps) / self.cfg.max_steps
        seam_link = plane == self.cfg.n_planes
        return not (seam_link and 0.45 <= phase <= 0.65)

    def _remaining_time_seconds_for_link(self, is_cross: bool, plane: int, pos: int) -> float:
        if not is_cross:
            return 999.0
        return 2.0 + ((plane + pos) % 8)

    def _progress_value(self, u: int, v: int, dst: int) -> float:
        pu, su = self._decode_sat(u)
        pv, sv = self._decode_sat(v)
        pd, sd = self._decode_sat(dst)
        dist_u = self._torus_distance(pu, su, pd, sd)
        dist_v = self._torus_distance(pv, sv, pd, sd)
        return (dist_u - dist_v) / max(1.0, self.cfg.n_planes + self.cfg.sats_per_plane)

    def _torus_distance(self, p1: int, s1: int, p2: int, s2: int) -> int:
        dp = abs(p1 - p2)
        dp = min(dp, self.cfg.n_planes - dp)
        ds = abs(s1 - s2)
        ds = min(ds, self.cfg.sats_per_plane - ds)
        return dp + ds

    @staticmethod
    def _jain_index(values: List[float]) -> float:
        vals = [x for x in values if x > 0]
        if not vals:
            return 1.0
        numerator = sum(vals) ** 2
        denominator = len(vals) * sum(x * x for x in vals)
        return numerator / denominator if denominator else 1.0


def run_smoke_episode(scenario: str = "medium_load") -> Dict:
    env = LeoRoutingEnv.from_scenario(scenario)
    obs = env.reset(src=1, dst=12)
    total_reward = 0.0
    info = {"status": "not_started"}
    for _ in range(env.cfg.max_local_hops):
        mappo_inputs = env.as_mappo_inputs(obs)
        valid = [i for i, ok in enumerate(mappo_inputs["action_mask"]) if ok]
        if not valid:
            info = {"status": "no_valid_action", "hop_count": env.packet.hop_count if env.packet else 0}
            break
        obs, reward, terminated, truncated, info = env.step(valid[0])
        total_reward += reward
        if terminated or truncated:
            break
    info["total_reward"] = round(total_reward, 4)
    info["scenario"] = scenario
    return info


if __name__ == "__main__":
    for name in ["low_load", "medium_load", "hotspot_high_load", "frequent_break", "fault_links"]:
        print(run_smoke_episode(name))
