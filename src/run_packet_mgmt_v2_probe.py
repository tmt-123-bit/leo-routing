"""Packet-management v2 development probe: three refinements of the champion
stack (feasibility purge + SRPF), each isolated against the v1 champion arm.

Arms (all environment-side, fixed GlobalDijkstra routing, TLE-66):
  base             FIFO (reference)
  purge_srpf       v1 champion: static-BFS feasibility purge + SRPF (pid tie)
  purge_srpf_dl    v2-C: same purge, SRPF tie-break by deadline (urgent first)
  purge_srpf_aging v2-B: same purge, SRPF with anti-starvation aging
                   (one hop of priority per AGING_PERIOD slots of waiting)
  purge_texp       v2-A: time-expanded feasibility purge - replace the static
                   BFS bound with time-respecting reachability over the
                   deterministic per-slot topology snapshots (the provider is
                   a lookup table, so future adjacency is exact, not guessed),
                   then the same SRPF ordering.

Safety: a packet is dropped only when no schedule of hops over the per-slot
usable links can deliver it by its effective bound. Hops are allowed during
slots self.slot..bound (bound - slot + 1 = remaining + 1, exactly the v1
budget), so the criterion remains a necessary condition for delivery and the
no-false-kill property is preserved. Unlike the v1 static bound, the
time-expanded bound both prunes packets whose only routes cross links that
expire within their deadline and keeps packets whose routes need links that
form later in the episode; both directions are exact because the provider
snapshots are deterministic. ``texp_extra`` counts drops the v1 bound would
not have made; ``texp_revival`` counts keeps the v1 bound would have dropped.
"""

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path
import time

import numpy as np
import torch

from hypatia_topology_provider_stub import HypatiaTopologyProvider
from leo_multiagent_env import MULTIAGENT_LOADS
from mappo_evaluation import GlobalDijkstraPolicy
from run_development_load_diagnostics import canonical, read_json, sha, write_json
import run_development_load_diagnostics as diagnostics
from run_infeasible_drop_probe import PurgeInfeasibleEnv
from run_srpf_probe import LeastSlackEnv
from run_packet_mgmt_family_probe import (
    TOPOLOGY,
    SCENARIO,
    FamilyWrapper,
    pair_stats,
    run_row,
    write_report,
)

WORKLOADS = list(range(880651, 880691))
LOADS = (12, 8)
AGING_PERIOD = 10

if "texp_infeasible" not in diagnostics.DROP_REASONS:
    diagnostics.DROP_REASONS = diagnostics.DROP_REASONS + ("texp_infeasible",)


class TimeExpandedSRPFEnv(PurgeInfeasibleEnv):
    """Time-expanded feasibility purge + SRPF ordering (v2-A)."""

    drop_reason_name = "texp_infeasible"
    reorder_queues = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.texp_extra = 0
        self.texp_revival = 0
        self._adj_cache = {}
        self._reach_cache = {}
        self._prov = cfg.env.topology_provider

    def _adjacent_at(self, t):
        """Directed edges usable during slot t (deterministic; cached)."""
        cached = self._adj_cache.get(t)
        if cached is not None:
            return cached
        if t <= self.slot or self._prov is None:
            adj = [(u, v) for (u, v), e in self.graph.items()
                   if u >= 1 and v >= 1 and e.available]
        else:
            graph = self._prov(t, self)
            adj = [(u, v) for (u, v) in graph]  # provider already filters available
        self._adj_cache[t] = adj
        return adj

    def _deliverable_set(self, dst, bound):
        """Bitmask: sats that can deliver dst by bound (hops during slots
        self.slot..bound); backward pass over the per-slot adjacency."""
        key = (dst, bound)
        cached = self._reach_cache.get(key)
        if cached is not None:
            return cached
        S = 1 << (dst - 1)
        for t in range(bound, self.slot - 1, -1):
            St = S
            for (u, v) in self._adjacent_at(t):
                if S >> (v - 1) & 1:
                    St |= 1 << (u - 1)
            S = St
        self._reach_cache[key] = S
        return S

    def _purge_infeasible_packets(self):
        if self.slot > self.cfg.episode_slots:
            return
        if not any(self.queues.values()):
            return
        self._refresh_graph()
        reverse = defaultdict(list)
        for (u, v), edge in self.graph.items():
            if u >= 1 and v >= 1 and edge.available:
                reverse[v].append(u)
        static_dist = {}
        for sat in range(1, self.n_agents + 1):
            for packet_id in self.queues[sat]:
                dst = self.packets[packet_id].dst
                if dst not in static_dist:
                    static_dist[dst] = self._bfs_distances(dst, reverse)
        for sat in range(1, self.n_agents + 1):
            if not self.queues[sat]:
                continue
            kept = deque()
            for packet_id in self.queues[sat]:
                packet = self.packets[packet_id]
                deadline = self.cfg.packet_class_deadlines[packet.traffic_class]
                bound = min(packet.created_slot + deadline - 1, self.cfg.episode_slots)
                remaining = bound - self.slot
                dist = static_dist[packet.dst].get(sat)
                static_alive = dist is not None and dist <= remaining + self.safety_margin_slots
                alive_mask = self._deliverable_set(packet.dst, bound)
                texp_alive = bool(alive_mask >> (sat - 1) & 1)
                if not texp_alive:
                    self._drop_packet(packet_id, self.drop_reason_name)
                    self.purged_count += 1
                    self.purged_by_class[packet.traffic_class] += 1
                    if static_alive:
                        self.texp_extra += 1
                else:
                    if not static_alive:
                        self.texp_revival += 1
                    kept.append(packet_id)
            def remaining_hops(pid):
                d = static_dist[self.packets[pid].dst].get(sat)
                d = d if d is not None else 10 ** 6
                return (d, pid)
            if self.reorder_queues:
                self.queues[sat] = deque(sorted(kept, key=remaining_hops))
            else:
                self.queues[sat] = kept


class SRPFDlTieEnv(LeastSlackEnv):
    """v1 purge + SRPF with deadline tie-break: equal remaining hops -> the
    tighter absolute deadline goes first (then packet id)."""

    def _purge_infeasible_packets(self):
        if self.slot > self.cfg.episode_slots:
            return
        if not any(self.queues.values()):
            return
        self._refresh_graph()
        reverse = defaultdict(list)
        for (u, v), edge in self.graph.items():
            if u >= 1 and v >= 1 and edge.available:
                reverse[v].append(u)
        distances = {}
        for sat in range(1, self.n_agents + 1):
            for packet_id in self.queues[sat]:
                dst = self.packets[packet_id].dst
                if dst not in distances:
                    distances[dst] = self._bfs_distances(dst, reverse)
        for sat in range(1, self.n_agents + 1):
            if not self.queues[sat]:
                continue
            kept = deque()
            for packet_id in self.queues[sat]:
                packet = self.packets[packet_id]
                deadline = self.cfg.packet_class_deadlines[packet.traffic_class]
                deadline_bound = packet.created_slot + deadline - 1
                horizon_bound = self.cfg.episode_slots
                effective_bound = min(deadline_bound, horizon_bound)
                remaining = effective_bound - self.slot
                dist = distances[packet.dst].get(sat)
                if dist is not None and dist > remaining + self.safety_margin_slots:
                    self._drop_packet(packet_id, self.drop_reason_name)
                    self.purged_count += 1
                    self.purged_by_class[packet.traffic_class] += 1
                else:
                    kept.append(packet_id)
            def order_key(pid):
                d = distances[self.packets[pid].dst].get(sat)
                d = d if d is not None else 10 ** 6
                b = min(self.packets[pid].created_slot
                        + self.cfg.packet_class_deadlines[self.packets[pid].traffic_class] - 1,
                        self.cfg.episode_slots)
                return (d, b, pid)
            self.queues[sat] = deque(sorted(kept, key=order_key))


class SRPFAgingEnv(LeastSlackEnv):
    """v1 purge + SRPF with anti-starvation aging: one hop of effective
    priority per AGING_PERIOD slots since creation (then deadline, then id)."""

    def _purge_infeasible_packets(self):
        if self.slot > self.cfg.episode_slots:
            return
        if not any(self.queues.values()):
            return
        self._refresh_graph()
        reverse = defaultdict(list)
        for (u, v), edge in self.graph.items():
            if u >= 1 and v >= 1 and edge.available:
                reverse[v].append(u)
        distances = {}
        for sat in range(1, self.n_agents + 1):
            for packet_id in self.queues[sat]:
                dst = self.packets[packet_id].dst
                if dst not in distances:
                    distances[dst] = self._bfs_distances(dst, reverse)
        for sat in range(1, self.n_agents + 1):
            if not self.queues[sat]:
                continue
            kept = deque()
            for packet_id in self.queues[sat]:
                packet = self.packets[packet_id]
                deadline = self.cfg.packet_class_deadlines[packet.traffic_class]
                deadline_bound = packet.created_slot + deadline - 1
                horizon_bound = self.cfg.episode_slots
                effective_bound = min(deadline_bound, horizon_bound)
                remaining = effective_bound - self.slot
                dist = distances[packet.dst].get(sat)
                if dist is not None and dist > remaining + self.safety_margin_slots:
                    self._drop_packet(packet_id, self.drop_reason_name)
                    self.purged_count += 1
                    self.purged_by_class[packet.traffic_class] += 1
                else:
                    kept.append(packet_id)
            def order_key(pid):
                packet = self.packets[pid]
                d = distances[packet.dst].get(sat)
                d = d if d is not None else 10 ** 6
                b = min(packet.created_slot
                        + self.cfg.packet_class_deadlines[packet.traffic_class] - 1,
                        self.cfg.episode_slots)
                effective = max(0, d - (self.slot - packet.created_slot) // AGING_PERIOD)
                return (effective, b, pid)
            self.queues[sat] = deque(sorted(kept, key=order_key))


ARMS = {
    "base": {"cls": None, "kwargs": {}, "family": "baseline"},
    "purge_srpf": {"cls": LeastSlackEnv, "kwargs": {}, "family": "ours-v1"},
    "purge_srpf_dl": {"cls": SRPFDlTieEnv, "kwargs": {}, "family": "ours-v2"},
    "purge_srpf_aging": {"cls": SRPFAgingEnv, "kwargs": {}, "family": "ours-v2"},
    "purge_texp": {"cls": TimeExpandedSRPFEnv, "kwargs": {}, "family": "ours-v2"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "main"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    FamilyWrapper._provider = HypatiaTopologyProvider.from_csv(TOPOLOGY)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    workloads = WORKLOADS[:3] if args.stage == "smoke" else WORKLOADS
    loads = (12,) if args.stage == "smoke" else LOADS
    started = time.monotonic()
    rows = []
    pairing = {}
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for load in loads:
            for w in workloads:
                for arm_name, spec in ARMS.items():
                    row = run_row(GlobalDijkstraPolicy(), f"dijkstra::{arm_name}", w, load, spec)
                    events = (row["arrival_sha256"], row["physical_sha256"])
                    key = (load, w)
                    if key in pairing:
                        if pairing[key] != events:
                            raise AssertionError(f"pairing changed at {key}")
                    else:
                        pairing[key] = events
                    stream.write(canonical(row) + "\n")
                    rows.append(row)
                stream.flush()
            print(f"{len(rows)} load={load} done elapsed={time.monotonic()-started:.0f}s", flush=True)
    if args.stage == "smoke":
        texp_rows = [r for r in rows if r["arm"] == "purge_texp"]
        write_json(output / "smoke.json", {
            "episodes": len(rows),
            "texp_mean_purged": float(np.mean([r["purged_infeasible"] for r in texp_rows])),
            "v1_mean_purged": float(np.mean([r["purged_infeasible"] for r in rows
                                             if r["arm"] == "purge_srpf"])),
            "mean_delivery_by_arm": {a: float(np.mean([r["delivery_ratio"] for r in rows
                                                       if r["arm"] == a])) for a in ARMS},
            "elapsed_seconds": time.monotonic() - started,
        })
        print(canonical(read_json(output / "smoke.json")), flush=True)
        return
    write_json(output / "manifest.json", {
        "role": "packet_mgmt_v2_probe_dev",
        "runner_sha256": sha(Path(__file__)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_note": ("development-tier v2 refinements: deadline tie-break (C), "
                          "anti-starvation aging (B), time-expanded purge criterion (A); "
                          "routing fixed to GlobalDijkstra; fresh workloads 880651-880690"),
        "topology": str(TOPOLOGY), "topology_sha256": sha(TOPOLOGY),
        "workloads": workloads, "loads": list(loads),
        "aging_period": AGING_PERIOD,
        "arms": {k: {"family": v["family"]} for k, v in ARMS.items()},
        "training": False, "sealed_test_access": False,
        "new_drop_reasons": ["texp_infeasible"],
    })
    base_rows = [r for r in rows if r["arm"] == "base"]
    v1_rows = [r for r in rows if r["arm"] == "purge_srpf"]
    per_load_stats = {}
    for load in loads:
        stats = []
        for arm_name in ARMS:
            ref = base_rows if arm_name == "purge_srpf" else v1_rows
            stats.append(pair_stats(rows, load, arm_name, [r for r in ref if r["load"] == load]))
        per_load_stats[load] = stats
    write_json(output / "v2_stats.json", per_load_stats)
    write_report(output, list(loads), ARMS, per_load_stats, {"aging_period": AGING_PERIOD}, "v2-probe")
    texp_rows = [r for r in rows if r["arm"] == "purge_texp"]
    with (output / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
        fields = ["load", "arm", "mean_delivery", "delta_pp_vs_base", "one_sided_95_lower",
                  "sign_p", "mean_purged"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for load in loads:
            for stat in per_load_stats[load]:
                writer.writerow({
                    "load": load, "arm": stat["arm"],
                    "mean_delivery": f"{stat['mean_delivery']:.6f}",
                    "delta_pp_vs_base": f"{stat['delta_pp_vs_base']:.4f}",
                    "one_sided_95_lower": (stat.get("bootstrap_vs_base", {}).get("one_sided_95_lower")
                                           if "bootstrap_vs_base" in stat else ""),
                    "sign_p": (stat.get("sign_test", {}).get("exact_two_sided_p")
                               if "sign_test" in stat else ""),
                    "mean_purged": f"{stat['mean_purged']:.2f}",
                })
    write_json(output / "completion.json", {
        "status": "complete", "episodes": len(rows),
        "pairings_verified": len(pairing),
        "texp_mean_purged": float(np.mean([r["purged_infeasible"] for r in texp_rows])),
        "v1_mean_purged": float(np.mean([r["purged_infeasible"] for r in v1_rows])),
        "elapsed_seconds": time.monotonic() - started,
        "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    })
    print("v2 probe complete", flush=True)


if __name__ == "__main__":
    main()
