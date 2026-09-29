"""Two targeted optimization probes at the 1008-satellite scale, both
hypothesis-driven from the v2 ablation and the full family matrix.

Stage ``hybrid``: purge + LCFS ordering. The full family matrix showed LCFS
is the best non-ours arm at 1008 stars (+4.0%: at extreme path lengths,
packet age is a first-order proxy for remaining deadline budget) while the
champion partner SRPF gained only +2.1%. Test whether feasibility purge +
recency ordering beats purge + SRPF at this scale. Runs on the same fresh
panel as the family-full study (880841-880860) so rows pair directly.

Stage ``timescale``: causal test of the geometric-boundary explanation.
Rescale time only - episode 60 slots, class deadlines x2 (60/24/40) - on
fresh workloads 880861-880870, arms base and purge_srpf. Prediction from
the three-regime law: with path-feasible deadline budgets restored, the
purge gain band returns toward the 66/156-star levels instead of the
+2% collapse observed at 30-slot horizons.
"""

import argparse
from collections import deque, defaultdict
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from hypatia_topology_provider_stub import HypatiaTopologyProvider
from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MultiAgentConfig
from run_development_load_diagnostics import (
    DiagnosticWrapper,
    canonical,
    packet_accounting,
    sha,
    write_json,
)
import run_development_load_diagnostics as diagnostics
from mappo_evaluation import evaluate_policy_with_constraint_metrics
from run_infeasible_drop_probe import PurgeInfeasibleEnv
from run_srpf_probe import LeastSlackEnv
from run_tle1008_calibration import CachedGlobalDijkstraPolicy

TOPOLOGY = Path("../data/starlink_1008_links.csv")
PLANES = 72
PER_PLANE = 14
SCENARIO = "hotspot_high_load"
LOAD = 12
HYBRID_WORKLOADS = tuple(range(880841, 880861))
TIMESCALE_WORKLOADS = tuple(range(880861, 880871))


class PurgeLCFSEnv(PurgeInfeasibleEnv):
    """Feasibility purge, then newest-first ordering (LCFS)."""

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
                effective_bound = min(deadline_bound, self.cfg.episode_slots)
                remaining = effective_bound - self.slot
                dist = distances[packet.dst].get(sat)
                if dist is not None and dist > remaining + self.safety_margin_slots:
                    self._drop_packet(packet_id, self.drop_reason_name)
                    self.purged_count += 1
                    self.purged_by_class[packet.traffic_class] += 1
                else:
                    kept.append(packet_id)
            self.queues[sat] = deque(sorted(
                kept, key=lambda pid: (-self.packets[pid].created_slot, -pid)))


class Wrapper1008(DiagnosticWrapper):
    _provider = None
    episode_slots = 30
    deadlines = (30, 12, 20)

    def __init__(self, workload_seed, arm_cls):
        cfg = MultiAgentConfig(
            env=EnvConfig(seed=workload_seed, scenario=SCENARIOS[SCENARIO],
                          topology_provider=Wrapper1008._provider,
                          n_planes=PLANES, sats_per_plane=PER_PLANE),
            initial_packets=LOAD, exogenous_packets_per_slot=LOAD,
            seed=workload_seed, variant="qos_only",
            episode_slots=self.episode_slots,
            packet_class_deadlines=self.deadlines,
        )
        super().__init__(scenario=SCENARIO, cfg=cfg, seed=workload_seed)
        if arm_cls is not None:
            self.env = arm_cls(self.env.cfg)


def run_row(policy, label, workload, arm_cls):
    wrapper = Wrapper1008(workload, arm_cls)
    result = evaluate_policy_with_constraint_metrics(
        SCENARIO, label, policy, -1, [workload],
        wrapper_factory=lambda _: wrapper, variant="qos_only",
    )[0]
    row = asdict(result)
    extra, _packets = packet_accounting(wrapper.env)
    row.update(extra)
    row.update(arm=label.split("::")[-1],
               managed=int(getattr(wrapper.env, "managed_count", 0)),
               purged_infeasible=int(getattr(wrapper.env, "purged_count", 0)),
               physical_sha256=wrapper.physical_digest.hexdigest())
    if sum(row[f"drop_{r}"] for r in diagnostics.DROP_REASONS) != row["dropped"]:
        raise AssertionError("drop partition mismatch")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("hybrid", "timescale", "timescale2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    Wrapper1008._provider = HypatiaTopologyProvider.from_csv(TOPOLOGY)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    rows = []
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        if args.stage == "hybrid":
            arms = {"purge_lcfs": PurgeLCFSEnv}
            workloads = HYBRID_WORKLOADS
        elif args.stage == "timescale":
            Wrapper1008.episode_slots = 60
            Wrapper1008.deadlines = (60, 24, 40)
            arms = {"base_60": None, "purge_srpf_60": LeastSlackEnv}
            workloads = TIMESCALE_WORKLOADS
        else:
            # matched-generated causal test: 60 slots x load 6 = same total
            # generated as 30 slots x load 12, with doubled time budgets.
            global LOAD
            LOAD = 6
            Wrapper1008.episode_slots = 60
            Wrapper1008.deadlines = (60, 24, 40)
            arms = {"base_60ld6": None, "purge_srpf_60ld6": LeastSlackEnv,
                    "purge_lcfs_60ld6": PurgeLCFSEnv}
            workloads = TIMESCALE_WORKLOADS
        for w in workloads:
            for arm_name, arm_cls in arms.items():
                row = run_row(CachedGlobalDijkstraPolicy(), f"dijkstra::{arm_name}", w, arm_cls)
                stream.write(canonical(row) + "\n")
                rows.append(row)
            stream.flush()
            print(f"{len(rows)} w={w} elapsed={time.monotonic()-started:.0f}s", flush=True)
    summary = {}
    for arm_name in arms:
        delivery = np.array([r["delivery_ratio"] for r in rows if r["arm"] == arm_name])
        summary[arm_name] = {
            "mean_delivery": float(delivery.mean()),
            "mean_purged": float(np.mean([r["purged_infeasible"] for r in rows
                                          if r["arm"] == arm_name])),
        }
    if args.stage == "timescale":
        gains = (summary["purge_srpf_60"]["mean_delivery"]
                 / summary["base_60"]["mean_delivery"] - 1.0)
        summary["purge_relative_gain"] = float(gains)
    write_json(output / "result.json", {
        "stage": args.stage,
        "workloads": list(workloads),
        "episode_slots": Wrapper1008.episode_slots,
        "deadlines": list(Wrapper1008.deadlines),
        "summary": summary,
        "episodes": len(rows),
        "training": False, "sealed_test_access": False,
    })
    print(canonical(read_json(output / "result.json")), flush=True)


if __name__ == "__main__":
    main()
