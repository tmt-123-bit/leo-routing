"""1008-satellite (72x14) scale calibration: find the saturation knee and
per-episode runtime before sizing any panel.

Uses a caching subclass of GlobalDijkstraPolicy that is bit-identical in
decisions (per-slot adjacency + one reverse Dijkstra per (slot, destination),
reusing the same delay weights and available-edge set as the original
per-candidate forward search) but computes each destination's shortest-path
tree once per slot instead of once per candidate edge.

Load sweep over {8, 12, 16, 24, 32} on fresh workloads 880801-880802,
scenario hotspot_high_load, Dijkstra base arm only. Output: per-(load,
workload) delivery ratio and wall-clock seconds per episode.
"""

import argparse
import json
import time
from pathlib import Path

import heapq
import numpy as np
import torch

from hypatia_topology_provider_stub import HypatiaTopologyProvider
from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MultiAgentConfig
from mappo_evaluation import GlobalDijkstraPolicy, evaluate_policy_with_constraint_metrics
from run_development_load_diagnostics import DiagnosticWrapper, canonical, sha, write_json

TOPOLOGY = Path("../data/starlink_1008_links.csv")
PLANES = 72
PER_PLANE = 14
SCENARIO = "hotspot_high_load"
LOADS = (8, 12, 16, 24, 32)
WORKLOADS = (880801, 880802)


class CachedGlobalDijkstraPolicy(GlobalDijkstraPolicy):
    """Decision-identical cache over GlobalDijkstraPolicy._distance."""

    def __init__(self):
        super().__init__()
        self._slot = None
        self._radj = None
        self._dist_cache = {}

    def _slot_reverse_adjacency(self):
        env = self.wrapper.env
        if self._slot == env.slot and self._radj is not None:
            return self._radj
        radj = {}
        for (u, v), edge in env.graph.items():
            if u >= 1 and v >= 1 and edge.available:
                radj.setdefault(v, []).append((u, edge.delay_ms))
        self._radj, self._slot, self._dist_cache = radj, env.slot, {}
        return radj

    def _distance(self, source: int, destination: int) -> float:
        radj = self._slot_reverse_adjacency()
        cached = self._dist_cache.get(destination)
        if cached is None:
            dist = {destination: 0.0}
            queue = [(0.0, destination)]
            while queue:
                d, node = heapq.heappop(queue)
                if d != dist.get(node):
                    continue
                for prev, delay in radj.get(node, ()):
                    cand = d + delay
                    if cand < dist.get(prev, float("inf")):
                        dist[prev] = cand
                        heapq.heappush(queue, (cand, prev))
            cached = dist
            self._dist_cache[destination] = cached
        return cached.get(source, float("inf"))


class Wrapper1008(DiagnosticWrapper):
    _provider = None

    def __init__(self, workload_seed, load):
        cfg = MultiAgentConfig(
            env=EnvConfig(seed=workload_seed, scenario=SCENARIOS[SCENARIO],
                          topology_provider=Wrapper1008._provider,
                          n_planes=PLANES, sats_per_plane=PER_PLANE),
            initial_packets=load, exogenous_packets_per_slot=load,
            seed=workload_seed, variant="qos_only",
        )
        super().__init__(scenario=SCENARIO, cfg=cfg, seed=workload_seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    Wrapper1008._provider = HypatiaTopologyProvider.from_csv(TOPOLOGY)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    started = time.monotonic()
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for load in LOADS:
            for w in WORKLOADS:
                t0 = time.monotonic()
                wrapper = Wrapper1008(w, load)
                result = evaluate_policy_with_constraint_metrics(
                    SCENARIO, "dijkstra_base", CachedGlobalDijkstraPolicy(), -1, [w],
                    wrapper_factory=lambda _: wrapper, variant="qos_only",
                )[0]
                elapsed = time.monotonic() - t0
                row = {
                    "load": load, "workload": w,
                    "delivery_ratio": float(result.delivery_ratio),
                    "dropped": int(result.dropped),
                    "ep_seconds": round(elapsed, 1),
                }
                stream.write(canonical(row) + "\n")
                stream.flush()
                rows.append(row)
                print(f"load={load} w={w} delivery={row['delivery_ratio']:.4f} "
                      f"ep={elapsed:.0f}s total={time.monotonic()-started:.0f}s", flush=True)
    summary = {}
    for load in LOADS:
        vals = [r["delivery_ratio"] for r in rows if r["load"] == load]
        secs = [r["ep_seconds"] for r in rows if r["load"] == load]
        summary[str(load)] = {"mean_delivery": float(np.mean(vals)),
                              "mean_ep_seconds": float(np.mean(secs))}
    write_json(output / "calibration.json", {
        "topology": str(TOPOLOGY), "topology_sha256": sha(TOPOLOGY),
        "planes": PLANES, "per_plane": PER_PLANE,
        "workloads": list(WORKLOADS), "loads": list(LOADS),
        "summary": summary, "episodes": rows,
        "training": False, "sealed_test_access": False,
    })
    print(canonical(summary), flush=True)


if __name__ == "__main__":
    main()
