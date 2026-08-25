"""Stale-link-state Dijkstra baseline — the REALISTIC centralized incumbent.

GlobalDijkstraPolicy sees the current instantaneous topology every slot (an
idealized oracle: zero-latency global link-state). A real centralized routing
system (ground-computed SPF, or OSPF with slow flooding) routes on a topology
snapshot that is K slots old. This baseline recomputes the delay-weighted
distance tables every K slots and forwards on the stale view in between; the
env's action mask still blocks currently-down links, so stale paths hit
unavailable next-hops and fall back to the best FEASIBLE action under stale
distances — exactly what a deployed system does.

K=1 ~ the fresh oracle (sanity check against global_dijkstra).

Mask parity: runs under the same env variant as the no_lifetime checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from mappo_evaluation import EpisodeMetrics, evaluate_policy, load_checkpoint_policy

VARIANT = "no_lifetime"


class StaleDijkstraPolicy:
    """Delay-weighted SPF on a topology snapshot refreshed every K slots."""

    def __init__(self, staleness: int = 5):
        self.K = max(1, staleness)
        self.wrapper: Optional[CleanMARLLeoMultiAgentWrapper] = None
        self._dist: Dict[int, Dict[int, float]] = {}   # dst -> {node: dist}
        self._last_refresh = -10**9

    def bind(self, wrapper: CleanMARLLeoMultiAgentWrapper) -> None:
        self.wrapper = wrapper
        self._dist = {}
        self._last_refresh = -10**9

    def _snapshot_distances(self) -> Dict[int, Dict[int, float]]:
        """Reverse-Dijkstra dist(node -> dst) for every dst on the CURRENT
        (possibly stale) topology. Called once per K slots."""
        graph = self.wrapper.env.graph
        nodes = sorted({u for (u, _) in graph} | {v for (_, v) in graph})
        out: Dict[int, Dict[int, float]] = {}
        for dst in nodes:
            dist = {dst: 0.0}
            queue = [(0.0, dst)]
            while queue:
                d, node = heapq.heappop(queue)
                if d != dist.get(node):
                    continue
                for (src, nbr), edge in graph.items():
                    if nbr != node or not edge.available:
                        continue
                    cand = d + edge.delay_ms
                    if cand < dist.get(src, float("inf")):
                        dist[src] = cand
                        heapq.heappush(queue, (cand, src))
            out[dst] = dist
        return out

    def __call__(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        assert self.wrapper is not None
        slot = self.wrapper.env.slot
        if slot - self._last_refresh >= self.K:
            self._dist = self._snapshot_distances()
            self._last_refresh = slot
        actions = np.zeros(self.wrapper.n_agents, dtype=np.int64)
        for external_index, internal_sat in enumerate(self.wrapper.external_to_internal):
            feasible = np.flatnonzero(mask[external_index] > 0.5)
            if len(feasible) <= 1:
                actions[external_index] = int(feasible[0]) if len(feasible) else 0
                continue
            obs = self.wrapper._obs[internal_sat - 1]
            packet = self.wrapper.env.packets[obs["hol_packet_id"]]
            stale = self._dist.get(packet.dst, {})
            best, best_score = 0, float("inf")
            for action in feasible:
                if action == 0:
                    continue
                neighbor = obs["neighbor_ids"][int(action) - 1]
                edge = self.wrapper.env.graph[(internal_sat, neighbor)]
                score = edge.delay_ms + stale.get(neighbor, float("inf"))
                if score < best_score:
                    best, best_score = int(action), score
            actions[external_index] = best if best_score < float("inf") else int(feasible[0])
        return actions


def bootstrap_ci(values, seed=4001):
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(5000)]
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=Path("experiments/train-main"))
    ap.add_argument("--output", type=Path, default=Path("experiments/stale-spf"))
    ap.add_argument("--scenarios", default="medium_load,frequent_break,fault_links")
    ap.add_argument("--staleness", default="1,3,5,10")
    ap.add_argument("--checkpoint-seeds", default="7,42,1024,123,456,789,2024,314",
                    help="MAPPO seeds to pair against (from EVALFIX, same wseeds)")
    ap.add_argument("--workload-seed-start", type=int, default=11001)
    ap.add_argument("--workload-seeds", type=int, default=50)
    args = ap.parse_args()

    wseeds = range(args.workload_seed_start, args.workload_seed_start + args.workload_seeds)
    ks = [int(k) for k in args.staleness.split(",")]
    all_rows = []
    for scenario in args.scenarios.split(","):
        print(f"\n=== {scenario} ===", flush=True)
        for k in ks:
            pol_name = "dijkstra_stale1" if k == 1 else f"dijkstra_stale{k}"
            t0 = time.time()
            rows = evaluate_policy(scenario, pol_name, StaleDijkstraPolicy(staleness=k),
                                   -1, wseeds, variant=VARIANT)
            all_rows.extend(rows)
            d = float(np.mean([r.delivery_ratio for r in rows]))
            print(f"  K={k:2d}: delivery={d:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "stale_matrix.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
        w.writeheader(); w.writerows([asdict(r) for r in all_rows])

    # aggregate + paired vs MAPPO (same wseeds as EVALFIX headline)
    evalfix = list(csv.DictReader(
        open("experiments/eval-main/episode_metrics.csv", encoding="utf-8-sig")))
    agg, paired = [], []
    for scenario in args.scenarios.split(","):
        mm = {}
        for r in evalfix:
            if r["scenario"] == scenario and r["policy"] == "mappo":
                mm.setdefault(int(r["workload_seed"]), []).append(float(r["delivery_ratio"]))
        for k in ks:
            pol_name = "dijkstra_stale1" if k == 1 else f"dijkstra_stale{k}"
            sel = [r for r in all_rows if r.scenario == scenario and r.policy == pol_name]
            mean, lo, hi = bootstrap_ci([r.delivery_ratio for r in sel])
            agg.append({"scenario": scenario, "policy": pol_name, "staleness": k,
                        "metric": "delivery_ratio", "n": len(sel), "mean": mean,
                        "ci95_low": lo, "ci95_high": hi})
            sm = {}
            for r in sel:
                sm.setdefault(r.workload_seed, []).append(float(r.delivery_ratio))
            common = sorted(set(sm) & set(mm))
            if common:
                a = np.array([np.mean(mm[x]) for x in common])
                b = np.array([np.mean(sm[x]) for x in common])
                from scipy import stats as st
                try:
                    p = float(st.wilcoxon(a - b).pvalue)
                except ValueError:
                    p = 1.0
                paired.append({"scenario": scenario, "policy": pol_name,
                               "mappo_minus_stale": float((a - b).mean()),
                               "wilcoxon_p": p})
    with open(args.output / "aggregate_stale.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader(); w.writerows(agg)
    with open(args.output / "paired_stale.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        w.writeheader(); w.writerows(paired)
    (args.output / "manifest.json").write_text(json.dumps({
        "experiment": "stale_link_state_dijkstra",
        "staleness_slots": ks, "scenarios": args.scenarios.split(","),
        "workload_seeds": [args.workload_seed_start, args.workload_seeds],
        "variant": VARIANT,
    }, indent=2), encoding="utf-8")
    print(f"\n=> wrote {args.output}/stale_matrix.csv + aggregate_stale.csv + paired_stale.csv")


if __name__ == "__main__":
    main()
