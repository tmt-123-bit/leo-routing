"""Backpressure baseline for the 66-sat environment: gamma sweep on a small
development panel, then base and +stack arms for the best gamma on a fresh
20-workload panel. Includes plain Dijkstra on the same panel as the anchor."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from backpressure_baseline import BackpressurePolicy
from hypatia_topology_provider_stub import HypatiaTopologyProvider
from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MultiAgentConfig
from mappo_evaluation import GlobalDijkstraPolicy, evaluate_policy_with_constraint_metrics
from run_development_load_diagnostics import DiagnosticWrapper, canonical, sha, write_json
from run_srpf_probe import LeastSlackEnv

TOPOLOGY = Path("../data/starlink_66_links.csv")
SCENARIO = "hotspot_high_load"


class W66(DiagnosticWrapper):
    _provider = None

    def __init__(self, workload_seed, load, arm):
        cfg = MultiAgentConfig(
            env=EnvConfig(seed=workload_seed, scenario=SCENARIOS[SCENARIO],
                          topology_provider=W66._provider, n_planes=11,
                          sats_per_plane=6),
            initial_packets=load, exogenous_packets_per_slot=load,
            seed=workload_seed, variant="qos_only",
        )
        super().__init__(scenario=SCENARIO, cfg=cfg, seed=workload_seed)
        if arm != "base":
            self.env = LeastSlackEnv(self.env.cfg)


def run_row(policy, label, seed, workload, load, arm):
    wrapper = W66(workload, load, arm)
    result = evaluate_policy_with_constraint_metrics(
        SCENARIO, label, policy, seed, [workload],
        wrapper_factory=lambda _: wrapper, variant="qos_only",
    )[0]
    return asdict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    W66._provider = HypatiaTopologyProvider.from_csv(TOPOLOGY)
    sweep_wl = list(range(880201, 880206))
    main_wl = list(range(880206, 880226))
    write_json(output / "manifest.json", {
        "role": "backpressure_baseline_dev",
        "runner_sha256": sha(Path(__file__)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gamma_sweep_workloads": sweep_wl, "main_workloads": main_wl,
        "note": ("EDR-BP style max-weight with shortest-path bias; gamma tuned "
                 "on the sweep panel, best gamma evaluated on the main panel"),
        "training": False, "sealed_test_access": False,
    })
    torch.set_num_threads(1)
    rows = []
    started = time.monotonic()
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        best_gamma, best_mean = None, None
        for gamma in (0.0, 1.0, 2.0, 4.0, 8.0):
            vals = [run_row(BackpressurePolicy(gamma), f"bp_g{gamma}", -1, w, 12, "base")["delivery_ratio"]
                    for w in sweep_wl]
            mean = float(np.mean(vals))
            print("gamma=%.1f sweep mean=%.4f" % (gamma, mean), flush=True)
            if best_mean is None or mean > best_mean:
                best_gamma, best_mean = gamma, mean
        for arm, label in (("base", "bp_base"), ("purge_srpf", "bp_mech")):
            for w in main_wl:
                row = run_row(BackpressurePolicy(best_gamma), label, -1, w, 12, arm)
                row["arm"] = arm
                stream.write(canonical(row) + "\n")
                rows.append(row)
            stream.flush()
        for arm, label in (("base", "dijkstra_base"), ("purge_srpf", "dijkstra_mech")):
            for w in main_wl:
                row = run_row(GlobalDijkstraPolicy(), label, -1, w, 12, arm)
                row["arm"] = arm
                stream.write(canonical(row) + "\n")
                rows.append(row)
            stream.flush()
    summary = {}
    for r in rows:
        stem = r["policy"].rsplit("_", 1)[0]
        kind = "mech" if r["policy"].endswith("_mech") else "base"
        summary.setdefault((stem, kind), []).append(r["delivery_ratio"])
    agg = {"best_gamma": best_gamma,
           "gamma_sweep_mean": best_mean}
    for (stem, kind), v in summary.items():
        agg[f"{stem}/{kind}"] = float(np.mean(v))
    write_json(output / "summary.json", agg)
    write_json(output / "completion.json", {"status": "complete", "episodes": len(rows),
               "elapsed_seconds": time.monotonic() - started,
               "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps(agg, indent=1), flush=True)


if __name__ == "__main__":
    main()
