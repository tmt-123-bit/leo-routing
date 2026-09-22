"""156-sat scale point: BP gamma mini-sweep, then Dijkstra and backpressure
base vs +stack on a fresh ten-workload panel at the calibrated knee (load 8)."""

import argparse
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

TOPOLOGY = Path("../data/starlink_156_links.csv")
SCENARIO = "hotspot_high_load"
LOAD = 24
SWEEP_WL = [880266, 880267, 880268]
MAIN_WL = list(range(880271, 880281))


class W156(DiagnosticWrapper):
    _provider = None

    def __init__(self, workload_seed, arm):
        cfg = MultiAgentConfig(
            env=EnvConfig(seed=workload_seed, scenario=SCENARIOS[SCENARIO],
                          topology_provider=W156._provider, n_planes=26,
                          sats_per_plane=6),
            initial_packets=LOAD, exogenous_packets_per_slot=LOAD,
            seed=workload_seed, variant="qos_only",
        )
        super().__init__(scenario=SCENARIO, cfg=cfg, seed=workload_seed)
        if arm != "base":
            self.env = LeastSlackEnv(self.env.cfg)


def run_row(policy, label, workload, arm):
    wrapper = W156(workload, arm)
    result = evaluate_policy_with_constraint_metrics(
        SCENARIO, label, policy, -1, [workload],
        wrapper_factory=lambda _: wrapper, variant="qos_only",
    )[0]
    return asdict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    W156._provider = HypatiaTopologyProvider.from_csv(TOPOLOGY)
    write_json(output / "manifest.json", {
        "role": "tle156_scale_point_dev",
        "runner_sha256": sha(Path(__file__)),
        "topology_sha256": sha(TOPOLOGY),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "load": LOAD, "sweep_workloads": SWEEP_WL, "main_workloads": MAIN_WL,
        "training": False, "sealed_test_access": False,
    })
    torch.set_num_threads(1)
    rows = []
    started = time.monotonic()
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        best_gamma, best_mean = None, None
        for gamma in (2.0, 4.0, 8.0):
            vals = [run_row(BackpressurePolicy(gamma), f"bp_g{gamma}", w, "base")["delivery_ratio"]
                    for w in SWEEP_WL]
            mean = float(np.mean(vals))
            print("gamma=%.1f mean=%.4f" % (gamma, mean), flush=True)
            if best_mean is None or mean > best_mean:
                best_gamma, best_mean = gamma, mean
        for policy, name in ((GlobalDijkstraPolicy(), "dijkstra"),
                             (BackpressurePolicy(best_gamma), "bp")):
            for arm, suffix in (("base", "base"), ("purge_srpf", "mech")):
                for w in MAIN_WL:
                    row = run_row(policy, f"{name}_{suffix}", w, arm)
                    row["arm"] = arm
                    stream.write(canonical(row) + "\n")
                    rows.append(row)
                stream.flush()
                vals = [r["delivery_ratio"] for r in rows if r["policy"] == f"{name}_{suffix}"]
                print(f"{name}_{suffix}: mean={np.mean(vals):.4f} elapsed={time.monotonic()-started:.0f}s", flush=True)
    summary = {"best_gamma": best_gamma}
    for name in ("dijkstra", "bp"):
        for suffix in ("base", "mech"):
            vals = [r["delivery_ratio"] for r in rows if r["policy"] == f"{name}_{suffix}"]
            summary[f"{name}/{suffix}"] = float(np.mean(vals))
    summary["gain_dijkstra"] = summary["dijkstra/mech"] / summary["dijkstra/base"] - 1
    summary["gain_bp"] = summary["bp/mech"] / summary["bp/base"] - 1
    write_json(output / "summary.json", summary)
    write_json(output / "completion.json", {"status": "complete", "episodes": len(rows),
               "elapsed_seconds": time.monotonic() - started,
               "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print(canonical(summary), flush=True)


if __name__ == "__main__":
    main()
