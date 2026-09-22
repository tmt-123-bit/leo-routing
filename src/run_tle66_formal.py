"""Formal runner for the TLE-66 packet-management preregistered study
(docs/TLE66_PACKET_MANAGEMENT_FORMAL_V1.md).

Stage ``gate``: plain vs +stack Dijkstra on workloads 880131..880150 at load
12; the formal panel must not be touched unless the stack mean exceeds plain.

Stage ``formal``: one-shot grid on workloads 880151..880200 — loads 12
(primary) and 10 (secondary), arms dijkstra base/mech and ILPR-style
persistent base/mech. Frozen statistics: workload bootstrap of the ratio of
means (5,000 draws), paired per-workload gains, exact sign test, and the
predeclared joint success rule.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from hypatia_topology_provider_stub import HypatiaTopologyProvider
from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MultiAgentConfig
from mappo_evaluation import GlobalDijkstraPolicy, evaluate_policy_with_constraint_metrics
from run_development_load_diagnostics import (
    DiagnosticWrapper, PersistentDijkstraPolicy, canonical, read_json, sha, write_json,
)
from run_srpf_probe import LeastSlackEnv

TOPOLOGY = Path("../data/starlink_66_links.csv")
GATE_WORKLOADS = list(range(880131, 880151))
FORMAL_WORKLOADS = list(range(880151, 880201))
LOADS = {12: "primary", 10: "secondary"}
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
    row = asdict(result)
    row["load"] = load
    row["arm"] = arm
    return row


def bootstrap_ratio(base, mech, draws=5000, seed=20260924):
    rng = np.random.default_rng(seed)
    n = len(base)
    out = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        out.append(mech[idx].mean() / base[idx].mean() - 1.0)
    out = np.array(out)
    return {
        "ratio_of_means": float(mech.mean() / base.mean() - 1.0),
        "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
        "one_sided_95_lower": float(np.percentile(out, 5.0)),
    }


def sign_test(gains):
    positive = int((gains > 0).sum())
    n = len(gains)
    from math import comb
    tail = sum(comb(n, k) for k in range(0, min(positive, n - positive) + 1))
    p = float(min(1.0, 2.0 * tail / 2 ** n))
    return {"positive": positive, "n": n, "exact_two_sided_p": p}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gate", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.stage == "formal":
        if args.gate_dir is None:
            raise ValueError("formal stage requires --gate-dir")
        gate = read_json(args.gate_dir.resolve() / "completion.json")
        if gate.get("status") != "complete" or not gate.get("gate_pass"):
            raise ValueError("gate did not pass; formal stage not authorized")

    W66._provider = HypatiaTopologyProvider.from_csv(TOPOLOGY)
    workloads = GATE_WORKLOADS if args.stage == "gate" else FORMAL_WORKLOADS
    write_json(output / "manifest.json", {
        "role": f"tle66_formal_{args.stage}",
        "runner_sha256": sha(Path(__file__)),
        "protocol": "docs/TLE66_PACKET_MANAGEMENT_FORMAL_V1.md",
        "protocol_sha256": sha(Path("../docs/TLE66_PACKET_MANAGEMENT_FORMAL_V1.md")),
        "topology_sha256": sha(TOPOLOGY),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workloads": workloads,
        "loads": list(LOADS) if args.stage == "formal" else [12],
        "training": False, "sealed_test_access": False,
    })
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    started = time.monotonic()
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        if args.stage == "gate":
            for arm, label in (("base", "dijkstra_base"), ("purge_srpf", "dijkstra_mech")):
                for w in workloads:
                    row = run_row(GlobalDijkstraPolicy(), label, -1, w, 12, arm)
                    stream.write(canonical(row) + "\n")
                    rows.append(row)
                stream.flush()
        else:
            for load in (12, 10):
                for policy, name, seed in ((GlobalDijkstraPolicy(), "dijkstra", -1),
                                           (PersistentDijkstraPolicy(), "persistent", -1)):
                    for arm, suffix in (("base", "base"), ("purge_srpf", "mech")):
                        for w in workloads:
                            row = run_row(policy, f"{name}_{suffix}", seed, w, load, arm)
                            stream.write(canonical(row) + "\n")
                            rows.append(row)
                        stream.flush()
                        print(f"{len(rows)} rows {name}_{suffix} load={load} "
                              f"elapsed={time.monotonic()-started:.0f}s", flush=True)
    if sha(TOPOLOGY) != json.loads((output / "manifest.json").read_text(encoding="utf-8"))["topology_sha256"]:
        raise AssertionError("topology changed during the run")

    result = {"stage": args.stage, "episodes": len(rows)}
    if args.stage == "gate":
        base = np.array([r["delivery_ratio"] for r in rows if r["arm"] == "base"])
        mech = np.array([r["delivery_ratio"] for r in rows if r["arm"] != "base"])
        result.update({"base_mean": float(base.mean()), "mech_mean": float(mech.mean()),
                       "gate_pass": bool(mech.mean() > base.mean())})
    else:
        stats = {}
        for load, role in LOADS.items():
            entry = {}
            for name in ("dijkstra", "persistent"):
                b = np.array([r["delivery_ratio"] for r in rows
                              if r["load"] == load and r["policy"] == f"{name}_base"])
                m = np.array([r["delivery_ratio"] for r in rows
                              if r["load"] == load and r["policy"] == f"{name}_mech"])
                entry[name] = {
                    "base_mean": float(b.mean()), "mech_mean": float(m.mean()),
                    "bootstrap": bootstrap_ratio(b, m),
                    "paired_gain_positive": int((m > b).sum()), "n": len(b),
                    "sign_test": sign_test(m - b),
                }
            if role == "primary":
                prim = entry["dijkstra"]
                result["primary_pass"] = bool(
                    prim["bootstrap"]["one_sided_95_lower"] >= 0.10
                    and prim["paired_gain_positive"] >= 48
                )
            stats[f"load_{load}"] = entry
        result["scenarios"] = stats
    write_json(output / "result.json", result)
    write_json(output / "completion.json", {"status": "complete", "episodes": len(rows),
               "elapsed_seconds": time.monotonic() - started,
               "gate_pass": result.get("gate_pass"),
               "primary_pass": result.get("primary_pass"),
               "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print(canonical({k: result[k] for k in ("stage", "episodes", "gate_pass", "primary_pass") if k in result}), flush=True)


if __name__ == "__main__":
    main()
