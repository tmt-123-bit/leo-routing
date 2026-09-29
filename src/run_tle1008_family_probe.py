"""1008-satellite development family probe: does the champion packet-
management stack (feasibility purge + SRPF) hold its lead at the fourth
scale point of the dual-factor curve?

Arms (fixed CachedGlobalDijkstraPolicy routing, scenario hotspot_high_load):
  base           FIFO (env default)
  purge_srpf     v1 champion stack (static-BFS feasibility purge + SRPF)
  class_priority strict traffic-class priority (strongest alternative arm
                 of the 66-star family study)
  codel          sojourn-based HOL drop at the 66-star-tuned target (12)

Fresh workloads 880811-880830 (20), load 12 (the 1008-star calibration's
comparability point, base 0.242 mirroring 66-star load 12 base 0.272).
Development tier: screening signal only; a preregistered gate->formal study
on untouched workload ranges follows if the signal holds.
"""

import argparse
from collections import deque
from dataclasses import asdict
from pathlib import Path
import heapq
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
    read_json,
    sha,
    write_json,
)
import run_development_load_diagnostics as diagnostics
from mappo_evaluation import evaluate_policy_with_constraint_metrics
from run_infeasible_drop_probe import PurgeInfeasibleEnv
from run_srpf_probe import LeastSlackEnv
from run_packet_mgmt_family_probe import CoDelEnv, ClassPriorityEnv
from run_tle1008_calibration import CachedGlobalDijkstraPolicy

TOPOLOGY = Path("../data/starlink_1008_links.csv")
PLANES = 72
PER_PLANE = 14
SCENARIO = "hotspot_high_load"
LOAD = 12
WORKLOADS = tuple(range(880811, 880831))

if "texp_infeasible" not in diagnostics.DROP_REASONS:
    pass  # no texp arm here; guard kept for symmetry with earlier probes


class Wrapper1008(DiagnosticWrapper):
    _provider = None

    def __init__(self, workload_seed, arm_spec):
        cfg = MultiAgentConfig(
            env=EnvConfig(seed=workload_seed, scenario=SCENARIOS[SCENARIO],
                          topology_provider=Wrapper1008._provider,
                          n_planes=PLANES, sats_per_plane=PER_PLANE),
            initial_packets=LOAD, exogenous_packets_per_slot=LOAD,
            seed=workload_seed, variant="qos_only",
        )
        super().__init__(scenario=SCENARIO, cfg=cfg, seed=workload_seed)
        if arm_spec["cls"] is not None:
            self.env = arm_spec["cls"](self.env.cfg, **arm_spec["kwargs"])


ARMS = {
    "base": {"cls": None, "kwargs": {}, "family": "baseline"},
    "purge_srpf": {"cls": LeastSlackEnv, "kwargs": {}, "family": "ours"},
    "class_priority": {"cls": ClassPriorityEnv, "kwargs": {}, "family": "scheduling"},
    "codel": {"cls": CoDelEnv, "kwargs": {"sojourn_target": 12}, "family": "dropping"},
}


def run_row(policy, label, workload, arm_spec):
    wrapper = Wrapper1008(workload, arm_spec)
    result = evaluate_policy_with_constraint_metrics(
        SCENARIO, label, policy, -1, [workload],
        wrapper_factory=lambda _: wrapper, variant="qos_only",
    )[0]
    row = asdict(result)
    extra, _packets = packet_accounting(wrapper.env)
    row.update(extra)
    row.update(arm=label.split("::")[-1], label=label,
               managed=int(getattr(wrapper.env, "managed_count", 0)),
               purged_infeasible=int(getattr(wrapper.env, "purged_count", 0)),
               physical_sha256=wrapper.physical_digest.hexdigest())
    if sum(row[f"drop_{r}"] for r in diagnostics.DROP_REASONS) != row["dropped"]:
        raise AssertionError("drop partition mismatch")
    return row


def bootstrap_ratio(base, mech, draws=5000, seed=20260926):
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
    started = time.monotonic()
    rows = []
    pairing = {}
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for w in WORKLOADS:
            for arm_name, spec in ARMS.items():
                row = run_row(CachedGlobalDijkstraPolicy(), f"dijkstra::{arm_name}", w, spec)
                events = (row["arrival_sha256"], row["physical_sha256"])
                if w in pairing:
                    if pairing[w] != events:
                        raise AssertionError(f"pairing changed at {w}")
                else:
                    pairing[w] = events
                stream.write(canonical(row) + "\n")
                rows.append(row)
            stream.flush()
            print(f"{len(rows)} w={w} elapsed={time.monotonic()-started:.0f}s", flush=True)
    base_rows = {r["workload_seed"]: r for r in rows if r["arm"] == "base"}
    stats = []
    for arm_name in ARMS:
        by_w = {r["workload_seed"]: r for r in rows if r["arm"] == arm_name}
        delivery = np.array([by_w[w]["delivery_ratio"] for w in WORKLOADS])
        base_delivery = np.array([base_rows[w]["delivery_ratio"] for w in WORKLOADS])
        gains = delivery - base_delivery
        stat = {
            "arm": arm_name,
            "mean_delivery": float(delivery.mean()),
            "delta_pp_vs_base": float(100 * gains.mean()),
            "positive_workloads": int((gains > 0).sum()),
            "mean_purged": float(np.mean([by_w[w]["purged_infeasible"] for w in WORKLOADS])),
        }
        if arm_name != "base":
            stat["bootstrap_vs_base"] = bootstrap_ratio(base_delivery, delivery)
        stats.append(stat)
    write_json(output / "probe_stats.json", {
        "topology": str(TOPOLOGY), "topology_sha256": sha(TOPOLOGY),
        "planes": PLANES, "per_plane": PER_PLANE, "load": LOAD,
        "workloads": list(WORKLOADS),
        "arms": {k: {"family": v["family"]} for k, v in ARMS.items()},
        "stats": stats,
        "training": False, "sealed_test_access": False,
    })
    lines = ["# 1008-star family probe (dev)", "",
             "| arm | family | mean delivery | Δ vs base (pp) | positive wls | one-sided lower |",
             "|---|---|---|---|---|---|"]
    for s in stats:
        lower = (s.get("bootstrap_vs_base", {}).get("one_sided_95_lower")
                 if "bootstrap_vs_base" in s else None)
        lines.append(f"| {s['arm']} | {ARMS[s['arm']]['family']} "
                     f"| {s['mean_delivery']:.4f} | {s['delta_pp_vs_base']:+.2f} "
                     f"| {s['positive_workloads']}/{len(WORKLOADS)} "
                     f"| {('' if lower is None else f'{lower:+.3f}')} |")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "completion.json", {
        "status": "complete", "episodes": len(rows),
        "pairings_verified": len(pairing),
        "elapsed_seconds": time.monotonic() - started,
        "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    })
    print(canonical(read_json(output / "probe_stats.json")), flush=True)


if __name__ == "__main__":
    main()
