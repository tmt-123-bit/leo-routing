"""Full 10-arm packet-management family comparison at the 1008-satellite
scale (72x14 TLE topology), mirroring the 66-star family study
(packet-mgmt-family-20260925-v1 / -formal-20260926-v1) arm-for-arm.

Arms: base, edf, class_priority, lcfs, codel, red, dropfront, purge,
purge_srpf, purge_edf. Dropping-family hyperparameters are frozen from the
66-star tune panel (codel sojourn target 12; RED min/max/p 12/24/0.1;
dropfront threshold 16) - tuned-best per family, disclosed; no re-tuning
here. Routing fixed to CachedGlobalDijkstraPolicy (decision-identical to
GlobalDijkstraPolicy). Fresh workloads 880841-880860 (20), load 12 (the
1008-star comparability point), one paired episode per (arm, workload).
Development tier.
"""

import argparse
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
from run_packet_mgmt_family_probe import (
    EDFEnv,
    ClassPriorityEnv,
    LCFSEnv,
    CoDelEnv,
    REDEnv,
    DropFrontEnv,
    PurgeEDFEnv,
    PurgeInfeasibleEnv,
    LeastSlackEnv,
    bootstrap_ratio,
    pair_stats,
    write_report,
)
from run_tle1008_calibration import CachedGlobalDijkstraPolicy

TOPOLOGY = Path("../data/starlink_1008_links.csv")
PLANES = 72
PER_PLANE = 14
SCENARIO = "hotspot_high_load"
LOAD = 12
WORKLOADS = tuple(range(880841, 880861))
TUNED = {"codel_sojourn_target": 12, "red_min_th": 12, "red_max_th": 24,
         "red_max_p": 0.1, "dropfront_threshold": 16}


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


def build_arms(tuned):
    return {
        "base": {"cls": None, "kwargs": {}, "family": "baseline"},
        "edf": {"cls": EDFEnv, "kwargs": {}, "family": "scheduling"},
        "class_priority": {"cls": ClassPriorityEnv, "kwargs": {}, "family": "scheduling"},
        "lcfs": {"cls": LCFSEnv, "kwargs": {}, "family": "scheduling"},
        "codel": {"cls": CoDelEnv,
                  "kwargs": {"sojourn_target": tuned["codel_sojourn_target"]},
                  "family": "dropping"},
        "red": {"cls": REDEnv,
                "kwargs": {"min_th": tuned["red_min_th"], "max_th": tuned["red_max_th"],
                           "max_p": tuned["red_max_p"]},
                "family": "dropping"},
        "dropfront": {"cls": DropFrontEnv,
                      "kwargs": {"queue_threshold": tuned["dropfront_threshold"]},
                      "family": "dropping"},
        "purge": {"cls": PurgeInfeasibleEnv, "kwargs": {}, "family": "ours"},
        "purge_srpf": {"cls": LeastSlackEnv, "kwargs": {}, "family": "ours"},
        "purge_edf": {"cls": PurgeEDFEnv, "kwargs": {}, "family": "ours"},
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
    row.update(load=LOAD, arm=label.split("::")[-1], label=label,
               managed=int(getattr(wrapper.env, "managed_count", 0)),
               purged_infeasible=int(getattr(wrapper.env, "purged_count", 0)),
               physical_sha256=wrapper.physical_digest.hexdigest())
    if sum(row[f"drop_{r}"] for r in diagnostics.DROP_REASONS) != row["dropped"]:
        raise AssertionError("drop partition mismatch")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arms = build_arms(TUNED)
    Wrapper1008._provider = HypatiaTopologyProvider.from_csv(TOPOLOGY)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    write_json(output / "manifest.json", {
        "role": "tle1008_family_full_dev",
        "runner_sha256": sha(Path(__file__)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_note": ("development-tier full family comparison at 1008 satellites; "
                          "routing fixed to CachedGlobalDijkstraPolicy (decision-identical "
                          "cache over GlobalDijkstraPolicy); dropping params frozen from "
                          "the 66-star tune panel; fresh workloads 880841-880860"),
        "tuned_params": TUNED,
        "topology": str(TOPOLOGY), "topology_sha256": sha(TOPOLOGY),
        "planes": PLANES, "per_plane": PER_PLANE,
        "workloads": list(WORKLOADS), "load": LOAD,
        "arms": {k: {"family": v["family"]} for k, v in arms.items()},
        "training": False, "sealed_test_access": False,
    })
    rows = []
    pairing = {}
    started = time.monotonic()
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for w in WORKLOADS:
            for arm_name, spec in arms.items():
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
    base_rows = [r for r in rows if r["arm"] == "base"]
    per_load_stats = {LOAD: [pair_stats(rows, LOAD, arm_name, base_rows)
                             for arm_name in arms]}
    write_json(output / "family_stats.json", per_load_stats)
    write_report(output, [LOAD], arms, per_load_stats, TUNED, "1008-star-full")
    write_json(output / "completion.json", {
        "status": "complete", "episodes": len(rows),
        "pairings_verified": len(pairing),
        "elapsed_seconds": time.monotonic() - started,
        "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    })
    print("1008-star full family probe complete", flush=True)


if __name__ == "__main__":
    main()
