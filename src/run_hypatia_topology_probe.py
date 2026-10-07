"""Validate the packet-management mechanism on a Hypatia +Grid ISL topology.

Uses satgenpy's generate_plus_grid_isls verbatim (from github.com/snkas/hypatia,
the ns-3 LEO constellation simulator behind the IEEE/JAS Hypatia paper) to
define the inter-satellite link rule for the 24-satellite shell (4 orbits x
6 sats, the frozen real-TLE selection in data/starlink_24_selected.tle).
The inter-orbit phase shift is chosen geometrically (minimum mean cross-plane
ISL length over all shifts at the TLE epoch), which is how the shift is tuned
in the Hypatia pipeline. Per-slot delays come from SGP4 positions (distance
over light speed) and are exported in the topology CSV schema that
HypatiaTopologyProvider already consumes, so the simulation environment is
unchanged except for the ISL topology rule.

Arms (fixed GlobalDijkstraPolicy routing, hotspot load 16, 20 fresh workloads
910101-910120): base, purge_srpf, class_priority, lcfs - run on BOTH the
default synthetic 24-star topology and the Hypatia +Grid topology, paired by
workload, to show the family ordering and the mechanism gain replicate under
the Hypatia ISL rule.
"""

import argparse
import csv
import importlib.util
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

from hypatia_topology_provider_stub import HypatiaTopologyProvider
from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MULTIAGENT_LOADS, MultiAgentConfig
from mappo_evaluation import GlobalDijkstraPolicy, evaluate_policy_with_constraint_metrics
from run_development_load_diagnostics import (
    DiagnosticWrapper,
    canonical,
    packet_accounting,
    sha,
    write_json,
)
import run_development_load_diagnostics as diagnostics
from run_srpf_probe import LeastSlackEnv
from run_packet_mgmt_family_probe import ClassPriorityEnv, LCFSEnv
from tle_topology_builder import (
    LIGHT_SPEED_KM_S,
    position_km,
    read_three_line_tle,
)
from sgp4.conveniences import sat_epoch_datetime

HYPATIA_ISLS = Path("F:/hypatia/satgenpy/satgen/isls/generate_plus_grid_isls.py")
TLE = Path("../data/starlink_24_selected.tle")
OUT_CSV = Path("../data/starlink_24_hypatia_links.csv")
SCENARIO = "hotspot_high_load"
LOAD = 16
SLOTS = 30
SLOT_SECONDS = 10
N_PLANES, N_SPP = 4, 6
WORKLOADS = tuple(range(910101, 910121))

ARMS = {
    "base": {"cls": None, "kwargs": {}, "family": "baseline"},
    "purge_srpf": {"cls": LeastSlackEnv, "kwargs": {}, "family": "ours"},
    "class_priority": {"cls": ClassPriorityEnv, "kwargs": {}, "family": "scheduling"},
    "lcfs": {"cls": LCFSEnv, "kwargs": {}, "family": "scheduling"},
}


def load_hypatia_generator():
    spec = importlib.util.spec_from_file_location(
        "hypatia_generate_plus_grid_isls", HYPATIA_ISLS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_plus_grid_isls


def export_hypatia_topology():
    sats = read_three_line_tle(str(TLE))
    assert len(sats) == N_PLANES * N_SPP, f"expected 24 sats, got {len(sats)}"
    epoch = max(sat_epoch_datetime(s.satrec) for s in sats)
    positions = {t + 1: [np.linalg.norm(position_km(s, epoch + timedelta(
        seconds=t * SLOT_SECONDS))) for s in sats] for t in range(SLOTS)}
    vecs = {t + 1: [position_km(s, epoch + timedelta(seconds=t * SLOT_SECONDS))
                    for s in sats] for t in range(SLOTS)}

    def plane_pos(sat_idx):
        return sat_idx // N_SPP, sat_idx % N_SPP

    def cross_length(shift, t):
        total, count = 0.0, 0
        for i in range(N_PLANES):
            for j in range(N_SPP):
                a = i * N_SPP + j
                b = ((i + 1) % N_PLANES) * N_SPP + ((j + shift) % N_SPP)
                total += float(np.linalg.norm(vecs[t][a] - vecs[t][b]))
                count += 1
        return total / count

    shift_scores = {s: cross_length(s, 1) for s in range(N_SPP)}
    best_shift = min(shift_scores, key=shift_scores.get)
    print(f"hypatia isl_shift scores (km): "
          + ", ".join(f"{s}:{v:.0f}" for s, v in shift_scores.items())
          + f" -> chosen {best_shift}", flush=True)

    generate = load_hypatia_generator()
    tmp = Path("../data/_hypatia_isls.txt")
    isls = generate(str(tmp), N_PLANES, N_SPP, best_shift)
    tmp.unlink()
    is_cross_of = {}
    for a, b in isls:
        cross = plane_pos(a)[0] != plane_pos(b)[0]
        is_cross_of[(a + 1, b + 1)] = cross
        is_cross_of[(b + 1, a + 1)] = cross
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_slot", "src", "dst", "delay_ms", "available",
                    "capacity_mbps", "reliability", "t_rem", "is_cross",
                    "shell_src", "shell_dst"])
        for t in range(1, SLOTS + 1):
            for (u0, v0) in isls:
                for u, v in ((u0 + 1, v0 + 1), (v0 + 1, u0 + 1)):
                    dist = float(np.linalg.norm(vecs[t][u - 1] - vecs[t][v - 1]))
                    w.writerow([t, u, v, round(dist / LIGHT_SPEED_KM_S * 1000.0, 6),
                                "True", 100.0, 0.995, 999.0,
                                is_cross_of[(u, v)], 1, 1])
    print(f"exported {OUT_CSV} ({len(isls)} undirected ISLs, "
          f"shift={best_shift}, {SLOTS} slots)", flush=True)
    return best_shift, len(isls)


class Hyp24Wrapper(DiagnosticWrapper):
    _provider = None

    def __init__(self, workload_seed, arm_spec, use_provider):
        cfg = MultiAgentConfig(
            env=EnvConfig(seed=workload_seed, scenario=SCENARIOS[SCENARIO],
                          topology_provider=(Hyp24Wrapper._provider
                                             if use_provider else None),
                          n_planes=N_PLANES, sats_per_plane=N_SPP),
            initial_packets=8, exogenous_packets_per_slot=LOAD,
            seed=workload_seed, variant="qos_only",
        )
        super().__init__(scenario=SCENARIO, cfg=cfg, seed=workload_seed)
        if arm_spec["cls"] is not None:
            self.env = arm_spec["cls"](self.env.cfg, **arm_spec["kwargs"])


def run_row(policy, label, workload, arm_spec, use_provider):
    wrapper = Hyp24Wrapper(workload, arm_spec, use_provider)
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shift, n_isls = (None, None)
    if not args.skip_export:
        shift, n_isls = export_hypatia_topology()
        Hyp24Wrapper._provider = HypatiaTopologyProvider.from_csv(OUT_CSV)
    else:
        Hyp24Wrapper._provider = HypatiaTopologyProvider.from_csv(OUT_CSV)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    write_json(output / "manifest.json", {
        "role": "hypatia_topology_probe_dev",
        "runner_sha256": sha(Path(__file__)),
        "topology_rule": "satgenpy generate_plus_grid_isls (github.com/snkas/hypatia)",
        "hypatia_isls_module_sha256": sha(HYPATIA_ISLS),
        "isl_shift": shift, "n_isls": n_isls,
        "topology_csv": str(OUT_CSV), "topology_csv_sha256": sha(OUT_CSV),
        "scenario": SCENARIO, "load": LOAD, "workloads": list(WORKLOADS),
        "arms": {k: {"family": v["family"]} for k, v in ARMS.items()},
        "training": False, "sealed_test_access": False,
    })
    started = time.monotonic()
    rows = []
    pairing = {}
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for topo in ("default", "hypatia"):
            for w in WORKLOADS:
                for arm_name, spec in ARMS.items():
                    row = run_row(GlobalDijkstraPolicy(), f"dijkstra::{arm_name}",
                                  w, spec, use_provider=(topo == "hypatia"))
                    row["topology"] = topo
                    events = (row["arrival_sha256"], row["physical_sha256"])
                    key = (topo, w)
                    if key in pairing:
                        if pairing[key] != events:
                            raise AssertionError(f"pairing changed at {key}")
                    else:
                        pairing[key] = events
                    stream.write(canonical(row) + "\n")
                    rows.append(row)
                stream.flush()
            print(f"{len(rows)} topology={topo} done "
                  f"elapsed={time.monotonic()-started:.0f}s", flush=True)
    stats = {}
    for topo in ("default", "hypatia"):
        base = {r["workload_seed"]: r["delivery_ratio"] for r in rows
                if r["topology"] == topo and r["arm"] == "base"}
        per_arm = {}
        for arm_name in ARMS:
            delivery = np.array([
                next(r["delivery_ratio"] for r in rows
                     if r["topology"] == topo and r["arm"] == arm_name
                     and r["workload_seed"] == w)
                for w in WORKLOADS])
            base_delivery = np.array([base[w] for w in WORKLOADS])
            per_arm[arm_name] = {
                "mean_delivery": float(delivery.mean()),
                "delta_pp_vs_base": float(100 * (delivery.mean() - base_delivery.mean())),
                "relative_gain": float(delivery.mean() / base_delivery.mean() - 1.0),
                "positive": int(((delivery - base_delivery) > 0).sum()),
            }
        stats[topo] = per_arm
    write_json(output / "topology_stats.json", {
        "stats": stats,
        "pairings_verified": len(pairing),
        "episodes": len(rows),
    })
    lines = ["# Hypatia +Grid topology probe (dev)", ""]
    for topo in ("default", "hypatia"):
        lines.append(f"## {topo}")
        lines.append("| arm | mean delivery | Δpp | relative | positive |")
        lines.append("|---|---|---|---|---|")
        for arm_name in ARMS:
            s = stats[topo][arm_name]
            lines.append(f"| {arm_name} | {s['mean_delivery']:.4f} "
                         f"| {s['delta_pp_vs_base']:+.2f} "
                         f"| {100*s['relative_gain']:+.1f}% "
                         f"| {s['positive']}/{len(WORKLOADS)} |")
        lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "completion.json", {
        "status": "complete", "episodes": len(rows),
        "pairings_verified": len(pairing),
        "isl_shift": shift,
        "elapsed_seconds": time.monotonic() - started,
        "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    })
    print(canonical(stats), flush=True)


if __name__ == "__main__":
    main()
