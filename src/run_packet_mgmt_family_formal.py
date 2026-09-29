"""Preregistered formal runner for the packet-management family study
(docs/PACKET_MGMT_FAMILY_FORMAL_V1.md).

Stage ``gate``: base / class_priority / purge_srpf on workloads 880591-880610
at load 12; the formal panel must not be touched unless purge_srpf beats both.

Stage ``formal``: one-shot grid on workloads 880611-880650 — load 12 (primary),
8 (secondary), 4 (descriptive); all ten arms. Frozen statistics: workload
bootstrap of the ratio of means (5,000 draws, seed 20260926), paired
per-workload gains, exact sign tests, and the predeclared endpoints P1/S1/S2
of the protocol.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from hypatia_topology_provider_stub import HypatiaTopologyProvider
from mappo_evaluation import GlobalDijkstraPolicy
from run_development_load_diagnostics import canonical, read_json, sha, write_json
from run_packet_mgmt_family_probe import (
    TOPOLOGY,
    FamilyWrapper,
    build_arms,
    pair_stats,
    run_row,
    write_report,
)

PROTOCOL = Path("../docs/PACKET_MGMT_FAMILY_FORMAL_V1.md")
TUNING = Path("../experiments/packet-mgmt-family-tune-20260925-v1/tuning.json")
GATE_WORKLOADS = list(range(880591, 880611))
FORMAL_WORKLOADS = list(range(880611, 880651))
LOADS = ((12, "primary"), (8, "secondary"), (4, "descriptive"))
GATE_ARMS = ("base", "class_priority", "purge_srpf")
BOOTSTRAP_SEED = 20260926


def paired_reference(rows, load, arm, ref_arm):
    ref_rows = [r for r in rows if r["load"] == load and r["arm"] == ref_arm]
    return pair_stats(rows, load, arm, ref_rows)


def one_sided_sign_p(gains):
    positive = int((np.array(gains) > 0).sum())
    n = len(gains)
    from math import comb
    tail = sum(comb(n, k) for k in range(positive, n + 1))
    return float(tail / 2 ** n)


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

    tuned = read_json(TUNING.resolve())["tuned"]
    arms = build_arms(tuned)
    FamilyWrapper._provider = HypatiaTopologyProvider.from_csv(TOPOLOGY)
    workloads = GATE_WORKLOADS if args.stage == "gate" else FORMAL_WORKLOADS
    write_json(output / "manifest.json", {
        "role": f"packet_mgmt_family_formal_{args.stage}",
        "runner_sha256": sha(Path(__file__)),
        "protocol": str(PROTOCOL), "protocol_sha256": sha(PROTOCOL),
        "tuning_source": str(TUNING), "tuning_sha256": sha(TUNING),
        "tuned_params": tuned,
        "topology": str(TOPOLOGY), "topology_sha256": sha(TOPOLOGY),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workloads": workloads,
        "loads": [12] if args.stage == "gate" else [l for l, _ in LOADS],
        "arms": sorted(arms) if args.stage == "formal" else list(GATE_ARMS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "training": False, "sealed_test_access": False,
    })
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    stage_arms = GATE_ARMS if args.stage == "gate" else list(arms)
    rows = []
    pairing = {}
    started = time.monotonic()
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for load, _role in ([(12, "gate")] if args.stage == "gate" else LOADS):
            for w in workloads:
                for arm_name in stage_arms:
                    row = run_row(GlobalDijkstraPolicy(),
                                  f"dijkstra::{arm_name}", w, load, arms[arm_name])
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
            print(f"{len(rows)} load={load} done elapsed={time.monotonic()-started:.0f}s",
                  flush=True)

    means = {}
    for arm_name in stage_arms:
        vals = [r["delivery_ratio"] for r in rows if r["arm"] == arm_name]
        means[arm_name] = float(np.mean(vals))

    if args.stage == "gate":
        gate_pass = (means["purge_srpf"] > means["base"]
                     and means["purge_srpf"] > means["class_priority"])
        write_json(output / "completion.json", {
            "status": "complete", "episodes": len(rows),
            "pairings_verified": len(pairing),
            "gate_pass": bool(gate_pass),
            "mean_delivery": means,
            "elapsed_seconds": time.monotonic() - started,
            "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        })
        print(canonical({"gate_pass": bool(gate_pass), "mean_delivery": means}), flush=True)
        return

    per_load_stats = {}
    for load, _role in LOADS:
        stats = [pair_stats(rows, load, arm_name, [r for r in rows if r["arm"] == "base"])
                 for arm_name in arms]
        per_load_stats[load] = stats

    p1 = paired_reference(rows, 12, "purge_srpf", "base")
    p1_lower = p1["bootstrap_vs_base"]["one_sided_95_lower"]
    p1_sign_p = one_sided_sign_p(p1["paired_gain_pp"])
    primary_pass = bool(p1_lower >= 0.10 and p1_sign_p < 0.001)

    s1 = paired_reference(rows, 12, "purge_srpf", "class_priority")
    s1_lower = s1["bootstrap_vs_base"]["one_sided_95_lower"]
    s1_sign_p = one_sided_sign_p(s1["paired_gain_pp"])
    s1_pass = bool(s1_lower > 0 and s1_sign_p < 0.01)

    s2 = {}
    for opponent in ("codel", "red", "dropfront"):
        st = paired_reference(rows, 12, "purge", opponent)
        s2[opponent] = {
            "delta_pp": st["delta_pp_vs_base"],
            "one_sided_95_lower": st["bootstrap_vs_base"]["one_sided_95_lower"],
            "sign_one_sided_p": one_sided_sign_p(st["paired_gain_pp"]),
        }
    s2_pass_count = sum(1 for v in s2.values() if v["one_sided_95_lower"] > 0)
    s2_pass = bool(s2_pass_count >= 2)

    write_report(
        output, [l for l, _ in LOADS], arms, per_load_stats, tuned, "formal")
    with (output / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
        fields = ["load", "arm", "mean_delivery", "delta_pp_vs_base", "one_sided_95_lower",
                  "sign_p", "mean_managed"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for load, _role in LOADS:
            for stat in per_load_stats[load]:
                writer.writerow({
                    "load": load, "arm": stat["arm"],
                    "mean_delivery": f"{stat['mean_delivery']:.6f}",
                    "delta_pp_vs_base": f"{stat['delta_pp_vs_base']:.4f}",
                    "one_sided_95_lower": (stat.get("bootstrap_vs_base", {}).get("one_sided_95_lower")
                                           if "bootstrap_vs_base" in stat else ""),
                    "sign_p": (stat.get("sign_test", {}).get("exact_two_sided_p")
                               if "sign_test" in stat else ""),
                    "mean_managed": f"{stat['mean_managed']:.2f}",
                })
    write_json(output / "endpoints.json", {
        "P1_purge_srpf_vs_base": {
            "delta_pp": p1["delta_pp_vs_base"], "one_sided_95_lower": p1_lower,
            "sign_one_sided_p": p1_sign_p, "pass": primary_pass, "rule": "lower>=+10% and sign p<0.001",
        },
        "S1_purge_srpf_vs_class_priority": {
            "delta_pp": s1["delta_pp_vs_base"], "one_sided_95_lower": s1_lower,
            "sign_one_sided_p": s1_sign_p, "pass": s1_pass, "rule": "lower>0 and sign p<0.01",
        },
        "S2_purge_vs_dropping_family": {**s2, "pass_count": s2_pass_count, "pass": s2_pass,
                                        "rule": "at least 2 of 3 with lower>0"},
        "primary_pass": primary_pass,
    })
    write_json(output / "completion.json", {
        "status": "complete", "episodes": len(rows),
        "pairings_verified": len(pairing),
        "primary_pass": primary_pass, "s1_pass": s1_pass, "s2_pass": s2_pass,
        "elapsed_seconds": time.monotonic() - started,
        "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    })
    print(canonical(read_json(output / "endpoints.json")), flush=True)


if __name__ == "__main__":
    main()
