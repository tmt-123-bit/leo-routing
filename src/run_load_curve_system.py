"""Load-curve probe: full system vs Q-routing across the load dimension.

The packet-management stack was previously evaluated only at the two standard
loads (6, 16). Dead-packet accumulation — and therefore recoverable service
turns — plausibly peaks in the intermediate-saturation region. This probe
evaluates the full system (both training scenario variants, 3 seeds each),
Q-routing base, and Q-routing + stack on one fresh panel across loads
2, 4, 6, 8, 10, 12, 16 and reports the complete curve.

Predeclared readout: a load point counts toward a >= +10% claim only if the
3-seed system mean is >= 10% above same-panel Q-routing base AND the
neighbouring load points show the same direction. The full curve is reported
regardless. Descriptive development tier; the preregistered formal claim is
untouched.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from mappo_evaluation import load_checkpoint_policy
from run_development_load_diagnostics import canonical, read_json, sha, write_json
import run_avoidable_switch_classical_baselines_formal as classical
from run_direct_delivery_probe import DirectDeliveryPolicy
from run_srpf_probe import run_episode

SCENARIO = "hotspot_high_load"
LOADS = [2, 4, 6, 8, 10, 12, 16]
SEEDS = [179055553, 183895110, 310818925]
WORKLOADS = list(range(910191, 910201))
SYSTEM_ROOTS = {
    "hotspot": Path("../outputs/dev-srpf-train-20260917"),
    "medium": Path("../outputs/dev-srpf-train-medium-20260917"),
}
CLASSICAL_ROOT = Path("../experiments/avoidable-switch-classical-baselines-formal-v1-r3")


def select_checkpoint(root, seed):
    runs = sorted((root / f"seed_{seed}").glob("*/"))
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    cands = {}
    for line in (latest / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("record_type") != "validation":
            continue
        sw = d.get("avoidable_switch_rate") or d.get("decision_avoidable_switch_rate")
        dr = d.get("delivery_ratio")
        if dr is None or sw is None or sw > 0.12:
            continue
        cands[d["environment_steps"]] = dr
    step = max(cands, key=lambda s: cands[s])
    return latest / f"validation_candidate_step_{step}.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    freeze = read_json(CLASSICAL_ROOT / "training_freeze.json")
    spec = read_json(CLASSICAL_ROOT / "preregistration.json")
    jobs = {j.policy_seed: j for j in classical.build_training_jobs()
            if j.scenario == SCENARIO}
    systems = {}
    for variant, root in SYSTEM_ROOTS.items():
        for s in SEEDS:
            systems[(variant, s)], _ = load_checkpoint_policy(select_checkpoint(root, s), device="cpu")
    q_policies = {}
    for s in SEEDS:
        entry = freeze["jobs"][jobs[s].job_id]["model"]
        q_policies[s], _ = classical.load_q_model(Path(entry["path"]), jobs[s], spec, require_frozen=True)
    write_json(output / "manifest.json", {
        "role": "load_curve_system_dev",
        "runner_sha256": sha(Path(__file__)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenario": SCENARIO, "loads": LOADS, "seeds": SEEDS, "workloads": WORKLOADS,
        "arms": ("system_hotspot_trained", "system_medium_trained", "q_base", "q_mech"),
        "readout_predeclared": ("a load point supports a >=+10% claim only with 3-seed "
                                "system mean >= 1.10x same-panel Q-routing base and the "
                                "same direction at neighbouring loads; full curve always "
                                "reported"),
        "training": False, "sealed_test_access": False,
    })
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    started = time.monotonic()
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for load in LOADS:
            for (variant, s), policy in systems.items():
                for w in WORKLOADS:
                    row = run_episode(SCENARIO, load, f"sys_{variant}_{s}",
                                      DirectDeliveryPolicy(policy), s, w, "purge_srpf")[0]
                    row["load"] = load
                    stream.write(canonical(row) + "\n")
                    rows.append(row)
            for s, qp in q_policies.items():
                for arm, label in (("base", f"q_{s}_base"), ("purge_srpf", f"q_{s}_mech")):
                    for w in WORKLOADS:
                        row = run_episode(SCENARIO, load, label, qp, s, w, arm)[0]
                        row["load"] = load
                        stream.write(canonical(row) + "\n")
                        rows.append(row)
            stream.flush()
            print(f"load={load} done, {len(rows)} rows, elapsed={time.monotonic()-started:.0f}s", flush=True)
    curve = {}
    for load in LOADS:
        entry = {}
        for variant in SYSTEM_ROOTS:
            per_seed = [np.mean([r["delivery_ratio"] for r in rows
                                 if r["load"] == load and r["policy"] == f"sys_{variant}_{s}"])
                        for s in SEEDS]
            entry[f"system_{variant}"] = {"mean": float(np.mean(per_seed)),
                                          "per_seed": [float(x) for x in per_seed]}
        q_base = float(np.mean([r["delivery_ratio"] for r in rows
                                if r["load"] == load and r["policy"].endswith("_base") and r["policy"].startswith("q_")]))
        q_mech = float(np.mean([r["delivery_ratio"] for r in rows
                                if r["load"] == load and r["policy"].endswith("_mech") and r["policy"].startswith("q_")]))
        entry["q_base"] = q_base
        entry["q_mech"] = q_mech
        entry["relative_gain_hotspot_trained"] = entry["system_hotspot"]["mean"] / q_base - 1
        entry["relative_gain_medium_trained"] = entry["system_medium"]["mean"] / q_base - 1
        curve[load] = entry
    write_json(output / "curve.json", curve)
    lines = ["load  q_base  sys(hot)  sys(med)  q_mech  rel(hot)  rel(med)"]
    for load, e in curve.items():
        lines.append("%4d  %.4f  %.4f  %.4f  %.4f  %+6.2f%%  %+6.2f%%" % (
            load, e["q_base"], e["system_hotspot"]["mean"], e["system_medium"]["mean"],
            e["q_mech"], 100 * e["relative_gain_hotspot_trained"], 100 * e["relative_gain_medium_trained"]))
    (output / "curve.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "completion.json", {"status": "complete", "episodes": len(rows),
               "elapsed_seconds": time.monotonic() - started,
               "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
