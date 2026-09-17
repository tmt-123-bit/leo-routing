"""Development-tier final system evaluation: purge-env-retrained constrained
MAPPO + direct-delivery override + infeasibility purge (+ SRPF reorder) versus
frozen Q-routing identities.

Readout rules are frozen in the manifest before any episode runs:

- primary endpoint: mean delivery ratio of the 8-seed system (best validation
  candidate per seed, argmax delivery among candidates within the 12% switch
  budget) versus mean delivery ratio of Q-routing identities on the same fresh
  panel; success requires relative gain >= +10%;
- supportive: per-seed relative gains versus Q-routing base all positive, and
  every system-seed avoidable-switch rate <= 12%;
- fairness contrast (always reported): Q-routing with the identical environment
  mechanisms (purge + SRPF reorder);
- decomposition arm: system without SRPF (purge + direct only).
"""

import argparse
import gzip
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
LOAD = 16
SYSTEM_SEEDS = [179055553, 183895110, 310818925, 515636025,
                626669596, 746965870, 985998595, 2136406109]
WORKLOAD_SEEDS = list(range(910121, 910131))
TRAIN_ROOT = Path("../outputs/dev-purge-train-20260916")
CLASSICAL_ROOT = Path("../experiments/avoidable-switch-classical-baselines-formal-v1-r3")


def select_checkpoint(seed: int) -> tuple[Path, float]:
    run_dirs = sorted((TRAIN_ROOT / f"seed_{seed}").glob("*/"))
    latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
    candidates, seen = [], set()
    for line in (latest / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("record_type") != "validation":
            continue
        step = d.get("environment_steps")
        if step in seen:
            continue
        seen.add(step)
        switch = d.get("avoidable_switch_rate") or d.get("decision_avoidable_switch_rate")
        candidates.append((step, d.get("delivery_ratio"), switch))
    eligible = [c for c in candidates if c[2] is not None and c[2] <= 0.12 and c[1] is not None]
    if not eligible:
        raise ValueError(f"seed {seed}: no validation candidate within switch budget")
    step, delivery, _ = max(eligible, key=lambda c: c[1])
    path = latest / f"validation_candidate_step_{step}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path, delivery


def main():
    global TRAIN_ROOT, WORKLOAD_SEEDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT,
                        help="root containing seed_*/ run directories")
    parser.add_argument("--workload-start", type=int, default=WORKLOAD_SEEDS[0])
    args = parser.parse_args()
    TRAIN_ROOT = args.train_root.resolve()
    WORKLOAD_SEEDS = list(range(args.workload_start, args.workload_start + 10))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    classical_freeze = read_json(CLASSICAL_ROOT / "training_freeze.json")
    classical_spec = read_json(CLASSICAL_ROOT / "preregistration.json")
    q_jobs = {j.policy_seed: j for j in classical.build_training_jobs()
              if j.scenario == SCENARIO}
    q_seeds = [s for s in SYSTEM_SEEDS if s in q_jobs]
    checkpoints = {s: select_checkpoint(s) for s in SYSTEM_SEEDS}

    manifest = {
        "role": "development_final_system_eval",
        "runner_sha256": sha(Path(__file__)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenario": SCENARIO, "load": LOAD, "workloads": WORKLOAD_SEEDS,
        "train_root": str(TRAIN_ROOT),
        "primary_endpoint": ("mean 8-seed system delivery vs mean Q-routing base delivery "
                             "on the same panel; success iff relative gain >= 0.10"),
        "supportive_endpoints": ["per-seed relative gain vs Q-routing base all positive",
                                 "system per-seed avoidable-switch rate <= 0.12"],
        "fairness_contrast": "Q-routing with identical purge+SRPF environment, always reported",
        "decomposition_arm": "system without SRPF (purge + direct only)",
        "checkpoint_selection": ("argmax validation delivery among training candidates within "
                                 "the 12% switch budget; selection uses training validation "
                                 "records only, never this panel"),
        "system_checkpoints": {str(s): {"path": str(p), "sha256": sha(p), "validation_delivery": d}
                               for s, (p, d) in checkpoints.items()},
        "q_routing_seeds": q_seeds,
        "training": False, "sealed_test_access": False,
    }
    write_json(output / "manifest.json", manifest)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    rows = []
    started = time.monotonic()
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        def log(row):
            stream.write(canonical(row) + "\n")
            rows.append(row)

        for seed in SYSTEM_SEEDS:
            path, _ = checkpoints[seed]
            policy, _ = load_checkpoint_policy(path, device="cpu")
            wrapped = DirectDeliveryPolicy(policy)
            for arm, label in (("purge", f"system_seed_{seed}_purge_direct"),
                               ("purge_srpf", f"system_seed_{seed}_purge_direct_srpf")):
                for workload in WORKLOAD_SEEDS:
                    row, _ = run_episode(SCENARIO, LOAD, label, wrapped, seed, workload, arm)
                    row["system_seed"] = seed
                    row["system_arm"] = arm
                    log(row)
                print(f"{len(rows)} {label} elapsed={time.monotonic()-started:.0f}s", flush=True)
        for seed in q_seeds:
            job = q_jobs[seed]
            entry = classical_freeze["jobs"][job.job_id]["model"]
            q_policy, _ = classical.load_q_model(Path(entry["path"]), job, classical_spec,
                                                 require_frozen=True)
            for arm, label in (("base", f"q_routing_seed_{seed}_base"),
                               ("purge_srpf", f"q_routing_seed_{seed}_purge_srpf")):
                for workload in WORKLOAD_SEEDS:
                    row, _ = run_episode(SCENARIO, LOAD, label, q_policy, seed, workload, arm)
                    row["system_seed"] = None
                    row["system_arm"] = None
                    log(row)
                print(f"{len(rows)} {label} elapsed={time.monotonic()-started:.0f}s", flush=True)

    def arm_mean(label_part):
        vals = [r["delivery_ratio"] for r in rows if label_part in r["policy"]]
        return float(np.mean(vals)), len(vals)

    q_base, q_base_n = arm_mean("q_routing_seed_")
    q_base = float(np.mean([r["delivery_ratio"] for r in rows if r["policy"].endswith("_base")]))
    q_fair, _ = arm_mean("_purge_srpf")
    sys_srpf = {}
    sys_nosrpf = {}
    switch_rates = {}
    for seed in SYSTEM_SEEDS:
        sys_srpf[seed] = float(np.mean([r["delivery_ratio"] for r in rows
                                        if r.get("system_seed") == seed and r["system_arm"] == "purge_srpf"]))
        sys_nosrpf[seed] = float(np.mean([r["delivery_ratio"] for r in rows
                                          if r.get("system_seed") == seed and r["system_arm"] == "purge"]))
        treat = [r for r in rows if r.get("system_seed") == seed and r["system_arm"] == "purge_srpf"]
        opp = sum(r["decision_switch_opportunities"] for r in treat)
        switch_rates[seed] = sum(r["decision_avoidable_switches"] for r in treat) / opp

    sys_mean = float(np.mean(list(sys_srpf.values())))
    per_seed_rel = {str(s): sys_srpf[s] / q_base - 1 for s in SYSTEM_SEEDS}
    result = {
        "q_routing_base_mean": q_base,
        "q_routing_purge_srpf_mean": q_fair,
        "system_mean": sys_mean,
        "system_per_seed": sys_srpf,
        "system_per_seed_without_srpf": sys_nosrpf,
        "system_switch_rates": switch_rates,
        "primary_relative_gain": sys_mean / q_base - 1,
        "primary_success": sys_mean / q_base - 1 >= 0.10,
        "per_seed_relative_gain": per_seed_rel,
        "supportive_all_positive": all(v > 0 for v in per_seed_rel.values()),
        "supportive_switch_within_budget": all(v <= 0.12 for v in switch_rates.values()),
        "fairness_relative_gain_of_q_with_mechanisms": q_fair / q_base - 1,
        "system_vs_q_with_mechanisms": sys_mean / q_fair - 1,
        "episodes": len(rows),
    }
    write_json(output / "result.json", result)
    write_json(output / "summary.json", result)
    write_json(output / "completion.json", {"status": "complete", "episodes": len(rows),
               "elapsed_seconds": time.monotonic() - started,
               "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
