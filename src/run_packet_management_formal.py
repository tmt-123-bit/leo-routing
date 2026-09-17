"""Formal evaluation runner for the deadline-aware packet-management study
(docs/DEADLINE_PACKET_MANAGEMENT_FORMAL_V1.md).

Stage ``gate``: hotspot, system (8 seeds) versus frozen Q-routing base on
workloads 77011..77020. The study stops if the system delivery mean is not
above the Q-routing mean.

Stage ``formal``: full frozen grid on workloads 940101..940150 — system (8
seeds), q_base (8), q_mech (8), qos_only_base (8), pd_mech (1 sentinel) — per
scenario, one shot. Requires the completed gate directory as input. Writes
per-episode rows, per-packet ledgers (local only), and frozen statistics:
crossed bootstrap over policy seeds and workloads, exact sign-flip
sensitivity, and Holm correction over the gated endpoint family.

No training, no selection, no threshold changes inside this runner. All model
inputs are hash-bound in the manifest before any episode runs and re-verified
afterwards.
"""

import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from mappo_evaluation import load_checkpoint_policy
from run_development_load_diagnostics import (
    PersistentDijkstraPolicy,
    canonical,
    read_json,
    sha,
    write_json,
)
import run_avoidable_switch_classical_baselines_formal as classical
from run_direct_delivery_probe import DirectDeliveryPolicy
from run_srpf_probe import run_episode

SCENARIOS = ("medium_load", "hotspot_high_load")
LOADS = {"medium_load": 6, "hotspot_high_load": 16}
SEEDS = [179055553, 183895110, 310818925, 515636025,
         626669596, 746965870, 985998595, 2136406109]
GATE_WORKLOADS = list(range(77011, 77021))
FORMAL_WORKLOADS = list(range(940101, 940151))
SYSTEM_ROOTS = {
    "hotspot_high_load": Path("../outputs/dev-srpf-train-20260917"),
    "medium_load": Path("../outputs/dev-srpf-train-medium-20260917"),
}
CLASSICAL_ROOT = Path("../experiments/avoidable-switch-classical-baselines-formal-v1-r3")
METHOD_ROOT = Path("../experiments/avoidable-switch-constraint-formal-v1-r2")


def select_checkpoint(scenario, seed):
    root = SYSTEM_ROOTS[scenario]
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
    if not cands:
        raise ValueError(f"{scenario}/{seed}: no candidate within switch budget")
    step = max(cands, key=lambda s: cands[s])
    path = latest / f"validation_candidate_step_{step}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha(path), "validation_step": step,
            "validation_delivery": cands[step]}


def bind_inputs(scenario):
    system = {str(s): select_checkpoint(scenario, s) for s in SEEDS}
    freeze = read_json(CLASSICAL_ROOT / "training_freeze.json")
    spec = read_json(CLASSICAL_ROOT / "preregistration.json")
    jobs = {j.policy_seed: j for j in classical.build_training_jobs()
            if j.scenario == scenario}
    q_models = {}
    for s in SEEDS:
        entry = freeze["jobs"][jobs[s].job_id]["model"]
        q_models[str(s)] = {"path": entry["path"], "sha256": sha(entry["path"])}
    method_freeze = read_json(METHOD_ROOT / "training_freeze.json")
    qos_models = {}
    for s in SEEDS:
        entry = method_freeze["jobs"][f"{scenario}/qos_only_baseline/seed_{s}"]["artifacts"]["selected_checkpoint"]
        qos_models[str(s)] = {"path": entry["path"], "sha256": sha(entry["path"])}
    return {"system": system, "q_base": q_models, "qos_only_base": qos_models,
            "classical_spec": spec}


def verify_inputs(bound):
    for group in ("system", "q_base", "qos_only_base"):
        for s, rec in bound[group].items():
            if sha(rec["path"]) != rec["sha256"]:
                raise AssertionError(f"frozen input changed: {group}/{s}")


def crossed_bootstrap_relative(system_by_seed, q_by_seed, draws=5000, seed=20260917):
    """system_by_seed/q_by_seed: {seed: {workload: delivery}} paired."""
    seeds = sorted(system_by_seed)
    workloads = sorted(next(iter(system_by_seed.values())))
    rng = np.random.default_rng(seed)
    sys_draws, q_draws = [], []
    sys_mat = np.array([[system_by_seed[s][w] for w in workloads] for s in seeds])
    q_mat = np.array([[q_by_seed[s][w] for w in workloads] for s in seeds])
    n_s, n_w = len(seeds), len(workloads)
    for _ in range(draws):
        rs = rng.integers(0, n_s, n_s)
        rw = rng.integers(0, n_w, n_w)
        sys_draws.append(sys_mat[np.ix_(rs, rw)].mean())
        q_draws.append(q_mat[np.ix_(rs, rw)].mean())
    rel = np.array(sys_draws) / np.array(q_draws) - 1.0
    return {"relative_mean": float(rel.mean()),
            "ci95": [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))],
            "one_sided_95_lower": float(np.percentile(rel, 5))}


def sign_flip(seed_effects):
    effects = np.array(seed_effects)
    n = len(effects)
    observed = float(effects.mean())
    all_stats = []
    for mask in range(2 ** n):
        signs = np.array([1.0 if (mask >> i) & 1 == 0 else -1.0 for i in range(n)])
        all_stats.append(float((effects * signs).mean()))
    p = float(np.mean([abs(t) >= abs(observed) - 1e-12 for t in all_stats]))
    return {"observed": observed, "exact_two_sided_p": p, "n_seeds": n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gate", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, default=None,
                        help="completed gate directory (formal stage only)")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    if args.stage == "formal":
        if args.gate_dir is None:
            raise ValueError("formal stage requires --gate-dir")
        gate = read_json(args.gate_dir.resolve() / "completion.json")
        if gate.get("status") != "complete" or not gate.get("gate_pass"):
            raise ValueError("gate did not pass; formal stage is not authorized")

    workloads = GATE_WORKLOADS if args.stage == "gate" else FORMAL_WORKLOADS
    scenarios = ("hotspot_high_load",) if args.stage == "gate" else SCENARIOS
    bound_all = {sc: bind_inputs(sc) for sc in scenarios}
    manifest = {
        "role": f"packet_management_{args.stage}",
        "runner_sha256": sha(Path(__file__)),
        "protocol": "docs/DEADLINE_PACKET_MANAGEMENT_FORMAL_V1.md",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workloads": workloads, "scenarios": scenarios, "seeds": SEEDS,
        "inputs": bound_all,
        "endpoints": ("primary: hotspot relative gain one-sided 95% lower >= +3% "
                      "and all 8 seed gains positive; secondary reported regardless"),
        "training": False, "sealed_test_access": False,
    }
    write_json(output / "manifest.json", manifest)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    started = time.monotonic()
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as episode_file:
        if args.stage == "formal":
            ledger = gzip.open(output / "packets.jsonl.gz", "wt", encoding="utf-8")
        for scenario in scenarios:
            bound = bound_all[scenario]
            load = LOADS[scenario]
            arms = []
            for s in SEEDS:
                arms.append((f"system_{s}", f"system", s, "purge_srpf", "system"))
                arms.append((f"q_base_{s}", "q_base", s, "base", "q_base"))
            if args.stage == "formal":
                for s in SEEDS:
                    arms.append((f"q_mech_{s}", "q_mech", s, "purge_srpf", "q_mech"))
                    arms.append((f"qos_only_base_{s}", "qos_only_base", s, "base", "qos_only"))
                arms.append(("pd_mech", "pd_mech", -1, "purge_srpf", "pd"))
            for label, _, seed, env_arm, kind in arms:
                if kind == "system":
                    policy, _ = load_checkpoint_policy(Path(bound["system"][str(seed)]["path"]), device="cpu")
                    policy = DirectDeliveryPolicy(policy)
                elif kind in ("q_base", "q_mech"):
                    job = next(j for j in classical.build_training_jobs()
                               if j.scenario == scenario and j.policy_seed == seed)
                    policy, _ = classical.load_q_model(
                        Path(bound["q_base"][str(seed)]["path"]), job,
                        bound["classical_spec"], require_frozen=True)
                elif kind == "qos_only":
                    policy, _ = load_checkpoint_policy(Path(bound["qos_only_base"][str(seed)]["path"]), device="cpu")
                else:
                    policy = PersistentDijkstraPolicy()
                for w in workloads:
                    row, packets = run_episode(scenario, load, label, policy, seed, w, env_arm)
                    row["arm_kind"] = kind
                    episode_file.write(canonical(row) + "\n")
                    rows.append(row)
                    if args.stage == "formal":
                        identity = {"scenario": scenario, "label": label,
                                    "policy_seed": seed, "workload_seed": w}
                        for packet in packets:
                            ledger.write(canonical({**identity, **packet}) + "\n")
                episode_file.flush()
                print(f"{len(rows)} {scenario} {label} elapsed={time.monotonic()-started:.0f}s", flush=True)
        if args.stage == "formal":
            ledger.close()

    for sc in scenarios:
        verify_inputs(bound_all[sc])

    def collect(scenario, kind):
        per = {}
        for r in rows:
            if r["scenario"] == scenario and r["arm_kind"] == kind:
                per.setdefault(r["policy_seed"], {})[r["workload_seed"]] = r["delivery_ratio"]
        return per

    result = {"stage": args.stage, "episodes": len(rows)}
    if args.stage == "gate":
        sys_per, q_per = collect("hotspot_high_load", "system"), collect("hotspot_high_load", "q_base")
        sys_mean = float(np.mean([v for d in sys_per.values() for v in d.values()]))
        q_mean = float(np.mean([v for d in q_per.values() for v in d.values()]))
        result.update({"system_mean": sys_mean, "q_base_mean": q_mean,
                       "gate_pass": bool(sys_mean > q_mean)})
    else:
        stats = {}
        for sc in SCENARIOS:
            sys_per, q_per = collect(sc, "system"), collect(sc, "q_base")
            seed_gains = [float(np.mean([sys_per[s][w] - q_per[s][w] for w in FORMAL_WORKLOADS]))
                          for s in SEEDS]
            rel = crossed_bootstrap_relative(sys_per, q_per)
            switch = {}
            delay_change = []
            for s in SEEDS:
                treat = [r for r in rows if r["scenario"] == sc and r["arm_kind"] == "system" and r["policy_seed"] == s]
                opp = sum(r["decision_switch_opportunities"] for r in treat)
                switch[s] = sum(r["decision_avoidable_switches"] for r in treat) / opp
            sys_del = [r["average_delay_slots"] for r in rows if r["scenario"] == sc and r["arm_kind"] == "system" and r["average_delay_slots"] is not None]
            q_del = [r["average_delay_slots"] for r in rows if r["scenario"] == sc and r["arm_kind"] == "q_base" and r["average_delay_slots"] is not None]
            q_mech_mean = float(np.mean([r["delivery_ratio"] for r in rows if r["scenario"] == sc and r["arm_kind"] == "q_mech"]))
            sys_mean = float(np.mean([r["delivery_ratio"] for r in rows if r["scenario"] == sc and r["arm_kind"] == "system"]))
            q_mean = float(np.mean([r["delivery_ratio"] for r in rows if r["scenario"] == sc and r["arm_kind"] == "q_base"]))
            stats[sc] = {
                "system_mean": sys_mean, "q_base_mean": q_mean, "q_mech_mean": q_mech_mean,
                "relative_gain": rel,
                "seed_effects_pp": [100 * g for g in seed_gains],
                "all_seeds_positive": bool(all(g > 0 for g in seed_gains)),
                "sign_flip": sign_flip(seed_gains),
                "switch_rates": switch,
                "switch_within_budget": bool(max(switch.values()) <= 0.12),
                "mean_success_delay_relative_change": float(np.mean(sys_del) / np.mean(q_del) - 1),
                "class_delivery": {f"class_{k}": {
                    "system": float(np.mean([r[f"class_{k}_delivery_ratio"] for r in rows if r["scenario"] == sc and r["arm_kind"] == "system"])),
                    "q_base": float(np.mean([r[f"class_{k}_delivery_ratio"] for r in rows if r["scenario"] == sc and r["arm_kind"] == "q_base"])),
                } for k in range(3)},
            }
        hot = stats["hotspot_high_load"]
        primary_pass = (hot["relative_gain"]["one_sided_95_lower"] >= 0.03
                        and hot["all_seeds_positive"])
        med = stats["medium_load"]
        med_gap_pp = 100 * (med["system_mean"] - med["q_base_mean"])
        stats["decision"] = {
            "primary_pass": bool(primary_pass),
            "medium_noninferiority_pp": med_gap_pp,
            "medium_noninferiority_pass": bool(med_gap_pp >= -2.0),
        }
        result["scenarios"] = stats
    write_json(output / "result.json", result)
    write_json(output / "completion.json", {"status": "complete", "episodes": len(rows),
               "elapsed_seconds": time.monotonic() - started,
               "gate_pass": result.get("gate_pass"),
               "primary_pass": (result.get("scenarios", {}).get("decision", {}) or {}).get("primary_pass"),
               "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print(canonical({"stage": args.stage, "episodes": len(rows),
                     "gate_pass": result.get("gate_pass"),
                     "decision": result.get("scenarios", {}).get("decision")}), flush=True)


if __name__ == "__main__":
    main()
