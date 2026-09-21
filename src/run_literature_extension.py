"""Literature-baseline extension: same-environment comparison (development tier).

Trains the neural Q-routing baseline (the modern learned, queue-aware routing
family, cf. arXiv 2605.04448) budget-matched at 500 episodes per seed, and
evaluates it against frozen Q-routing and persistent Dijkstra (the ILPR-style
keep-old-route rule) on one fresh development panel, each with and without the
packet-management stack. All arms share the no_lifetime observation schema so
the DQN feature dimension matches training. Descriptive comparison only: the
preregistered formal claim (packet-management-formal-20260918-v1) is untouched
and this panel is new.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MULTIAGENT_LOADS, MultiAgentConfig
from mappo_evaluation import evaluate_policy_with_constraint_metrics
from run_development_load_diagnostics import (
    DiagnosticWrapper,
    PersistentDijkstraPolicy,
    canonical,
    read_json,
    sha,
    write_json,
)
import run_development_load_diagnostics as diagnostics
import run_avoidable_switch_classical_baselines_formal as classical
from run_dqn_baseline import train_dqn
from run_infeasible_drop_probe import PurgeInfeasibleEnv
from run_srpf_probe import LeastSlackEnv

EVAL_SCENARIOS = ("medium_load", "hotspot_high_load")
LOADS = {"medium_load": 6, "hotspot_high_load": 16}
SEEDS = [7, 42, 1024]
CLASSICAL_IDENTITIES = [179055553, 183895110, 310818925]
DQN_EPISODES = 500
WORKLOADS = list(range(910181, 910191))
VARIANT = "no_lifetime"


class LitDiagnosticWrapper(DiagnosticWrapper):
    def __init__(self, *args, arm: str = "base", **kwargs):
        super().__init__(*args, **kwargs)
        if arm == "base":
            return
        env_cls = PurgeInfeasibleEnv if arm == "purge1" else LeastSlackEnv
        self.env = env_cls(self.env.cfg)


def run_episode(scenario, load, label, policy, policy_seed, workload, arm):
    initial, _ = MULTIAGENT_LOADS[scenario]
    cfg = MultiAgentConfig(
        env=EnvConfig(seed=workload, scenario=SCENARIOS[scenario]),
        initial_packets=initial, exogenous_packets_per_slot=load,
        seed=workload, variant=VARIANT,
    )
    wrapper = LitDiagnosticWrapper(scenario=scenario, cfg=cfg, arm=arm)
    result = evaluate_policy_with_constraint_metrics(
        scenario, label, policy, policy_seed, [workload],
        wrapper_factory=lambda _: wrapper, variant=VARIANT,
    )[0]
    row = asdict(result)
    row["arm"] = arm
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    classical_freeze = read_json(diagnostics.CLASSICAL_ROOT / "training_freeze.json")
    classical_spec = read_json(diagnostics.CLASSICAL_ROOT / "preregistration.json")
    write_json(output / "manifest.json", {
        "role": "literature_baseline_extension_dev",
        "runner_sha256": sha(Path(__file__)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workloads": WORKLOADS, "seeds": SEEDS, "dqn_episodes": DQN_EPISODES,
        "variant": VARIANT,
        "arms": ("dqn_base", "dqn_mech", "ilpr_base", "ilpr_mech", "q_base"),
        "note": ("descriptive same-environment comparison; formal preregistered "
                 "claim is unaffected; ILPR operationalized as persistent cached "
                 "next hop (keep while feasible, recompute on break); DQN is the "
                 "queue-aware neural Q-routing family"),
        "training": True, "sealed_test_access": False,
    })
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    started = time.monotonic()
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for scenario in EVAL_SCENARIOS:
            load = LOADS[scenario]
            jobs = {j.policy_seed: j for j in classical.build_training_jobs()
                    if j.scenario == scenario}
            q_policies = {}
            for s, identity in zip(SEEDS, CLASSICAL_IDENTITIES):
                job = next(j for j in classical.build_training_jobs()
                           if j.scenario == scenario and j.policy_seed == identity)
                entry = classical_freeze["jobs"][job.job_id]["model"]
                policy, _ = classical.load_q_model(Path(entry["path"]), job, classical_spec, require_frozen=True)
                q_policies[s] = policy
            for s in SEEDS:
                t0 = time.monotonic()
                dqn = train_dqn(scenario, seed=s, episodes=DQN_EPISODES)
                print(f"trained dqn {scenario} seed={s} in {time.monotonic()-t0:.0f}s", flush=True)
                for arm, label in (("base", f"dqn_{s}_base"), ("purge_srpf", f"dqn_{s}_mech")):
                    for w in WORKLOADS:
                        row = run_episode(scenario, load, label, dqn, s, w, arm)
                        stream.write(canonical(row) + "\n")
                        rows.append(row)
                    stream.flush()
            for arm, label in (("base", "ilpr_base"), ("purge_srpf", "ilpr_mech")):
                for w in WORKLOADS:
                    row = run_episode(scenario, load, label, PersistentDijkstraPolicy(), -1, w, arm)
                    stream.write(canonical(row) + "\n")
                    rows.append(row)
            for s in SEEDS:
                for arm, label in (("base", f"q_{s}_base"), ("purge_srpf", f"q_{s}_mech")):
                    for w in WORKLOADS:
                        row = run_episode(scenario, load, label, q_policies[s], s, w, arm)
                        stream.write(canonical(row) + "\n")
                        rows.append(row)
                stream.flush()
            print(f"{scenario} done, {len(rows)} rows, elapsed={time.monotonic()-started:.0f}s", flush=True)
    summary = {}
    for r in rows:
        label = r["policy"]
        base = label.rsplit("_", 1)[0]
        kind = "mech" if label.endswith("_mech") else "base"
        summary.setdefault((r["scenario"], base, kind), []).append(r["delivery_ratio"])
    lines = []
    agg = {}
    for (scenario, base, kind), vals in sorted(summary.items()):
        agg[f"{scenario}/{base}/{kind}"] = float(np.mean(vals))
        lines.append(f"{scenario:22s} {base:22s} {kind:5s} {np.mean(vals):8.4f} n={len(vals)}")
    write_json(output / "summary.json", agg)
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "completion.json", {"status": "complete", "episodes": len(rows),
               "elapsed_seconds": time.monotonic() - started,
               "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
