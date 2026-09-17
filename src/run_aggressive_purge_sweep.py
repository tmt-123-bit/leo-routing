"""Development sweep: queue-aware aggressive admission on top of the SRPF system.

The safe purge removes only provably-undeliverable packets (hop distance
exceeds remaining deliverable slots with one slot of margin). This sweep adds
an estimator-based reserve: drop earlier when the local queue is deep, i.e.
drop when dist + alpha + beta * (queue_length - 1) > remaining + 1. With
alpha = beta = 0 this reproduces the safe system exactly; positive values may
drop salvageable packets in exchange for freeing service turns, tuned on this
development panel only. Q-routing is evaluated with the identical mechanism
for the fairness contrast.
"""

import argparse
import json
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MULTIAGENT_LOADS, MultiAgentConfig
from mappo_evaluation import evaluate_policy_with_constraint_metrics, load_checkpoint_policy
from run_development_load_diagnostics import (
    DiagnosticWrapper,
    canonical,
    packet_accounting,
    read_json,
    sha,
    write_json,
)
import run_development_load_diagnostics as diagnostics
import run_avoidable_switch_classical_baselines_formal as classical
from run_direct_delivery_probe import DirectDeliveryPolicy

if "early_infeasible" not in diagnostics.DROP_REASONS:
    diagnostics.DROP_REASONS = diagnostics.DROP_REASONS + ("early_infeasible",)

SCENARIO = "hotspot_high_load"
LOAD = 16
SWEEP_SEEDS = [179055553, 183895110, 310818925]
GRID = [(a, b) for a in (0, 1, 2, 3) for b in (0.0, 0.25, 0.5)]
SRPF_ROOT = Path("../outputs/dev-srpf-train-20260917")


class AggressivePurgeEnv:
    """Purge with queue-aware reserve plus SRPF reorder; built by composition."""

    def __new__(cls, cfg, alpha=0, beta=0.0):
        from run_infeasible_drop_probe import PurgeInfeasibleEnv

        class _Env(PurgeInfeasibleEnv):
            _alpha = alpha
            _beta = beta

            def _purge_infeasible_packets(self):
                if self.slot > self.cfg.episode_slots:
                    return
                if not any(self.queues.values()):
                    return
                self._refresh_graph()
                reverse = defaultdict(list)
                for (u, v), edge in self.graph.items():
                    if u >= 1 and v >= 1 and edge.available:
                        reverse[v].append(u)
                distances = {}
                for sat in range(1, self.n_agents + 1):
                    for packet_id in self.queues[sat]:
                        dst = self.packets[packet_id].dst
                        if dst not in distances:
                            distances[dst] = self._bfs_distances(dst, reverse)
                for sat in range(1, self.n_agents + 1):
                    if not self.queues[sat]:
                        continue
                    ids = list(self.queues[sat])
                    reserve = self._alpha + self._beta * max(0, len(ids) - 1)
                    kept = deque()
                    for packet_id in ids:
                        packet = self.packets[packet_id]
                        deadline = self.cfg.packet_class_deadlines[packet.traffic_class]
                        effective_bound = min(packet.created_slot + deadline - 1,
                                              self.cfg.episode_slots)
                        remaining = effective_bound - self.slot
                        dist = distances[packet.dst].get(sat)
                        if dist is not None and dist + reserve > remaining + self.safety_margin_slots:
                            self._drop_packet(packet_id, self.drop_reason_name)
                            self.purged_count += 1
                            self.purged_by_class[packet.traffic_class] += 1
                        else:
                            kept.append(packet_id)

                    def remaining_hops(packet_id):
                        dist = distances[self.packets[packet_id].dst].get(sat)
                        return (dist if dist is not None else 10 ** 6, packet_id)

                    self.queues[sat] = deque(sorted(kept, key=remaining_hops))

        return _Env(cfg)


class SweepWrapper(DiagnosticWrapper):
    def __init__(self, *args, alpha=0, beta=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = AggressivePurgeEnv(self.env.cfg, alpha=alpha, beta=beta)


def run_episode(policy, policy_seed, workload_seed, alpha, beta, label):
    initial, _ = MULTIAGENT_LOADS[SCENARIO]
    cfg = MultiAgentConfig(
        env=EnvConfig(seed=workload_seed, scenario=SCENARIOS[SCENARIO]),
        initial_packets=initial, exogenous_packets_per_slot=LOAD,
        seed=workload_seed, variant="qos_only",
    )
    wrapper = SweepWrapper(scenario=SCENARIO, cfg=cfg, alpha=alpha, beta=beta)
    result = evaluate_policy_with_constraint_metrics(
        SCENARIO, label, policy, policy_seed, [workload_seed],
        wrapper_factory=lambda _: wrapper, variant="qos_only",
    )[0]
    row = asdict(result)
    extra, _ = packet_accounting(wrapper.env)
    row.update(extra)
    return row


def select_checkpoint(seed):
    runs = sorted((SRPF_ROOT / f"seed_{seed}").glob("*/"))
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
    workloads = list(range(910151, 910161))
    write_json(output / "manifest.json", {
        "role": "development_parameter_sweep",
        "runner_sha256": sha(Path(__file__)),
        "grid": GRID, "workloads": workloads, "seeds": SWEEP_SEEDS,
        "note": ("alpha=beta=0 reproduces the safe system; tuned parameters are "
                 "development-only and any confirmatory run must use a fresh panel"),
        "training": False, "sealed_test_access": False,
    })
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    classical_freeze = read_json("../experiments/avoidable-switch-classical-baselines-formal-v1-r3/training_freeze.json")
    classical_spec = read_json("../experiments/avoidable-switch-classical-baselines-formal-v1-r3/preregistration.json")
    jobs = {j.policy_seed: j for j in classical.build_training_jobs() if j.scenario == SCENARIO}

    policies = {}
    for seed in SWEEP_SEEDS:
        base, _ = load_checkpoint_policy(select_checkpoint(seed), device="cpu")
        policies[("sys", seed)] = DirectDeliveryPolicy(base)
    for seed in SWEEP_SEEDS:
        job = jobs[seed]
        entry = classical_freeze["jobs"][job.job_id]["model"]
        q, _ = classical.load_q_model(Path(entry["path"]), job, classical_spec, require_frozen=True)
        policies[("q", seed)] = q

    started = time.monotonic()
    results = []
    q_base_rows = [run_episode(policies[("q", s)], s, w, 0, 0.0, "q_base")
                   for s in SWEEP_SEEDS for w in workloads]
    q_base = float(np.mean([r["delivery_ratio"] for r in q_base_rows]))
    for alpha, beta in GRID:
        per_seed = []
        for seed in SWEEP_SEEDS:
            rows = [run_episode(policies[("sys", seed)], seed, w, alpha, beta,
                                f"sys_{seed}_a{alpha}_b{beta}") for w in workloads]
            per_seed.append(float(np.mean([r["delivery_ratio"] for r in rows])))
        mean = float(np.mean(per_seed))
        results.append({"alpha": alpha, "beta": beta, "per_seed": per_seed,
                        "mean": mean, "relative_vs_q_base": mean / q_base - 1})
        print("alpha=%d beta=%.2f  mean=%.4f  rel=%+.2f%%  elapsed=%.0fs" % (
            alpha, beta, mean, 100 * (mean / q_base - 1), time.monotonic() - started), flush=True)
    q_fair = None
    best = max(results, key=lambda r: r["mean"])
    q_fair_rows = [run_episode(policies[("q", s)], s, w, best["alpha"], best["beta"], "q_fair")
                   for s in SWEEP_SEEDS for w in workloads]
    q_fair = float(np.mean([r["delivery_ratio"] for r in q_fair_rows]))
    write_json(output / "sweep.json", {
        "q_base_mean": q_base, "q_base_config_mean": q_fair, "grid_results": results,
        "best": best,
    })
    write_json(output / "completion.json", {"status": "complete",
               "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print("q_base=%.4f  q+bestcfg=%.4f  best=%.4f (a=%d b=%.2f)" % (
        q_base, q_fair, best["mean"], best["alpha"], best["beta"]), flush=True)


if __name__ == "__main__":
    main()
