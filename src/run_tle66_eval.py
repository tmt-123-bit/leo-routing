"""One-day 66-satellite feasibility evaluation (development tier).

Evaluates on one fresh workload panel: the SRPF-environment-trained MAPPO
system (3 seeds, best training-validation checkpoint) with and without the
packet-management stack, Dijkstra base and with the stack, and freshly
retrained Q-routing (500 episodes per identity, budget-matched) base and with
the stack. Answers one question: does the mechanism gain at the new
environment's saturation knee support a >= +10% relative delivery claim over
the strongest base baseline? Full curve of arms is reported regardless.
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
from leo_multiagent_env import MULTIAGENT_LOADS, MultiAgentConfig
from mappo_evaluation import (
    GlobalDijkstraPolicy,
    QRoutingPolicy,
    evaluate_policy_with_constraint_metrics,
    load_checkpoint_policy,
)
from run_development_load_diagnostics import canonical, sha, write_json
from run_direct_delivery_probe import DirectDeliveryPolicy
from run_infeasible_drop_probe import PurgeInfeasibleEnv
from run_srpf_probe import LeastSlackEnv
from run_development_load_diagnostics import DiagnosticWrapper

TOPOLOGY = Path("../data/starlink_66_links.csv")
TRAIN_ROOT = Path("../outputs/tle66-train-20260922")
SEEDS = [179055553, 183895110, 310818925]
Q_IDENTITIES = [7, 42]
LOAD = 8
WORKLOADS = list(range(880011, 880021))
SCENARIO = "hotspot_high_load"


class Wrapper66(DiagnosticWrapper):
    def __init__(self, workload_seed, arm="base"):
        provider = Wrapper66._provider
        cfg = MultiAgentConfig(
            env=EnvConfig(
                seed=workload_seed,
                scenario=SCENARIOS[SCENARIO],
                topology_provider=provider,
                n_planes=11,
                sats_per_plane=6,
            ),
            initial_packets=LOAD,
            exogenous_packets_per_slot=LOAD,
            seed=workload_seed,
            variant="qos_only",
        )
        super().__init__(scenario=SCENARIO, cfg=cfg, seed=workload_seed)
        if arm == "base":
            return
        env_cls = PurgeInfeasibleEnv if arm == "purge1" else LeastSlackEnv
        self.env = env_cls(self.env.cfg)

    _provider = None


def run_episode66(policy, label, policy_seed, workload, arm):
    wrapper = Wrapper66(workload, arm)
    result = evaluate_policy_with_constraint_metrics(
        SCENARIO, label, policy, policy_seed, [workload],
        wrapper_factory=lambda _: wrapper, variant="qos_only",
    )[0]
    row = asdict(result)
    row["arm"] = arm
    return row


def train_q_routing(identity, episodes):
    policy = QRoutingPolicy(n_nodes=66, seed=identity)
    rng = np.random.default_rng(identity)
    for episode in range(episodes):
        workload = int(rng.integers(76001, 76200))
        wrapper = Wrapper66(workload, "base")
        policy.bind(wrapper)
        observation, _ = wrapper.reset(seed=workload)
        terminated = truncated = False
        while not (terminated or truncated):
            actions = policy(observation, wrapper.get_avail_actions())
            observation, _, terminated, truncated, info = wrapper.step(actions)
            policy.observe_transition(info)
    return policy


def select_checkpoint(seed):
    runs = sorted((TRAIN_ROOT / f"seed_{seed}").glob("*/"))
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    for preferred in ("validation_best.pt", "final.pt"):
        p = latest / preferred
        if p.exists():
            return p
    raise FileNotFoundError(f"no checkpoint under {latest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    Wrapper66._provider = HypatiaTopologyProvider.from_csv(TOPOLOGY)
    checkpoints = {s: select_checkpoint(s) for s in SEEDS}
    write_json(output / "manifest.json", {
        "role": "tle66_feasibility_eval_dev",
        "runner_sha256": sha(Path(__file__)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topology": str(TOPOLOGY), "load": LOAD, "workloads": WORKLOADS,
        "system_checkpoints": {str(s): {"path": str(p), "sha256": sha(p)}
                               for s, p in checkpoints.items()},
        "q_identities": Q_IDENTITIES, "q_episodes": 500,
        "note": ("one-day feasibility signal; unconstrained QoS MAPPO trained in "
                 "the SRPF environment (the TLE trainer copy predates the switch "
                 "constraint); Dijkstra and Q-routing retrained budget-matched"),
        "training": False, "sealed_test_access": False,
    })
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    started = time.monotonic()
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for s, ckpt in checkpoints.items():
            policy, _ = load_checkpoint_policy(ckpt, device="cpu")
            for arm, label in (("base", f"mappo_{s}_base"),
                               ("purge_srpf", f"mappo_{s}_mech")):
                pol = DirectDeliveryPolicy(policy) if arm != "base" else policy
                for w in WORKLOADS:
                    row = run_episode66(pol, label, s, w, arm)
                    stream.write(canonical(row) + "\n")
                    rows.append(row)
                stream.flush()
                print(f"{len(rows)} {label} elapsed={time.monotonic()-started:.0f}s", flush=True)
        for identity in Q_IDENTITIES:
            t0 = time.monotonic()
            qp = train_q_routing(identity, 500)
            print(f"trained q_routing identity={identity} in {time.monotonic()-t0:.0f}s", flush=True)
            for arm, label in (("base", f"q_{identity}_base"),
                               ("purge_srpf", f"q_{identity}_mech")):
                for w in WORKLOADS:
                    row = run_episode66(qp, label, identity, w, arm)
                    stream.write(canonical(row) + "\n")
                    rows.append(row)
                stream.flush()
        for arm, label in (("base", "dijkstra_base"), ("purge_srpf", "dijkstra_mech")):
            for w in WORKLOADS:
                row = run_episode66(GlobalDijkstraPolicy(), label, -1, w, arm)
                stream.write(canonical(row) + "\n")
                rows.append(row)
            stream.flush()
    summary = {}
    for r in rows:
        label = r["policy"]
        stem = label.rsplit("_", 1)[0]
        kind = "mech" if label.endswith("_mech") else "base"
        summary.setdefault((stem, kind), []).append(r["delivery_ratio"])
    agg = {f"{stem}/{kind}": float(np.mean(v)) for (stem, kind), v in summary.items()}
    write_json(output / "summary.json", agg)
    write_json(output / "completion.json", {"status": "complete", "episodes": len(rows),
               "elapsed_seconds": time.monotonic() - started,
               "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    for k in sorted(agg):
        print("%-24s %.4f" % (k, agg[k]), flush=True)


if __name__ == "__main__":
    main()
