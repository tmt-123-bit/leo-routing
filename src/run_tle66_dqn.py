"""66-sat DQN baseline: train queue-aware neural Q-routing at load 12 and
evaluate with and without the packet-management stack on the confirmation
panel, so the 'strongest baseline' reference includes a scalable learned
method, not only Dijkstra."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from dqn_baseline import DqnRoutingPolicy
from hypatia_topology_provider_stub import HypatiaTopologyProvider
from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MultiAgentConfig
from mappo_evaluation import evaluate_policy_with_constraint_metrics
from run_development_load_diagnostics import canonical, sha, write_json
from run_srpf_probe import LeastSlackEnv
from run_development_load_diagnostics import DiagnosticWrapper

TOPOLOGY = Path("../data/starlink_66_links.csv")
LOAD = 12
IDENTITIES = [7, 42]
TRAIN_EPISODES = 500
WORKLOADS = list(range(880041, 880081))
SCENARIO = "hotspot_high_load"


class W66(DiagnosticWrapper):
    _provider = None

    def __init__(self, workload_seed, arm="base"):
        cfg = MultiAgentConfig(
            env=EnvConfig(seed=workload_seed, scenario=SCENARIOS[SCENARIO],
                          topology_provider=W66._provider, n_planes=11,
                          sats_per_plane=6),
            initial_packets=LOAD, exogenous_packets_per_slot=LOAD,
            seed=workload_seed, variant="qos_only",
        )
        super().__init__(scenario=SCENARIO, cfg=cfg, seed=workload_seed)
        if arm != "base":
            self.env = LeastSlackEnv(self.env.cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    W66._provider = HypatiaTopologyProvider.from_csv(TOPOLOGY)
    write_json(output / "manifest.json", {
        "role": "tle66_dqn_baseline_dev",
        "runner_sha256": sha(Path(__file__)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topology": str(TOPOLOGY), "load": LOAD, "identities": IDENTITIES,
        "train_episodes": TRAIN_EPISODES, "workloads": WORKLOADS,
        "training": True, "sealed_test_access": False,
    })
    torch.set_num_threads(1)
    rows = []
    started = time.monotonic()
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for identity in IDENTITIES:
            t0 = time.monotonic()
            policy = DqnRoutingPolicy(seed=identity)
            rng = np.random.default_rng(identity)
            for episode in range(TRAIN_EPISODES):
                workload = int(rng.integers(76001, 76200))
                wrapper = W66(workload, "base")
                policy.on_episode_start()
                observation, _ = wrapper.reset(seed=workload)
                policy.bind(wrapper)
                terminated = truncated = False
                while not (terminated or truncated):
                    actions = policy(observation, wrapper.get_avail_actions())
                    observation, _, terminated, truncated, info = wrapper.step(actions)
                    policy.observe_transition(info)
                if episode % 100 == 0:
                    print(f"identity {identity}: episode {episode} "
                          f"({time.monotonic()-t0:.0f}s)", flush=True)
            policy.freeze()
            print(f"trained dqn66 identity={identity} in {time.monotonic()-t0:.0f}s", flush=True)
            for arm, label in (("base", f"dqn_{identity}_base"),
                               ("purge_srpf", f"dqn_{identity}_mech")):
                for w in WORKLOADS:
                    wrapper = W66(w, arm)
                    result = evaluate_policy_with_constraint_metrics(
                        SCENARIO, label, policy, identity, [w],
                        wrapper_factory=lambda _: wrapper, variant="qos_only",
                    )[0]
                    row = dict(result.__dict__) if hasattr(result, "__dict__") else dict(result)
                    from dataclasses import asdict, is_dataclass
                    if is_dataclass(result):
                        row = asdict(result)
                    row["arm"] = arm
                    stream.write(canonical(row) + "\n")
                    rows.append(row)
                stream.flush()
    summary = {}
    for r in rows:
        stem = r["policy"].rsplit("_", 1)[0]
        kind = "mech" if r["policy"].endswith("_mech") else "base"
        summary.setdefault((stem, kind), []).append(r["delivery_ratio"])
    agg = {f"{s}/{k}": float(np.mean(v)) for (s, k), v in summary.items()}
    write_json(output / "summary.json", agg)
    write_json(output / "completion.json", {"status": "complete", "episodes": len(rows),
               "elapsed_seconds": time.monotonic() - started,
               "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    for k in sorted(agg):
        print("%-24s %.4f" % (k, agg[k]), flush=True)


if __name__ == "__main__":
    main()
