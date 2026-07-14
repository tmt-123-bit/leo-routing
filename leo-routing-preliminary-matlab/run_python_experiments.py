"""
Batch evaluation entry for the LEO routing environment.

This script is not the final MAPPO trainer. It provides the reproducible
experiment/evaluation loop needed before plugging in MAPPO/CleanMARL:
- run multiple scenarios
- compare hand-written heuristic policies and random feasible routing
- export CSV metrics used by the paper draft

Later, a trained MAPPO actor can be evaluated by replacing choose_action().
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from leo_marl_env import LeoRoutingEnv, SCENARIOS
from train_linear_policy import MODEL_PATH, MaskedLinearPolicy


POLICIES = [
    "random_feasible",
    "delay_only",
    "queue_load",
    "full_masked_heuristic",
    "linear_policy",
]

_LINEAR_POLICY = None


def load_linear_policy():
    global _LINEAR_POLICY
    if _LINEAR_POLICY is None and MODEL_PATH.exists():
        try:
            _LINEAR_POLICY = MaskedLinearPolicy.load(MODEL_PATH)
        except ValueError as exc:
            print(f"Ignoring incompatible linear policy: {exc}")
    return _LINEAR_POLICY


@dataclass
class EpisodeResult:
    scenario: str
    policy: str
    delivered: int
    dropped: int
    delay_ms: float
    hops: int
    total_reward: float
    control_overhead_bytes: int
    control_overhead_ratio: float
    decision_flops: int
    status: str


def choose_action(env: LeoRoutingEnv, obs: Dict, policy: str) -> int:
    neighbors = obs["neighbor_ids"]
    mask = obs["action_mask"]
    valid = [idx for idx, ok in enumerate(mask) if ok]
    if not valid:
        return -1
    if policy == "random_feasible":
        return env.rng.choice(valid)
    if policy == "linear_policy":
        learned = load_linear_policy()
        if learned is None:
            # If no trained weights exist yet, fall back to full heuristic.
            policy = "full_masked_heuristic"
        else:
            return learned.act(obs, greedy=True)

    current = obs["current"]
    dst = obs["dst"]
    best_idx = valid[0]
    best_cost = math.inf

    for idx in valid:
        nxt = neighbors[idx]
        edge = env.graph[(current, nxt)]
        q_next = env.queues[nxt] / env.cfg.q_max
        remaining_bw = max(0.0, env._remaining_bandwidth(current, nxt) / env.cfg.capacity_mbps)
        progress = env._progress_value(current, nxt, dst)
        lifetime_penalty = env.cfg.t_safe / max(env.cfg.t_safe, edge.t_rem)
        risk = 1.0 - edge.reliability

        if policy == "delay_only":
            cost = edge.delay_ms / env.cfg.d_ref_ms
        elif policy == "queue_load":
            cost = edge.delay_ms / env.cfg.d_ref_ms + 1.2 * q_next + 0.8 * edge.rho
        elif policy == "full_masked_heuristic":
            cost = (
                edge.delay_ms / env.cfg.d_ref_ms
                + 1.2 * q_next
                + 0.8 * edge.rho
                + 2.0 * risk
                + 1.0 * lifetime_penalty
                - 0.7 * progress
                - 0.2 * remaining_bw
            )
        else:
            raise ValueError(f"Unknown policy: {policy}")

        if cost < best_cost:
            best_cost = cost
            best_idx = idx
    return best_idx


def run_episode(scenario: str, policy: str, seed_offset: int) -> EpisodeResult:
    env = LeoRoutingEnv.from_scenario(scenario)
    env.rng.seed(env.cfg.seed + seed_offset)
    src, dst = env.sample_traffic_batch()[0]
    obs = env.reset(src=src, dst=dst)
    total_reward = 0.0
    total_delay = 0.0
    total_flops = 0
    info = {"status": "not_started", "control_overhead_ratio": 0.0, "control_overhead_bytes": 0}

    for _ in range(env.cfg.max_local_hops):
        action = choose_action(env, obs, policy)
        if action < 0:
            info = {"status": "no_valid_action", "control_overhead_ratio": env.control_overhead_ratio(), "control_overhead_bytes": env.control_bytes_acc}
            env.dropped_packets += 1
            break
        current = obs["current"]
        nxt = obs["neighbor_ids"][action]
        edge = env.graph[(current, nxt)]
        total_delay += edge.delay_ms + env.queues[nxt] / env.cfg.service_packets_per_slot + edge.rho * env.cfg.d_ref_ms
        total_flops += env.estimate_decision_flops(len(obs["neighbor_ids"]))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    delivered = 1 if info.get("status") == "delivered" else 0
    dropped = 1 - delivered
    return EpisodeResult(
        scenario=scenario,
        policy=policy,
        delivered=delivered,
        dropped=dropped,
        delay_ms=total_delay if delivered else math.inf,
        hops=info.get("hop_count", env.packet.hop_count if env.packet else 0),
        total_reward=total_reward,
        control_overhead_bytes=info.get("control_overhead_bytes", env.control_bytes_acc),
        control_overhead_ratio=info.get("control_overhead_ratio", env.control_overhead_ratio()),
        decision_flops=total_flops,
        status=info.get("status", "unknown"),
    )


def percentile(values: List[float], p: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return math.inf
    idx = 1 + (len(vals) - 1) * p / 100
    lo = math.floor(idx) - 1
    hi = math.ceil(idx) - 1
    if lo == hi:
        return vals[lo]
    return vals[lo] + (idx - math.floor(idx)) * (vals[hi] - vals[lo])


def aggregate(results: List[EpisodeResult]) -> List[Dict]:
    rows = []
    groups: Dict[Tuple[str, str], List[EpisodeResult]] = {}
    for r in results:
        groups.setdefault((r.scenario, r.policy), []).append(r)
    for (scenario, policy), items in sorted(groups.items()):
        delays = [x.delay_ms for x in items if x.delivered]
        rows.append({
            "scenario": scenario,
            "policy": policy,
            "episodes": len(items),
            "delivered": sum(x.delivered for x in items),
            "dropped": sum(x.dropped for x in items),
            "deliveryRatio": sum(x.delivered for x in items) / len(items),
            "dropRate": sum(x.dropped for x in items) / len(items),
            "avgDelayMs": mean(delays) if delays else math.inf,
            "p95DelayMs": percentile(delays, 95),
            "avgHops": mean(x.hops for x in items),
            "avgReward": mean(x.total_reward for x in items),
            "avgControlOverheadBytes": mean(x.control_overhead_bytes for x in items),
            "avgControlOverheadRatio": mean(x.control_overhead_ratio for x in items),
            "avgDecisionFLOPs": mean(x.decision_flops for x in items),
        })
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    episodes_per_pair = 50
    results: List[EpisodeResult] = []
    for scenario in SCENARIOS:
        for policy in POLICIES:
            for ep in range(episodes_per_pair):
                results.append(run_episode(scenario, policy, ep))

    rows = aggregate(results)
    out_dir = Path(__file__).resolve().parent / "outputs"
    write_csv(out_dir / "python_policy_eval_metrics.csv", rows)

    print("Generated:", out_dir / "python_policy_eval_metrics.csv")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
