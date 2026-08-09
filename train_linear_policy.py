"""
Lightweight learning baseline for LEO local next-hop routing.

This is a small linear control baseline, not the MAPPO implementation. It uses
a masked linear softmax policy with a REINFORCE-style update.

Why it exists:
- proves the environment supports learning-style rollouts
- saves a learned policy that can be evaluated with run_python_experiments.py
- keeps the interface close to later MAPPO/CleanMARL integration

Outputs:
- models/linear_policy_weights.npz
- outputs/linear_policy_training_log.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from leo_marl_env import LeoRoutingEnv, SCENARIOS


FEATURE_DIM = LeoRoutingEnv.candidate_feature_dim
MODEL_DIR = Path(__file__).resolve().parent / "models"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_PATH = MODEL_DIR / "linear_policy_weights.npz"
LOG_PATH = OUTPUT_DIR / "linear_policy_training_log.csv"


class MaskedLinearPolicy:
    def __init__(self, feature_dim: int = FEATURE_DIM, seed: int = 11):
        self.rng = np.random.default_rng(seed)
        self.w = self.rng.normal(0.0, 0.05, size=(feature_dim,))
        self.b = 0.0

    def scores(self, features: List[List[float]]) -> np.ndarray:
        if not features:
            return np.zeros((0,), dtype=float)
        x = np.asarray(features, dtype=float)
        return x @ self.w + self.b

    def probs(self, features: List[List[float]], mask: List[bool]) -> np.ndarray:
        scores = self.scores(features)
        valid = np.asarray(mask, dtype=bool)
        if scores.size == 0 or not valid.any():
            return np.zeros_like(scores)
        masked_scores = np.where(valid, scores, -1e9)
        masked_scores = masked_scores - np.max(masked_scores)
        exp_scores = np.exp(masked_scores) * valid
        return exp_scores / max(1e-12, exp_scores.sum())

    def act(self, obs: Dict, greedy: bool = False) -> int:
        probs = self.probs(obs["neighbor_features"], obs["action_mask"])
        if probs.size == 0 or probs.sum() <= 0:
            return -1
        if greedy:
            return int(np.argmax(probs))
        return int(self.rng.choice(len(probs), p=probs))

    def update(self, trajectory: List[Tuple[List[List[float]], List[bool], int]], returns: List[float], lr: float) -> None:
        grad_w = np.zeros_like(self.w)
        grad_b = 0.0
        for (features, mask, action), ret in zip(trajectory, returns):
            if action < 0:
                continue
            x = np.asarray(features, dtype=float)
            probs = self.probs(features, mask)
            if probs.size == 0 or probs.sum() <= 0:
                continue
            one_hot = np.zeros_like(probs)
            one_hot[action] = 1.0
            # grad log pi(a|s) for linear softmax.
            grad_scores = one_hot - probs
            grad_w += ret * (grad_scores[:, None] * x).sum(axis=0)
            grad_b += ret * grad_scores.sum()
        self.w += lr * grad_w
        self.b += lr * grad_b

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w=self.w, b=np.asarray([self.b]))

    @classmethod
    def load(
        cls, path: Path, expected_feature_dim: int = FEATURE_DIM
    ) -> "MaskedLinearPolicy":
        data = np.load(path)
        if len(data["w"]) != expected_feature_dim:
            raise ValueError(
                f"saved linear policy has {len(data['w'])} features; "
                f"environment expects {expected_feature_dim}. Retrain it."
            )
        obj = cls(feature_dim=len(data["w"]))
        obj.w = data["w"].astype(float)
        obj.b = float(data["b"][0])
        return obj


def discounted_returns(rewards: List[float], gamma: float = 0.95) -> List[float]:
    out = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        out.append(running)
    out.reverse()
    if len(out) > 1:
        arr = np.asarray(out, dtype=float)
        std = arr.std()
        if std > 1e-8:
            arr = (arr - arr.mean()) / std
        out = arr.tolist()
    return out


def run_training_episode(policy: MaskedLinearPolicy, scenario: str, seed_offset: int, greedy: bool = False):
    env = LeoRoutingEnv.from_scenario(scenario)
    env.rng.seed(env.cfg.seed + seed_offset)
    src, dst = env.sample_traffic_batch()[0]
    obs = env.reset(src=src, dst=dst)
    trajectory = []
    rewards = []
    total_reward = 0.0
    info = {"status": "not_started", "hop_count": 0}

    for _ in range(env.cfg.max_local_hops):
        action = policy.act(obs, greedy=greedy)
        trajectory.append((obs["neighbor_features"], obs["action_mask"], action))
        if action < 0:
            rewards.append(-env.cfg.w_invalid)
            info = {"status": "no_valid_action", "hop_count": env.packet.hop_count if env.packet else 0}
            break
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        total_reward += reward
        if terminated or truncated:
            break

    delivered = 1 if info.get("status") == "delivered" else 0
    return trajectory, rewards, {
        "scenario": scenario,
        "status": info.get("status", "unknown"),
        "delivered": delivered,
        "dropped": 1 - delivered,
        "hops": info.get("hop_count", 0),
        "total_reward": total_reward,
        "control_overhead_ratio": info.get("control_overhead_ratio", env.control_overhead_ratio()),
    }


def train() -> None:
    policy = MaskedLinearPolicy(seed=11)
    scenarios = list(SCENARIOS.keys())
    episodes = 800
    lr = 0.015
    gamma = 0.95
    log_rows = []

    for ep in range(episodes):
        scenario = scenarios[ep % len(scenarios)]
        traj, rewards, result = run_training_episode(policy, scenario, seed_offset=ep, greedy=False)
        returns = discounted_returns(rewards, gamma=gamma)
        policy.update(traj, returns, lr=lr)
        result["episode"] = ep + 1
        result["phase"] = "train"
        log_rows.append(result)

        if (ep + 1) % 100 == 0:
            eval_rows = []
            for s in scenarios:
                _, _, eval_result = run_training_episode(policy, s, seed_offset=10_000 + ep, greedy=True)
                eval_result["episode"] = ep + 1
                eval_result["phase"] = "eval"
                eval_rows.append(eval_result)
                log_rows.append(eval_result)
            delivery = sum(r["delivered"] for r in eval_rows) / len(eval_rows)
            avg_reward = sum(r["total_reward"] for r in eval_rows) / len(eval_rows)
            print(f"episode={ep+1} eval_delivery={delivery:.3f} eval_reward={avg_reward:.3f}")

    policy.save(MODEL_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["episode", "phase", "scenario", "status", "delivered", "dropped", "hops", "total_reward", "control_overhead_ratio"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)
    print("Saved model:", MODEL_PATH)
    print("Saved log:", LOG_PATH)


if __name__ == "__main__":
    train()
