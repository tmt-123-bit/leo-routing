"""
CleanMARL-style rollout stub for the LEO routing project.

This file does NOT run PyTorch MAPPO. Instead, it mirrors the rollout data
collection style of F:/cleanmarl/cleanmarl/mappo.py as closely as possible,
so that when PyTorch is available later, the transition to the real CleanMARL
trainer is low-risk.

What it does now:
- instantiate CleanMARLLeoWrapper
- collect episode dictionaries with keys similar to cleanmarl/mappo.py
- save a small rollout preview CSV for inspection
- verify observation/state/action-mask shapes

Why it exists:
- current environment has no torch
- we still want to prove our wrapper and rollout schema are aligned with
  CleanMARL's expected training data layout
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from cleanmarl_leo_wrapper import CleanMARLLeoWrapper


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ROLLOUT_PREVIEW_PATH = OUTPUT_DIR / "cleanmarl_rollout_preview.csv"


@dataclass
class RolloutSummary:
    episode: int
    scenario: str
    steps: int
    delivered: int
    dropped: int
    total_reward: float
    final_status: str
    obs_shape: str
    state_shape: str
    avail_shape: str


def choose_first_valid(avail_actions: np.ndarray) -> np.ndarray:
    valid = np.where(avail_actions[0] > 0)[0]
    if len(valid) == 0:
        return np.asarray([-1], dtype=np.int64)
    return np.asarray([valid[0]], dtype=np.int64)


def collect_episode(env: CleanMARLLeoWrapper) -> tuple[Dict[str, List], RolloutSummary]:
    episode = {
        "obs": [],
        "actions": [],
        "log_prob": [],
        "reward": [],
        "states": [],
        "done": [],
        "avail_actions": [],
    }

    obs, info = env.reset()
    total_reward = 0.0
    done = False
    truncated = False
    steps = 0
    final_status = "not_started"

    while not done and not truncated and steps < env.env.cfg.max_local_hops:
        avail = env.get_avail_actions()
        state = env.get_state()
        action = choose_first_valid(avail)
        # Placeholder because real log_prob comes from the torch actor in cleanmarl.
        log_prob = np.asarray([0.0], dtype=np.float32)

        next_obs, reward, done, truncated, infos = env.step(action)
        episode["obs"].append(obs.copy())
        episode["actions"].append(action.copy())
        episode["log_prob"].append(log_prob.copy())
        episode["reward"].append(float(reward))
        episode["states"].append(state.copy())
        episode["done"].append(bool(done))
        episode["avail_actions"].append(avail.copy())

        total_reward += reward
        obs = next_obs
        steps += 1
        final_status = infos.get("status", "unknown")

    delivered = 1 if final_status == "delivered" else 0
    summary = RolloutSummary(
        episode=0,
        scenario=env.env.cfg.scenario.name,
        steps=steps,
        delivered=delivered,
        dropped=1 - delivered,
        total_reward=round(total_reward, 4),
        final_status=final_status,
        obs_shape=str(tuple(obs.shape)),
        state_shape=str(tuple(env.get_state().shape)),
        avail_shape=str(tuple(env.get_avail_actions().shape)),
    )
    return episode, summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: List[RolloutSummary] = []
    for idx, scenario in enumerate(["low_load", "medium_load", "hotspot_high_load", "frequent_break", "fault_links"], start=1):
        env = CleanMARLLeoWrapper(scenario=scenario)
        episode, summary = collect_episode(env)
        summary.episode = idx
        summaries.append(summary)
        print({
            "episode": idx,
            "scenario": scenario,
            "steps": summary.steps,
            "status": summary.final_status,
            "obs_len": len(episode["obs"]),
            "avail_shape": summary.avail_shape,
        })

    with ROLLOUT_PREVIEW_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(x) for x in summaries)

    print("Saved rollout preview:", ROLLOUT_PREVIEW_PATH)


if __name__ == "__main__":
    main()
