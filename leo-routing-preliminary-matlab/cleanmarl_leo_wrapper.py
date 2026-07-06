"""
CleanMARL-style wrapper for LeoRoutingEnv.

This file is written after reading F:/cleanmarl/cleanmarl/mappo.py.
CleanMARL MAPPO expects an environment object with:
- n_agents
- reset() -> obs, info
- step(actions) -> next_obs, reward, done, truncated, infos
- get_avail_actions()
- get_state()
- get_obs_size()
- get_state_size()
- get_action_size()

The current LEO environment is a single-packet local next-hop environment, so
this wrapper exposes it as a one-agent cooperative MARL task first. This is the
least risky bridge: it keeps the LEO routing environment correct, while making
its data layout compatible with cleanmarl MAPPO/IPPO scripts.

Later extension:
- multiple simultaneous packets can be mapped to multiple agents;
- shared reward can be replaced by R_global;
- this wrapper can be moved into cleanmarl/env/ once dependencies are ready.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np

from leo_marl_env import LeoRoutingEnv, EnvConfig, SCENARIOS


class CleanMARLLeoWrapper:
    def __init__(self, scenario: str = "medium_load", cfg: Optional[EnvConfig] = None, max_degree: int = 6):
        if cfg is None:
            cfg = EnvConfig(scenario=SCENARIOS[scenario])
        self.env = LeoRoutingEnv(cfg)
        self.n_agents = 1
        self.max_degree = max_degree
        self.feature_dim = 11
        self._obs: Optional[Dict] = None
        self._last_info: Dict = {}

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self._obs = self.env.reset()
        self._last_info = {"scenario": self.env.cfg.scenario.name}
        return self._obs_array(), self._last_info

    def step(self, actions):
        if self._obs is None:
            self.reset()
        action = int(np.asarray(actions).reshape(-1)[0])
        self._obs, reward, done, truncated, info = self.env.step(action)
        self._last_info = info
        return self._obs_array(), float(reward), bool(done), bool(truncated), info

    def get_avail_actions(self) -> np.ndarray:
        if self._obs is None:
            self.reset()
        mask = np.zeros((self.n_agents, self.max_degree), dtype=np.float32)
        for i, ok in enumerate(self._obs["action_mask"][: self.max_degree]):
            mask[0, i] = 1.0 if ok else 0.0
        return mask

    def get_state(self) -> np.ndarray:
        if self._obs is None:
            self.reset()
        state = self._obs["global_state_for_critic"]
        arr = np.asarray([
            state["avg_queue"],
            state["max_queue"],
            state["avg_rho"],
            state["jain_load"],
            state["num_available_links"] / max(1, self.env.cfg.n_sats * 4),
            state["control_overhead_ratio"],
            state["delivery_ratio"],
            state["drop_rate"],
            state.get("switch_count", 0),
        ], dtype=np.float32)
        return arr

    def get_obs_size(self) -> int:
        return self.max_degree * self.feature_dim

    def get_state_size(self) -> int:
        return 9

    def get_action_size(self) -> int:
        return self.max_degree

    def _obs_array(self) -> np.ndarray:
        assert self._obs is not None
        padded = np.zeros((self.n_agents, self.max_degree, self.feature_dim), dtype=np.float32)
        for i, feat in enumerate(self._obs["neighbor_features"][: self.max_degree]):
            padded[0, i, :] = np.asarray(feat, dtype=np.float32)
        return padded.reshape(self.n_agents, -1)


def smoke_test() -> None:
    env = CleanMARLLeoWrapper("medium_load")
    obs, info = env.reset()
    total_reward = 0.0
    done = False
    truncated = False
    steps = 0
    while not done and not truncated and steps < 20:
        avail = env.get_avail_actions()
        valid = np.where(avail[0] > 0)[0]
        if len(valid) == 0:
            break
        obs, reward, done, truncated, info = env.step(np.asarray([valid[0]]))
        total_reward += reward
        steps += 1
    print({
        "obs_shape": tuple(obs.shape),
        "state_shape": tuple(env.get_state().shape),
        "avail_shape": tuple(env.get_avail_actions().shape),
        "action_size": env.get_action_size(),
        "steps": steps,
        "reward": round(total_reward, 4),
        "status": info.get("status"),
    })


if __name__ == "__main__":
    smoke_test()
