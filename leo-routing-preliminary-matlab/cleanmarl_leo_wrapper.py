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

This wrapper is the revised single-active-packet PPO baseline. It deliberately
reports n_agents=1; the synchronous satellite-level MAPPO environment lives in
leo_multiagent_env.py and must be used for multi-agent/CTDE claims.

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
        if max_degree != self.env.max_degree:
            raise ValueError(
                f"max_degree={max_degree} does not match environment schema "
                f"{self.env.max_degree}"
            )
        self.max_degree = self.env.max_degree
        self.feature_dim = self.env.candidate_feature_dim
        self._obs: Optional[Dict] = None
        self._last_info: Dict = {}
        self._state_size: Optional[int] = None

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
        arr = np.asarray(
            self.env.as_mappo_inputs(self._obs)["critic_state"],
            dtype=np.float32,
        )
        if self._state_size is None:
            self._state_size = int(arr.size)
        return arr

    def get_obs_size(self) -> int:
        return self.max_degree * self.feature_dim

    def get_state_size(self) -> int:
        if self._state_size is None:
            self.get_state()
        assert self._state_size is not None
        return self._state_size

    def get_action_size(self) -> int:
        return self.max_degree

    def get_policy_active_mask(self) -> np.ndarray:
        return np.ones((self.n_agents,), dtype=np.float32)

    def get_candidate_feature_dim(self) -> int:
        return self.feature_dim

    def get_candidate_shape(self) -> Tuple[int, int]:
        return self.max_degree, self.feature_dim

    def close(self) -> None:
        return None

    def _obs_array(self) -> np.ndarray:
        assert self._obs is not None
        mappo_inputs = self.env.as_mappo_inputs(self._obs)
        padded = np.asarray(mappo_inputs["candidate_obs"], dtype=np.float32)
        padded = padded.reshape(self.n_agents, self.max_degree, self.feature_dim)
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
