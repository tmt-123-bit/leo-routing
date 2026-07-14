"""CleanMARL adapter for the synchronous satellite-level environment."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np

from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MultiAgentConfig, SynchronousLeoMultiAgentEnv


class CleanMARLLeoMultiAgentWrapper:
    def __init__(
        self,
        scenario: str = "medium_load",
        cfg: Optional[MultiAgentConfig] = None,
    ):
        if cfg is None:
            env_cfg = EnvConfig(scenario=SCENARIOS[scenario])
            cfg = MultiAgentConfig(env=env_cfg)
        self.env = SynchronousLeoMultiAgentEnv(cfg)
        self.n_agents = self.env.n_agents
        self.max_degree = self.env.max_degree
        self.feature_dim = self.env.candidate_feature_dim
        self.action_size = self.env.action_size
        self._obs: Optional[list[Dict]] = None
        self._info: Dict = {}
        self._state_size: Optional[int] = None

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self._obs, self._info = self.env.reset()
        return self._obs_array(), self._info

    def step(self, actions):
        if self._obs is None:
            self.reset()
        action_list = np.asarray(actions).reshape(self.n_agents).astype(int).tolist()
        self._obs, rewards, terminated, truncated, self._info = self.env.step(
            action_list
        )
        # MAPPO uses one shared team reward; inactive agents remain excluded by
        # their forced NO_OP and can later receive an explicit policy-loss mask.
        reward = float(self._info["global_reward"])
        return self._obs_array(), reward, terminated, truncated, self._info

    def get_avail_actions(self) -> np.ndarray:
        if self._obs is None:
            self.reset()
        return np.asarray(
            [obs["action_mask"] for obs in self._obs], dtype=np.float32
        )

    def get_state(self) -> np.ndarray:
        state = self.env.global_state()
        flat = [float(state["slot_phase"])]
        flat.extend(x for row in state["node_features"] for x in row)
        flat.extend(x for row in state["hol_features"] for x in row)

        edge_map = {
            (
                int(round(row[0] * self.n_agents)),
                int(round(row[1] * self.n_agents)),
            ): row[2:]
            for row in state["edge_features"]
        }
        for u in range(1, self.n_agents + 1):
            for v in range(1, self.n_agents + 1):
                edge = edge_map.get((u, v))
                if edge is None:
                    flat.extend([0.0] * 5)
                else:
                    flat.extend([1.0, *edge])
        arr = np.asarray(flat, dtype=np.float32)
        if self._state_size is None:
            self._state_size = int(arr.size)
        return arr

    def get_obs_size(self) -> int:
        return self.action_size * self.feature_dim

    def get_state_size(self) -> int:
        if self._state_size is None:
            self.get_state()
        assert self._state_size is not None
        return self._state_size

    def get_action_size(self) -> int:
        return self.action_size

    def get_policy_active_mask(self) -> np.ndarray:
        if self._obs is None:
            self.reset()
        return np.asarray(
            [obs["hol_packet_id"] is not None for obs in self._obs],
            dtype=np.float32,
        )

    def get_candidate_feature_dim(self) -> int:
        return self.feature_dim

    def close(self) -> None:
        return None

    def _obs_array(self) -> np.ndarray:
        assert self._obs is not None
        rows = []
        for obs in self._obs:
            # Slot 0 is the explicit NO_OP action. It is only feasible when the
            # satellite has no HOL packet.
            candidates = [[0.0] * self.feature_dim]
            candidates.extend(obs["candidate_features"])
            rows.append([x for candidate in candidates for x in candidate])
        return np.asarray(rows, dtype=np.float32)


def smoke_test() -> None:
    env = CleanMARLLeoMultiAgentWrapper("medium_load")
    obs, _ = env.reset()
    avail = env.get_avail_actions()
    actions = np.asarray(
        [np.flatnonzero(mask)[0] for mask in avail], dtype=np.int64
    )
    next_obs, reward, terminated, truncated, info = env.step(actions)
    print(
        {
            "n_agents": env.n_agents,
            "obs_shape": tuple(obs.shape),
            "next_obs_shape": tuple(next_obs.shape),
            "state_shape": tuple(env.get_state().shape),
            "avail_shape": tuple(avail.shape),
            "concurrent_non_noop": info["concurrent_non_noop"],
            "reward": round(reward, 4),
            "terminated": terminated,
            "truncated": truncated,
        }
    )


if __name__ == "__main__":
    smoke_test()
