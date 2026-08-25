"""CleanMARL adapter for the synchronous satellite-level environment."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence, Tuple
import numpy as np

from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import (
    MULTIAGENT_LOADS,
    MultiAgentConfig,
    SynchronousLeoMultiAgentEnv,
)


def _apply_reward_overrides(env_cfg: EnvConfig) -> None:
    """Apply reward-weight overrides from the LEO_REWARD_OVERRIDES env var.

    The var holds a JSON object mapping EnvConfig reward-weight field names
    (w_deliver, w_delay, w_load, w_queue, w_switch, ...) to numeric values.
    Backward-compatible: unset / empty -> no change. Used by the reward-sensitivity
    sweep; training-only (eval metrics are reward-independent).
    """
    raw = os.environ.get("LEO_REWARD_OVERRIDES", "").strip()
    if not raw:
        return
    try:
        overrides = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LEO_REWARD_OVERRIDES is not valid JSON: {e}") from e
    for k, v in overrides.items():
        if hasattr(env_cfg, k) and isinstance(v, (int, float)) and not isinstance(v, bool):
            setattr(env_cfg, k, float(v))


class CleanMARLLeoMultiAgentWrapper:
    def __init__(
        self,
        scenario: str = "medium_load",
        cfg: Optional[MultiAgentConfig] = None,
        seed: Optional[int] = None,
        agent_permutation: Optional[Sequence[int]] = None,
        variant: str = "full",
    ):
        if cfg is None:
            effective_seed = 11 if seed is None else seed
            env_cfg = EnvConfig(seed=effective_seed, scenario=SCENARIOS[scenario])
            _apply_reward_overrides(env_cfg)
            initial_packets, exogenous_packets = MULTIAGENT_LOADS[scenario]
            cfg = MultiAgentConfig(
                env=env_cfg,
                initial_packets=initial_packets,
                exogenous_packets_per_slot=exogenous_packets,
                seed=effective_seed,
                variant=variant,
            )
        elif seed is not None:
            cfg.seed = seed
            cfg.env.seed = seed
        self.env = SynchronousLeoMultiAgentEnv(cfg)
        self.n_agents = self.env.n_agents
        self.max_degree = self.env.max_degree
        self.feature_dim = self.env.candidate_feature_dim
        self.action_size = self.env.action_size
        self._obs: Optional[list[Dict]] = None
        self._info: Dict = {}
        self._state_size: Optional[int] = None
        self.variant = cfg.variant
        if agent_permutation is None:
            agent_permutation = list(range(1, self.n_agents + 1))
        self.external_to_internal = [int(x) for x in agent_permutation]
        if sorted(self.external_to_internal) != list(range(1, self.n_agents + 1)):
            raise ValueError("agent_permutation must contain every satellite exactly once")

    def reset(
        self,
        seed: Optional[int] = None,
        initial_pairs=None,
    ) -> Tuple[np.ndarray, Dict]:
        self._obs, self._info = self.env.reset(
            seed=seed, initial_pairs=initial_pairs
        )
        return self._obs_array(), self._info

    def step(self, actions):
        if self._obs is None:
            self.reset()
        external_actions = (
            np.asarray(actions).reshape(self.n_agents).astype(int).tolist()
        )
        action_list = [0 for _ in range(self.n_agents)]
        for external_index, internal_sat in enumerate(self.external_to_internal):
            action_list[internal_sat - 1] = external_actions[external_index]
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
            [
                self._obs[internal_sat - 1]["action_mask"]
                for internal_sat in self.external_to_internal
            ],
            dtype=np.float32,
        )

    def get_state(self) -> np.ndarray:
        state = self.env.global_state()
        flat = [float(x) for x in state["global_features"]]
        for internal_sat in self.external_to_internal:
            flat.extend(state["node_features"][internal_sat - 1])
        for internal_src in self.external_to_internal:
            for internal_dst in self.external_to_internal:
                flat.extend(
                    state["edge_features"][internal_src - 1][internal_dst - 1]
                )
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

    def get_critic_spec(self) -> Dict:
        if self.variant == "flat_critic":
            return None
        state = self.env.global_state()
        return {"type": "graph_attention", **state["schema"]}

    def get_last_agent_rewards(self) -> np.ndarray:
        internal_rewards = self._info.get(
            "agent_rewards", [self._info.get("global_reward", 0.0)] * self.n_agents
        )
        return np.asarray(
            [internal_rewards[sat - 1] for sat in self.external_to_internal],
            dtype=np.float32,
        )

    def get_policy_active_mask(self) -> np.ndarray:
        if self._obs is None:
            self.reset()
        return np.asarray(
            [
                self._obs[internal_sat - 1]["hol_packet_id"] is not None
                and any(
                    self._obs[internal_sat - 1]["action_mask"][1:]
                )
                for internal_sat in self.external_to_internal
            ],
            dtype=np.float32,
        )

    def get_candidate_feature_dim(self) -> int:
        return self.feature_dim

    def close(self) -> None:
        return None

    def _obs_array(self) -> np.ndarray:
        assert self._obs is not None
        rows = []
        for internal_sat in self.external_to_internal:
            obs = self._obs[internal_sat - 1]
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
