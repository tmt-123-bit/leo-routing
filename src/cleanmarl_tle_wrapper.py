"""CleanMARL wrapper for training directly on frozen TLE link snapshots."""

from __future__ import annotations

from pathlib import Path

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from hypatia_topology_provider_stub import HypatiaTopologyProvider
from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MULTIAGENT_LOADS, MultiAgentConfig


class CleanMARLTLEWrapper(CleanMARLLeoMultiAgentWrapper):
    def __init__(
        self,
        topology_csv: str | Path,
        scenario: str = "medium_load",
        seed: int = 11,
        variant: str = "full",
    ):
        provider = HypatiaTopologyProvider.from_csv(topology_csv)
        env_cfg = EnvConfig(
            seed=seed,
            scenario=SCENARIOS[scenario],
            topology_provider=provider,
        )
        initial_packets, exogenous_packets = MULTIAGENT_LOADS[scenario]
        cfg = MultiAgentConfig(
            env=env_cfg,
            initial_packets=initial_packets,
            exogenous_packets_per_slot=exogenous_packets,
            seed=seed,
            variant=variant,
        )
        super().__init__(scenario=scenario, cfg=cfg, seed=seed)
