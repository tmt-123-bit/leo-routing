"""CleanMARL wrapper for training directly on frozen TLE link snapshots."""

from __future__ import annotations

import os
from pathlib import Path

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from hypatia_topology_provider_stub import HypatiaTopologyProvider
from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MULTIAGENT_LOADS, MultiAgentConfig


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


class CleanMARLTLEWrapper(CleanMARLLeoMultiAgentWrapper):
    """TLE-snapshot wrapper with optional environment-variable overrides.

    LEO_TLE_N_PLANES / LEO_TLE_SATS_PER_PLANE override the constellation
    geometry derived from the scenario preset (required when the snapshot
    contains more satellites than the preset's 4x6 grid). LEO_TLE_INITIAL /
    LEO_TLE_EXOGENOUS override the traffic loads. Unset variables keep the
    stock scenario behavior, so frozen runs are unaffected.
    """

    def __init__(
        self,
        topology_csv: str | Path,
        scenario: str = "medium_load",
        seed: int = 11,
        variant: str = "proposed",
    ):
        provider = HypatiaTopologyProvider.from_csv(topology_csv)
        env_cfg = EnvConfig(
            seed=seed,
            scenario=SCENARIOS[scenario],
            topology_provider=provider,
        )
        n_planes = _env_int("LEO_TLE_N_PLANES")
        sats_per_plane = _env_int("LEO_TLE_SATS_PER_PLANE")
        if n_planes is not None:
            env_cfg.n_planes = n_planes
        if sats_per_plane is not None:
            env_cfg.sats_per_plane = sats_per_plane
        initial_packets, exogenous_packets = MULTIAGENT_LOADS[scenario]
        initial_override = _env_int("LEO_TLE_INITIAL")
        exogenous_override = _env_int("LEO_TLE_EXOGENOUS")
        if initial_override is not None:
            initial_packets = initial_override
        if exogenous_override is not None:
            exogenous_packets = exogenous_override
        cfg = MultiAgentConfig(
            env=env_cfg,
            initial_packets=initial_packets,
            exogenous_packets_per_slot=exogenous_packets,
            seed=seed,
            variant=variant,
        )
        super().__init__(scenario=scenario, cfg=cfg, seed=seed)
