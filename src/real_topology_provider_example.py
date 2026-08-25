"""
Example real/near-real constellation topology provider for LeoRoutingEnv.

Purpose:
- define the exact return format expected by EnvConfig.topology_provider
- show how external topology data (TLE / Hypatia / StarryNet / custom snapshot)
  should be converted into the environment's LinkState dictionary
- avoid later refactors when replacing the toy topology

This file is a TEMPLATE, not a full TLE/SGP4 parser.
"""

from __future__ import annotations

from typing import Dict, Tuple

from leo_marl_env import LeoRoutingEnv, LinkState


def example_topology_provider(t: int, env: LeoRoutingEnv) -> Dict[Tuple[int, int], LinkState]:
    """Return a minimal topology snapshot in the format LeoRoutingEnv expects.

    Required contract:
    - return type: Dict[(src, dst), LinkState]
    - keys are directed edges
    - values must include delay/capacity/used_rate/rho/reliability/p_out/t_rem/available
    - if an edge is absent, it is treated as unavailable

    This example builds a tiny 4-node ring-like topology just to document the
    expected shape. In a real implementation, replace this with topology loaded
    from TLE/Hypatia/StarryNet outputs.
    """
    g: Dict[Tuple[int, int], LinkState] = {}

    def add_bi(u: int, v: int, delay_ms: float, reliability: float, t_rem: float, is_cross: bool = False) -> None:
        for a, b in [(u, v), (v, u)]:
            used = env.used_rate[a][b]
            rho = min(1.0, used / env.cfg.capacity_mbps)
            rel = min(reliability, max(0.0, reliability - 0.05 * rho))
            g[(a, b)] = LinkState(
                delay_ms=delay_ms,
                capacity_mbps=env.cfg.capacity_mbps,
                used_rate_mbps=used,
                rho=rho,
                reliability=rel,
                p_out=1.0 - rel,
                t_rem=t_rem,
                available=True,
                is_cross=is_cross,
                shell_src=env.cfg.shell_id,
                shell_dst=env.cfg.shell_id,
            )

    # Example only. A real provider would usually map satellite IDs from a
    # constellation snapshot file and compute delay / reliability / lifetime from
    # the external source.
    add_bi(1, 2, delay_ms=8.2, reliability=0.98, t_rem=999.0)
    add_bi(2, 3, delay_ms=8.5, reliability=0.97, t_rem=999.0)
    add_bi(3, 4, delay_ms=12.0, reliability=0.94, t_rem=4.0, is_cross=True)
    add_bi(4, 1, delay_ms=11.7, reliability=0.95, t_rem=5.0, is_cross=True)

    return g


REAL_TOPOLOGY_FIELD_GUIDE = """
Expected minimal per-link fields when converting external topology data:

(src, dst) -> LinkState(
    delay_ms=float,          # propagation (or propagation+transmission proxy)
    capacity_mbps=float,     # residual or nominal capacity
    used_rate_mbps=float,    # current load already allocated on this link
    rho=float,               # load ratio in [0,1]
    reliability=float,       # success probability in [0,1]
    p_out=float,             # outage/failure probability in [0,1]
    t_rem=float,             # remaining usable time in seconds or slot-equivalent
    available=bool,          # whether the link exists now
    is_cross=bool,           # optional: cross-plane / cross-shell hint
    shell_src=int,           # optional shell id
    shell_dst=int,           # optional shell id
)
"""


if __name__ == "__main__":
    print("This file is a topology-provider template, not a standalone simulator.")
    print(REAL_TOPOLOGY_FIELD_GUIDE)
