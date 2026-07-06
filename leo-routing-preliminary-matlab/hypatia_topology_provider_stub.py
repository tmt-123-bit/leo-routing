"""
Hypatia-style topology provider stub for LeoRoutingEnv.

This file does NOT depend on Hypatia directly. It is a bridge template showing
how time-sliced topology exports from Hypatia (or a similar simulator) can be
converted into the LinkState dictionary expected by leo_marl_env.py.

Design goal:
- keep all future real-topology logic in one place
- avoid rewriting LeoRoutingEnv when switching from toy topology to quasi-real
  constellation snapshots
- make the required input schema explicit
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from leo_marl_env import LeoRoutingEnv, LinkState


@dataclass
class LinkSnapshot:
    time_slot: int
    src: int
    dst: int
    delay_ms: float
    available: bool
    capacity_mbps: float = 100.0
    reliability: float = 0.98
    t_rem: float = 999.0
    is_cross: bool = False
    shell_src: int = 1
    shell_dst: int = 1


def read_link_snapshots_csv(path: str | Path) -> List[LinkSnapshot]:
    """Read a CSV export into LinkSnapshot records.

    Expected columns (minimum):
    - time_slot
    - src
    - dst
    - delay_ms
    - available

    Optional columns:
    - capacity_mbps
    - reliability
    - t_rem
    - is_cross
    - shell_src
    - shell_dst
    """
    path = Path(path)
    rows: List[LinkSnapshot] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                LinkSnapshot(
                    time_slot=int(row["time_slot"]),
                    src=int(row["src"]),
                    dst=int(row["dst"]),
                    delay_ms=float(row["delay_ms"]),
                    available=_to_bool(row["available"]),
                    capacity_mbps=float(row.get("capacity_mbps", 100.0)),
                    reliability=float(row.get("reliability", 0.98)),
                    t_rem=float(row.get("t_rem", 999.0)),
                    is_cross=_to_bool(row.get("is_cross", "False")),
                    shell_src=int(row.get("shell_src", 1)),
                    shell_dst=int(row.get("shell_dst", 1)),
                )
            )
    return rows


def build_snapshot_index(rows: Iterable[LinkSnapshot]) -> Dict[int, List[LinkSnapshot]]:
    index: Dict[int, List[LinkSnapshot]] = {}
    for row in rows:
        index.setdefault(row.time_slot, []).append(row)
    return index


class HypatiaTopologyProvider:
    """Callable provider that can be plugged into EnvConfig(topology_provider=...).

    Usage later:

        provider = HypatiaTopologyProvider.from_csv("hypatia_links.csv")
        cfg = EnvConfig(topology_provider=provider)
        env = LeoRoutingEnv(cfg)
    """

    def __init__(self, snapshot_index: Dict[int, List[LinkSnapshot]]):
        self.snapshot_index = snapshot_index

    @classmethod
    def from_csv(cls, path: str | Path) -> "HypatiaTopologyProvider":
        rows = read_link_snapshots_csv(path)
        return cls(build_snapshot_index(rows))

    def __call__(self, t: int, env: LeoRoutingEnv) -> Dict[Tuple[int, int], LinkState]:
        graph: Dict[Tuple[int, int], LinkState] = {}
        rows = self.snapshot_index.get(t, [])
        for row in rows:
            if not row.available:
                continue
            used = env.used_rate[row.src][row.dst]
            rho = min(1.0, used / max(1e-6, row.capacity_mbps))
            reliability = max(0.0, min(1.0, row.reliability - 0.05 * rho))
            graph[(row.src, row.dst)] = LinkState(
                delay_ms=row.delay_ms,
                capacity_mbps=row.capacity_mbps,
                used_rate_mbps=used,
                rho=rho,
                reliability=reliability,
                p_out=1.0 - reliability,
                t_rem=row.t_rem,
                available=True,
                is_cross=row.is_cross,
                shell_src=row.shell_src,
                shell_dst=row.shell_dst,
            )
        return graph


def write_demo_csv(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_slot", "src", "dst", "delay_ms", "available",
        "capacity_mbps", "reliability", "t_rem", "is_cross", "shell_src", "shell_dst",
    ]
    demo_rows = [
        {"time_slot": 1, "src": 1, "dst": 2, "delay_ms": 8.2, "available": True, "capacity_mbps": 100.0, "reliability": 0.98, "t_rem": 999.0, "is_cross": False, "shell_src": 1, "shell_dst": 1},
        {"time_slot": 1, "src": 2, "dst": 1, "delay_ms": 8.2, "available": True, "capacity_mbps": 100.0, "reliability": 0.98, "t_rem": 999.0, "is_cross": False, "shell_src": 1, "shell_dst": 1},
        {"time_slot": 1, "src": 2, "dst": 3, "delay_ms": 8.5, "available": True, "capacity_mbps": 100.0, "reliability": 0.97, "t_rem": 999.0, "is_cross": False, "shell_src": 1, "shell_dst": 1},
        {"time_slot": 1, "src": 3, "dst": 2, "delay_ms": 8.5, "available": True, "capacity_mbps": 100.0, "reliability": 0.97, "t_rem": 999.0, "is_cross": False, "shell_src": 1, "shell_dst": 1},
        {"time_slot": 1, "src": 3, "dst": 4, "delay_ms": 12.0, "available": True, "capacity_mbps": 100.0, "reliability": 0.94, "t_rem": 4.0, "is_cross": True, "shell_src": 1, "shell_dst": 1},
        {"time_slot": 1, "src": 4, "dst": 3, "delay_ms": 12.0, "available": True, "capacity_mbps": 100.0, "reliability": 0.94, "t_rem": 4.0, "is_cross": True, "shell_src": 1, "shell_dst": 1},
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(demo_rows)


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    demo_path = Path(__file__).resolve().parent / "outputs" / "hypatia_topology_demo.csv"
    write_demo_csv(demo_path)
    provider = HypatiaTopologyProvider.from_csv(demo_path)
    print("Wrote demo CSV:", demo_path)
    print("Available time slots:", sorted(provider.snapshot_index.keys()))
    print("Rows at t=1:", len(provider.snapshot_index.get(1, [])))
