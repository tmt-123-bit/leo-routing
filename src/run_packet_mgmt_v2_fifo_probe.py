"""Complete the packet-management 2x2: purge criterion {static, time-expanded}
x queue order {FIFO, SRPF}, on the same fresh panel as the v2 probe
(workloads 880651-880690). The SRPF cells come from
packet-mgmt-v2-probe-20260926-v1; this runner fills the two FIFO cells.

Hypothesis under test: the purge's delivery gain without reordering comes
from removing certain-loss packets that FIFO would otherwise serve; once
SRPF ordering shields the head of line, the criterion's tightness is
delivery-irrelevant (the v2 probe saw exactly that: the exact
time-expanded criterion purges 27x fewer packets yet delivers the same).
"""

import argparse
from collections import deque
from pathlib import Path
import time

import numpy as np
import torch

from hypatia_topology_provider_stub import HypatiaTopologyProvider
from mappo_evaluation import GlobalDijkstraPolicy
from run_development_load_diagnostics import canonical, read_json, sha, write_json
from run_infeasible_drop_probe import PurgeInfeasibleEnv
from run_packet_mgmt_family_probe import (
    TOPOLOGY,
    FamilyWrapper,
    run_row,
)
from run_packet_mgmt_v2_probe import TimeExpandedSRPFEnv, WORKLOADS, LOADS


class TimeExpandedFIFOEnv(TimeExpandedSRPFEnv):
    """Time-expanded purge without reordering (native FIFO service order)."""

    reorder_queues = False


ARMS = {
    "purge_static_fifo": {"cls": PurgeInfeasibleEnv, "kwargs": {}, "family": "2x2"},
    "purge_texp_fifo": {"cls": TimeExpandedFIFOEnv, "kwargs": {}, "family": "2x2"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    FamilyWrapper._provider = HypatiaTopologyProvider.from_csv(TOPOLOGY)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    rows = []
    pairing = {}
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for load in LOADS:
            for w in WORKLOADS:
                for arm_name, spec in ARMS.items():
                    row = run_row(GlobalDijkstraPolicy(), f"dijkstra::{arm_name}", w, load, spec)
                    events = (row["arrival_sha256"], row["physical_sha256"])
                    key = (load, w)
                    if key in pairing:
                        if pairing[key] != events:
                            raise AssertionError(f"pairing changed at {key}")
                    else:
                        pairing[key] = events
                    stream.write(canonical(row) + "\n")
                    rows.append(row)
                stream.flush()
            print(f"{len(rows)} load={load} done elapsed={time.monotonic()-started:.0f}s", flush=True)
    write_json(output / "manifest.json", {
        "role": "packet_mgmt_v2_fifo_cells_dev",
        "runner_sha256": sha(Path(__file__)),
        "workloads": WORKLOADS, "loads": list(LOADS),
        "arms": list(ARMS),
        "training": False, "sealed_test_access": False,
    })
    means = {}
    for load in LOADS:
        for arm_name in ARMS:
            vals = [r["delivery_ratio"] for r in rows
                    if r["load"] == load and r["arm"] == arm_name]
            means[f"{load}:{arm_name}"] = float(np.mean(vals))
    write_json(output / "fifo_cells.json", {
        "mean_delivery": means,
        "texp_fifo_mean_purged": float(np.mean([r["purged_infeasible"] for r in rows
                                                if r["arm"] == "purge_texp_fifo"])),
        "static_fifo_mean_purged": float(np.mean([r["purged_infeasible"] for r in rows
                                                  if r["arm"] == "purge_static_fifo"])),
        "episodes": len(rows), "pairings_verified": len(pairing),
        "elapsed_seconds": time.monotonic() - started,
    })
    print(canonical(read_json(output / "fifo_cells.json")), flush=True)


if __name__ == "__main__":
    main()
