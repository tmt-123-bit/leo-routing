"""Generalized P-plane x N-per-plane TLE snapshot builder.

Extends tle_topology_builder.py (whose selection is fixed at 4x6) to arbitrary
plane counts for the one-day 66-satellite feasibility study. Reuses the frozen
helpers (TLE parsing, SGP4 positions, line-of-sight, connectivity validation,
output format) so the exported CSV feeds the same HypatiaTopologyProvider.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sgp4.conveniences import sat_epoch_datetime

from tle_topology_builder import (
    LIGHT_SPEED_KM_S,
    group_raan_planes,
    has_line_of_sight,
    position_km,
    propagated_orbital_phase,
    read_three_line_tle,
    validate_snapshot_connectivity,
    write_outputs,
)


def select_planes_by_n(
    satellites,
    planes: int,
    per_plane: int,
    inclination_range: tuple[float, float],
    altitude_range: tuple[float, float],
    max_gap_deg: float,
    max_inplane_km: float = 3500.0,
):
    reference_time = max(sat_epoch_datetime(sat.satrec) for sat in satellites)
    usable = []
    for satellite in satellites:
        altitude = np.linalg.norm(position_km(satellite, reference_time)) - 6378.137
        if altitude_range[0] <= altitude <= altitude_range[1]:
            usable.append(satellite)
    groups = group_raan_planes(usable, inclination_range=inclination_range)
    scored = []
    for group in groups:
        if len(group) < per_plane:
            continue
        phased = sorted(
            group, key=lambda s: propagated_orbital_phase(s, reference_time)
        )
        positions = [position_km(s, reference_time) for s in phased]
        best_window, best_gap = None, None
        for start in range(len(phased) - per_plane + 1):
            window = phased[start : start + per_plane]
            gaps = [
                float(
                    np.linalg.norm(positions[start + k] - positions[start + k + 1])
                )
                for k in range(per_plane - 1)
            ]
            gap = max(gaps)
            if best_gap is None or gap < best_gap:
                best_window, best_gap = window, gap
        if best_window is None or best_gap > max_inplane_km:
            continue
        scored.append(
            (
                float(
                    np.mean([math.degrees(s.satrec.nodeo) % 360.0 for s in group])
                ),
                best_window,
            )
        )
    scored.sort(key=lambda entry: entry[0])
    if len(scored) < planes:
        raise ValueError(
            f"only {len(scored)} feasible planes (in-plane gap <= "
            f"{max_inplane_km} km); need {planes}"
        )
    best = None
    for start in range(len(scored) - planes + 1):
        window = scored[start : start + planes]
        gaps = [b[0] - a[0] for a, b in zip(window, window[1:])]
        if max(gaps) > max_gap_deg:
            continue
        spread = max(gaps) - min(gaps)
        if best is None or spread < best[0]:
            best = (spread, window)
    if best is None:
        raise ValueError(f"no {planes}-plane window within {max_gap_deg} deg spacing")
    selected = []
    for plane_id, (_, picks) in enumerate(best[1]):
        for local_index, sat in enumerate(picks):
            sat.plane = plane_id
            sat.local_index = local_index
            selected.append(sat)
    selected.sort(key=lambda s: (s.plane, s.local_index))
    if len({s.satrec.satnum for s in selected}) != len(selected):
        raise ValueError("duplicate satellites across selected planes")
    return selected


def slot_edges_pn(satellites, when: datetime, max_isl_km: float, cross_cap: int):
    n_planes = 1 + max(s.plane for s in satellites)
    positions = [position_km(satellite, when) for satellite in satellites]
    by_plane = {
        plane: [i for i, s in enumerate(satellites) if s.plane == plane]
        for plane in range(n_planes)
    }
    undirected = set()
    for members in by_plane.values():
        ordered = sorted(members, key=lambda i: satellites[i].local_index)
        for source, target in zip(ordered, ordered[1:]):
            undirected.add(tuple(sorted((source, target))))
    for source in range(len(satellites)):
        candidates = sorted(
            (
                float(np.linalg.norm(positions[source] - positions[target])),
                target,
            )
            for target in range(len(satellites))
            if satellites[target].plane != satellites[source].plane
        )
        added = 0
        for distance, target in candidates:
            if distance > max_isl_km or not has_line_of_sight(
                positions[source], positions[target]
            ):
                continue
            undirected.add(tuple(sorted((source, target))))
            added += 1
            if added >= cross_cap:
                break
    edges = {}
    for source, target in sorted(undirected):
        distance = float(np.linalg.norm(positions[source] - positions[target]))
        if distance > max_isl_km or not has_line_of_sight(
            positions[source], positions[target]
        ):
            continue
        is_cross = satellites[source].plane != satellites[target].plane
        edges[(source + 1, target + 1)] = (distance, is_cross)
        edges[(target + 1, source + 1)] = (distance, is_cross)
    return edges


def build_snapshots_pn(satellites, start, slots, slot_seconds, max_isl_km, cross_cap):
    lookahead = 12
    raw = [
        slot_edges_pn(
            satellites,
            start + timedelta(seconds=index * slot_seconds),
            max_isl_km,
            cross_cap,
        )
        for index in range(slots + lookahead)
    ]
    rows = []
    for slot in range(slots):
        for (source, target), (distance, is_cross) in raw[slot].items():
            remaining = 1
            for future in range(slot + 1, min(len(raw), slot + lookahead)):
                if (source, target) not in raw[future]:
                    break
                remaining += 1
            rows.append(
                {
                    "time_slot": slot + 1,
                    "src": source,
                    "dst": target,
                    "delay_ms": 1000.0 * distance / LIGHT_SPEED_KM_S,
                    "available": True,
                    "capacity_mbps": 100.0,
                    "reliability": 0.995,
                    "t_rem": remaining * slot_seconds,
                    "is_cross": is_cross,
                    "shell_src": 1,
                    "shell_dst": 1,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-tle", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--planes", type=int, required=True)
    parser.add_argument("--per-plane", type=int, required=True)
    parser.add_argument("--slots", type=int, default=30)
    parser.add_argument("--slot-seconds", type=int, default=10)
    parser.add_argument("--max-isl-km", type=float, default=5000.0)
    parser.add_argument("--cross-cap", type=int, default=3)
    parser.add_argument("--inclination-min", type=float, default=52.0)
    parser.add_argument("--inclination-max", type=float, default=54.0)
    parser.add_argument("--altitude-min", type=float, default=400.0)
    parser.add_argument("--altitude-max", type=float, default=650.0)
    parser.add_argument("--max-gap-deg", type=float, default=12.0)
    parser.add_argument(
        "--source-url",
        default="https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    )
    args = parser.parse_args()
    satellites = read_three_line_tle(args.tle)
    selected = select_planes_by_n(
        satellites,
        planes=args.planes,
        per_plane=args.per_plane,
        inclination_range=(args.inclination_min, args.inclination_max),
        altitude_range=(args.altitude_min, args.altitude_max),
        max_gap_deg=args.max_gap_deg,
    )
    n_nodes = args.planes * args.per_plane
    if len(selected) != n_nodes:
        raise ValueError(f"selected {len(selected)} sats, expected {n_nodes}")
    start = max(sat_epoch_datetime(sat.satrec) for sat in selected)
    rows = build_snapshots_pn(
        selected, start, slots=args.slots, slot_seconds=args.slot_seconds,
        max_isl_km=args.max_isl_km, cross_cap=args.cross_cap,
    )
    validate_snapshot_connectivity(rows, n_nodes=n_nodes)
    write_outputs(
        rows, selected, args.output, args.selected_tle, args.metadata,
        args.tle, start, args.slot_seconds, args.source_url,
    )
    print(
        f"exported {n_nodes} sats ({args.planes}x{args.per_plane}), "
        f"{args.slots} slots, {len(rows)} directed link rows"
    )


if __name__ == "__main__":
    main()
