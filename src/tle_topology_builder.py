"""Build reproducible LEO link snapshots from a frozen three-line TLE file."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from sgp4.api import Satrec, jday
from sgp4.conveniences import sat_epoch_datetime


EARTH_RADIUS_KM = 6378.137
LIGHT_SPEED_KM_S = 299792.458


@dataclass
class TleSatellite:
    name: str
    line1: str
    line2: str
    satrec: Satrec
    plane: int = -1
    local_index: int = -1


def orbital_phase(satellite: TleSatellite) -> float:
    return (satellite.satrec.mo + satellite.satrec.argpo) % (2 * math.pi)


def propagated_orbital_phase(satellite: TleSatellite, when: datetime) -> float:
    position = position_km(satellite, when)
    raan = satellite.satrec.nodeo
    inclination = satellite.satrec.inclo
    in_plane_x = position[0] * math.cos(raan) + position[1] * math.sin(raan)
    in_plane_y = position[2] / max(1e-6, math.sin(inclination))
    return math.atan2(in_plane_y, in_plane_x) % (2 * math.pi)


def read_three_line_tle(path: str | Path) -> list[TleSatellite]:
    lines = [line.rstrip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 3:
        raise ValueError("TLE file must contain name/line1/line2 triples")
    satellites = []
    for index in range(0, len(lines), 3):
        name, line1, line2 = lines[index : index + 3]
        satellites.append(
            TleSatellite(name, line1, line2, Satrec.twoline2rv(line1, line2))
        )
    return satellites


def group_raan_planes(
    satellites: Iterable[TleSatellite],
    bin_width_deg: float = 1.0,
    inclination_range: tuple[float, float] = (52.0, 54.0),
) -> list[list[TleSatellite]]:
    bins: dict[int, list[TleSatellite]] = {}
    for satellite in satellites:
        inclination = math.degrees(satellite.satrec.inclo)
        if not inclination_range[0] <= inclination <= inclination_range[1]:
            continue
        raan = math.degrees(satellite.satrec.nodeo) % 360.0
        bin_index = int(round(raan / bin_width_deg)) % int(
            round(360.0 / bin_width_deg)
        )
        bins.setdefault(bin_index, []).append(satellite)
    return [bins[index] for index in sorted(bins) if len(bins[index]) >= 6]


def select_four_by_six(
    satellites: list[TleSatellite],
    inclination_range: tuple[float, float] = (52.0, 54.0),
    plane_spacing_deg: float = 5.0,
    altitude_range: tuple[float, float] = (400.0, 650.0),
) -> list[TleSatellite]:
    reference_time = max(sat_epoch_datetime(sat.satrec) for sat in satellites)
    usable = []
    phase_by_satnum = {}
    for satellite in satellites:
        position = position_km(satellite, reference_time)
        altitude = np.linalg.norm(position) - EARTH_RADIUS_KM
        if altitude_range[0] <= altitude <= altitude_range[1]:
            usable.append(satellite)
            phase_by_satnum[satellite.satrec.satnum] = propagated_orbital_phase(
                satellite, reference_time
            )
    groups = group_raan_planes(
        usable, inclination_range=inclination_range
    )
    if len(groups) < 4:
        raise ValueError(
            "fewer than four usable RAAN planes in the requested inclination "
            "and altitude ranges"
        )
    entries = sorted(
        (
            float(
                np.mean(
                    [math.degrees(sat.satrec.nodeo) % 360.0 for sat in group]
                )
            ),
            group,
        )
        for group in groups
        if len(group) >= 12
    )
    chains = []
    for center, group in entries:
        chain = [(center, group)]
        for offset in range(1, 4):
            target = center + plane_spacing_deg * offset
            matches = [
                entry
                for entry in entries
                if abs(entry[0] - target)
                <= max(1.25, 0.2 * plane_spacing_deg)
            ]
            if not matches:
                break
            chain.append(min(matches, key=lambda entry: abs(entry[0] - target)))
        if len(chain) == 4 and len({round(item[0], 3) for item in chain}) == 4:
            chains.append(chain)
    if not chains:
        raise ValueError(
            "no four approximately "
            f"{plane_spacing_deg:g}-degree-spaced RAAN planes were found"
        )

    best = None
    for chain in chains:
        for target_phase in np.linspace(0.0, 2 * math.pi, 72, endpoint=False):
            chosen_groups = []
            max_phase_distance = 0.0
            for _, group in chain:
                ranked = sorted(
                    group,
                    key=lambda sat: min(
                        abs(phase_by_satnum[sat.satrec.satnum] - target_phase),
                        2 * math.pi
                        - abs(phase_by_satnum[sat.satrec.satnum] - target_phase),
                    ),
                )[:6]
                chosen_groups.append(ranked)
                max_phase_distance = max(
                    max_phase_distance,
                    max(
                        min(
                            abs(phase_by_satnum[sat.satrec.satnum] - target_phase),
                            2 * math.pi
                            - abs(phase_by_satnum[sat.satrec.satnum] - target_phase),
                        )
                        for sat in ranked
                    ),
                )
            candidate = (max_phase_distance, -min(len(item[1]) for item in chain))
            if best is None or candidate < best[0]:
                best = (candidate, target_phase, chosen_groups)

    assert best is not None
    _, target_phase, groups_selected = best
    selected = []
    for plane, group in enumerate(groups_selected):
        group.sort(
            key=lambda sat: (
                (
                    phase_by_satnum[sat.satrec.satnum]
                    - target_phase
                    + math.pi
                )
                % (2 * math.pi)
            )
            - math.pi
        )
        for local_index, satellite in enumerate(group):
            satellite.plane = plane
            satellite.local_index = local_index
            selected.append(satellite)
    return selected


def position_km(satellite: TleSatellite, when: datetime) -> np.ndarray:
    utc = when.astimezone(timezone.utc)
    jd, fraction = jday(
        utc.year,
        utc.month,
        utc.day,
        utc.hour,
        utc.minute,
        utc.second + utc.microsecond / 1e6,
    )
    error, position, _ = satellite.satrec.sgp4(jd, fraction)
    if error:
        raise RuntimeError(f"SGP4 error {error} for {satellite.name}")
    return np.asarray(position, dtype=float)


def has_line_of_sight(a: np.ndarray, b: np.ndarray) -> bool:
    radius_a = np.linalg.norm(a)
    radius_b = np.linalg.norm(b)
    cosine = np.dot(a, b) / (radius_a * radius_b)
    central_angle = math.acos(float(np.clip(cosine, -1.0, 1.0)))
    horizon = math.acos(EARTH_RADIUS_KM / radius_a) + math.acos(
        EARTH_RADIUS_KM / radius_b
    )
    return central_angle <= horizon


def slot_edges(
    satellites: list[TleSatellite],
    when: datetime,
    max_isl_km: float,
) -> dict[tuple[int, int], tuple[float, bool]]:
    positions = [position_km(satellite, when) for satellite in satellites]
    by_plane = {
        plane: [index for index, sat in enumerate(satellites) if sat.plane == plane]
        for plane in range(4)
    }
    undirected = set()
    for members in by_plane.values():
        ordered = sorted(members, key=lambda index: satellites[index].local_index)
        for source, target in zip(ordered, ordered[1:]):
            undirected.add(tuple(sorted((source, target))))
    for plane in range(4):
        next_plane = (plane + 1) % 4
        candidates = sorted(
            (
                float(np.linalg.norm(positions[source] - positions[target])),
                source,
                target,
            )
            for source in by_plane[plane]
            for target in by_plane[next_plane]
        )
        added = 0
        for distance, source, target in candidates:
            if distance > max_isl_km or not has_line_of_sight(
                positions[source], positions[target]
            ):
                continue
            undirected.add(tuple(sorted((source, target))))
            added += 1
            if added >= 3:
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


def build_snapshots(
    satellites: list[TleSatellite],
    start: datetime,
    slots: int = 30,
    slot_seconds: int = 10,
    max_isl_km: float = 5000.0,
) -> list[dict]:
    lookahead = 12
    raw = [
        slot_edges(
            satellites,
            start + timedelta(seconds=index * slot_seconds),
            max_isl_km,
        )
        for index in range(slots + lookahead)
    ]
    rows = []
    for slot in range(slots):
        for (source, target), (distance, is_cross) in raw[slot].items():
            remaining_slots = 1
            for future in range(slot + 1, min(len(raw), slot + lookahead)):
                if (source, target) not in raw[future]:
                    break
                remaining_slots += 1
            rows.append(
                {
                    "time_slot": slot + 1,
                    "src": source,
                    "dst": target,
                    "delay_ms": 1000.0 * distance / LIGHT_SPEED_KM_S,
                    "available": True,
                    "capacity_mbps": 100.0,
                    "reliability": 0.995,
                    "t_rem": remaining_slots * slot_seconds,
                    "is_cross": is_cross,
                    "shell_src": 1,
                    "shell_dst": 1,
                }
            )
    return rows


def validate_snapshot_connectivity(rows: list[dict], n_nodes: int = 24) -> None:
    slots = sorted({int(row["time_slot"]) for row in rows})
    for slot in slots:
        adjacency = {node: set() for node in range(1, n_nodes + 1)}
        for row in rows:
            if int(row["time_slot"]) != slot:
                continue
            source, target = int(row["src"]), int(row["dst"])
            adjacency[source].add(target)
            adjacency[target].add(source)
        reached = {1}
        frontier = [1]
        while frontier:
            node = frontier.pop()
            for neighbor in adjacency[node] - reached:
                reached.add(neighbor)
                frontier.append(neighbor)
        if len(reached) != n_nodes:
            raise ValueError(
                f"TLE topology is disconnected at slot {slot}: "
                f"reached {len(reached)}/{n_nodes} nodes"
            )


def write_outputs(
    rows: list[dict],
    satellites: list[TleSatellite],
    output_csv: Path,
    selected_tle: Path,
    metadata_path: Path,
    source_tle: Path,
    start: datetime,
    slot_seconds: int,
    source_url: str,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected_tle.write_text(
        "\n".join(
            line for sat in satellites for line in (sat.name, sat.line1, sat.line2)
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "source_tle": str(source_tle.resolve()),
        "source_tle_sha256": hashlib.sha256(source_tle.read_bytes()).hexdigest(),
        "source_url": source_url,
        "selected_satellites": [
            {
                "environment_id": index + 1,
                "name": sat.name,
                "norad_id": sat.satrec.satnum,
                "plane": sat.plane,
                "local_index": sat.local_index,
            }
            for index, sat in enumerate(satellites)
        ],
        "start_utc": start.isoformat(),
        "slot_seconds": slot_seconds,
        "slots": max(row["time_slot"] for row in rows),
        "model_limits": [
            "SGP4 propagation uses the frozen TLE epoch.",
            "ISLs are constructed as same-plane neighbor links plus nearest adjacent-plane links.",
            "Capacity and reliability are experiment assumptions, not TLE measurements.",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-tle", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=30)
    parser.add_argument("--slot-seconds", type=int, default=10)
    parser.add_argument("--already-selected", action="store_true")
    parser.add_argument("--inclination-min", type=float, default=52.0)
    parser.add_argument("--inclination-max", type=float, default=54.0)
    parser.add_argument("--plane-spacing-deg", type=float, default=5.0)
    parser.add_argument("--altitude-min", type=float, default=400.0)
    parser.add_argument("--altitude-max", type=float, default=650.0)
    parser.add_argument(
        "--source-url",
        default="https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    )
    args = parser.parse_args()
    if args.altitude_min >= args.altitude_max:
        parser.error("--altitude-min must be less than --altitude-max")
    satellites = read_three_line_tle(args.tle)
    if args.already_selected:
        if len(satellites) != 24:
            raise ValueError("--already-selected requires exactly 24 TLE records")
        selected = satellites
        for index, satellite in enumerate(selected):
            satellite.plane = index // 6
            satellite.local_index = index % 6
    else:
        selected = select_four_by_six(
            satellites,
            inclination_range=(args.inclination_min, args.inclination_max),
            plane_spacing_deg=args.plane_spacing_deg,
            altitude_range=(args.altitude_min, args.altitude_max),
        )
    start = max(sat_epoch_datetime(sat.satrec) for sat in selected)
    rows = build_snapshots(
        selected,
        start,
        slots=args.slots,
        slot_seconds=args.slot_seconds,
    )
    validate_snapshot_connectivity(rows, n_nodes=len(selected))
    write_outputs(
        rows,
        selected,
        args.output,
        args.selected_tle,
        args.metadata,
        args.tle,
        start,
        args.slot_seconds,
        args.source_url,
    )
    print(f"selected satellites: {len(selected)}")
    print(f"snapshot rows: {len(rows)}")
    print(f"output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
