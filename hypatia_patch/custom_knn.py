# Small-shell generator for the official Hypatia framework.
#
# The official +Grid ISL rule cannot exist below ~156 satellites at 550 km
# (the official generator rejects it: ISL lengths exceed the physical limit).
# For the 24- and 66-satellite scale points this script finds the LOWEST
# altitude at which a K-nearest-neighbor ISL graph is fully connected with
# every edge inside the official max-ISL length for that altitude, and then
# runs the FULL official pipeline: official TLE generator, official
# top-100 ground stations, official description + GSL interfaces + dynamic
# state (Floyd-Warshall forwarding) -- only the isls.txt edge list is the
# k-nearest-neighbor graph instead of +Grid.
#
# Usage: python3 custom_knn.py <n_orbs> <sats_per_orb> <duration_s> <timestep_ms> <threads>
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "satgenpy"))
import satgen

EARTH_RADIUS_KM = 6378.135   # official constant (m: 6378135)
ISL_FLOOR_KM = 80.0          # official weather floor for laser ISLs
MU = 398600.4418             # km^3/s^2
KNN = 4                      # links per satellite (matches +Grid degree)
ALT_CANDIDATES = [550, 600, 650, 700, 750, 800, 900, 1000, 1100, 1200,
                  1400, 1600, 2000, 2400]


def mean_motion_rev_per_day(alt_km):
    period_s = 2 * math.pi * math.sqrt((EARTH_RADIUS_KM + alt_km) ** 3 / MU)
    return 86400.0 / period_s


def max_isl_km(alt_km):
    r = EARTH_RADIUS_KM + alt_km
    return 2.0 * math.sqrt(r * r - (EARTH_RADIUS_KM + ISL_FLOOR_KM) ** 2)


def max_gsl_km(alt_km):
    cone_km = 940.7 * alt_km / 550.0   # 25-deg-elevation equivalent, scaled
    return math.sqrt(cone_km * cone_km + alt_km * alt_km)


def read_positions(path, n_sats, offsets_s=(0.0,)):
    """ECI positions (km) of all satellites, propagated to epoch+each offset."""
    from sgp4.api import Satrec
    lines = [l.strip() for l in open(path) if l.strip()]
    if lines and not lines[0].startswith("1 "):
        lines = lines[1:]          # drop "P N" header if present
    sats = []
    for i in range(0, len(lines), 3):
        sats.append(Satrec.twoline2rv(lines[i + 1], lines[i + 2]))
    assert len(sats) == n_sats, f"expected {n_sats} sats, parsed {len(sats)}"
    all_pos = []
    for off in offsets_s:
        pos = []
        for s in sats:
            jd = s.jdsatepoch + off / 86400.0
            e, r, _v = s.sgp4(jd, 0.0)
            assert e == 0
            pos.append(r)
        all_pos.append(pos)
    return all_pos


def edge_lengths_ok(positions_sets, edges, limit_km):
    """True if every edge stays within the limit at every sampled time."""
    for pos in positions_sets:
        for a, b in edges:
            dx = pos[a][0] - pos[b][0]
            dy = pos[a][1] - pos[b][1]
            dz = pos[a][2] - pos[b][2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) > limit_km:
                return False
    return True


def knn_edges(positions, limit_km):
    """K-nearest-neighbor edges (deduped), clipped to the ISL length limit."""
    n = len(positions)
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dz = positions[i][2] - positions[j][2]
            dist[(i, j)] = math.sqrt(dx * dx + dy * dy + dz * dz)
    edges = set()
    for i in range(n):
        nearest = sorted(
            ((d, (b if a == i else a)) for (a, b), d in dist.items()
             if a == i or b == i),
            key=lambda x: x[0],
        )[:KNN]
        for d, other in nearest:
            if other != i and d <= limit_km:
                edges.add((min(i, other), max(i, other)))
    return edges


def connected(n, edges):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    return len({find(i) for i in range(n)}) == 1


def main():
    args = sys.argv[1:]
    if len(args) != 5:
        print("Usage: python3 custom_knn.py <n_orbs> <sats> <dur_s> <step_ms> <threads>")
        sys.exit(1)
    n_orbs, spp = int(args[0]), int(args[1])
    dur_s, step_ms, threads = int(args[2]), int(args[3]), int(args[4])
    n_sats = n_orbs * spp

    chosen = None
    for alt in ALT_CANDIDATES:
        tmp = f"gen_data/_knn_probe"
        os.makedirs(tmp, exist_ok=True)
        satgen.generate_tles_from_scratch_manual(
            tmp + "/tles.txt", f"Probe-{n_orbs}x{spp}", n_orbs, spp, True,
            53.0, 0.0000001, 0.0, mean_motion_rev_per_day(alt),
        )
        pos_sets = read_positions(tmp + "/tles.txt", n_sats,
                                  offsets_s=(0.0, dur_s / 2.0, float(dur_s)))
        edges = knn_edges(pos_sets[0], max_isl_km(alt))
        ok = connected(n_sats, edges) and edge_lengths_ok(pos_sets, edges,
                                                          max_isl_km(alt))
        print(f"alt={alt} km: max_isl={max_isl_km(alt):.0f} km, "
              f"edges={len(edges)}, connected+stable={ok}")
        if ok:
            chosen = (alt, edges)
            break
    if chosen is None:
        print("NO_VALID_ALTITUDE")
        sys.exit(1)
    alt, edges = chosen

    name = f"knn_{n_orbs}x{spp}_isls_knn_ground_stations_top_100_algorithm_free_one_only_over_isls"
    out = f"gen_data/{name}"
    os.makedirs(out, exist_ok=True)

    print(f"==> chosen altitude {alt} km with {len(edges)} kNN ISLs")
    satgen.generate_tles_from_scratch_manual(
        out + "/tles.txt", f"CustomKNN-{n_orbs}x{spp}", n_orbs, spp, True,
        53.0, 0.0000001, 0.0, mean_motion_rev_per_day(alt),
    )
    with open(out + "/isls.txt", "w") as f:
        for a, b in sorted(edges):
            f.write(f"{a} {b}\n")
    satgen.extend_ground_stations(
        "input_data/ground_stations_cities_sorted_by_estimated_2025_pop_top_100.basic.txt",
        out + "/ground_stations.txt",
    )
    satgen.generate_description(out + "/description.txt",
                                max_gsl_km(alt) * 1000, max_isl_km(alt) * 1000)
    satgen.generate_simple_gsl_interfaces_info(
        out + "/gsl_interfaces_info.txt", n_sats, 100, 1, 1, 1, 1,
    )
    satgen.help_dynamic_state(
        "gen_data", threads, name, step_ms, dur_s,
        max_gsl_km(alt) * 1000, max_isl_km(alt) * 1000,
        "algorithm_free_one_only_over_isls", True,
    )
    print(f"KNN_SHELL_DONE {name} alt={alt}")


if __name__ == "__main__":
    main()
