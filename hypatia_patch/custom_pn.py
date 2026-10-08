# Generic P x N shell generator on the official Hypatia MainHelper
# (Starlink-550 FCC parameters, top-100 ground stations, +Grid ISLs,
#  free_one_only_over_isls routing).
# Usage: python3 custom_pn.py <n_orbs> <sats_per_orb> <duration_s> <timestep_ms> <threads>
import sys
import math
from main_helper import MainHelper

EARTH_RADIUS = 6378135.0
ECCENTRICITY = 0.0000001
ARG_OF_PERIGEE_DEGREE = 0.0
PHASE_DIFF = True
MEAN_MOTION_REV_PER_DAY = 15.19
ALTITUDE_M = 550000
SATELLITE_CONE_RADIUS_M = 940700
MAX_GSL_LENGTH_M = math.sqrt(math.pow(SATELLITE_CONE_RADIUS_M, 2) + math.pow(ALTITUDE_M, 2))
MAX_ISL_LENGTH_M = 2 * math.sqrt(math.pow(EARTH_RADIUS + ALTITUDE_M, 2) - math.pow(EARTH_RADIUS + 80000, 2))
INCLINATION_DEGREE = 53

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 5:
        print("Usage: python3 custom_pn.py <n_orbs> <sats_per_orb> <duration_s> <timestep_ms> <threads>")
        sys.exit(1)
    n_orbs, spp = int(args[0]), int(args[1])
    helper = MainHelper(
        f"custom_{n_orbs}x{spp}",
        f"Custom-{n_orbs}x{spp}",
        ECCENTRICITY, ARG_OF_PERIGEE_DEGREE, PHASE_DIFF,
        MEAN_MOTION_REV_PER_DAY, ALTITUDE_M,
        MAX_GSL_LENGTH_M, MAX_ISL_LENGTH_M,
        n_orbs, spp, INCLINATION_DEGREE,
    )
    helper.calculate(
        "gen_data",
        int(args[2]), int(args[3]),
        "isls_plus_grid",
        "ground_stations_top_100",
        "algorithm_free_one_only_over_isls",
        int(args[4]),
    )
