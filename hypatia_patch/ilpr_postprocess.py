# ILPR-style persistent routing for official Hypatia (fstate post-processor).
# v3: delta-aware. Hypatia fstate files are cumulative only in the FIRST
# interval; every later file lists just the CHANGED (node, target) entries.
# Persistence: for each changed entry, keep the previously-installed next
# hop whenever its ISL still exists (static +Grid links exist throughout,
# mirroring the slot-env route-cache persistence where links could break);
# emit the kept entry as the delta so the arbiter leaves the route alone.
import os
import shutil
import sys


def load_isls(path):
    adj = {}
    for line in open(path):
        a, b = (int(x) for x in line.split())
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def parse_line(line):
    parts = line.strip().split(",")
    if len(parts) != 5:
        return None
    a, b, c, d, e = (int(x) for x in parts)
    return a, b, c, d, e


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for f in ("tles.txt", "isls.txt", "ground_stations.txt",
              "gsl_interfaces_info.txt", "description.txt"):
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))

    adj = load_isls(os.path.join(src, "isls.txt"))
    state_dir = os.path.join(src, "dynamic_state_100ms_for_10s")
    out_dir = os.path.join(dst, "dynamic_state_100ms_for_10s")
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(state_dir):
        if not fname.startswith("fstate_"):
            shutil.copyfile(os.path.join(state_dir, fname),
                            os.path.join(out_dir, fname))

    accumulated = {}   # node -> {target: (nh, my_if, next_if)} as installed
    total_kept = total_adopted = 0
    for fname in sorted(os.listdir(state_dir)):
        if not fname.startswith("fstate_"):
            continue
        out_lines = []
        for line in open(os.path.join(state_dir, fname)):
            parsed = parse_line(line)
            if parsed is None:
                continue
            node, tgt, nh, mi, ni = parsed
            old = accumulated.get(node, {}).get(tgt)
            if old is not None and old[0] in adj.get(node, set()) \
                    and old[0] != nh:
                out_lines.append(f"{node},{tgt},{old[0]},{old[1]},{old[2]}")
                accumulated.setdefault(node, {})[tgt] = old
                total_kept += 1
            else:
                out_lines.append(f"{node},{tgt},{nh},{mi},{ni}")
                accumulated.setdefault(node, {})[tgt] = (nh, mi, ni)
                total_adopted += 1
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write("\n".join(out_lines) + "\n")
    print(f"ILPR_POST_DONE kept={total_kept} adopted={total_adopted}")


if __name__ == "__main__":
    main()
