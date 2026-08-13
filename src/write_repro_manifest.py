"""Reproducibility manifest — pip freeze + git refs + source/fixture hashes.

Closes the last anti-rejection checklist item: a reviewer must be able to
reconstruct the exact code + environment that produced the numbers. run_exp004
already writes per-run `code_fingerprint.json` (source SHA256); this adds the
environment + VCS + fixture layer and emits one top-level manifest.

Run it from the project dir, with the SAME python that will run the experiments
(the F: venv after GPU setup):
    /f/leo-venv/Scripts/python.exe write_repro_manifest.py \
        --output experiments/IEEE-EXP-004-FULL/repro_manifest.json

Idempotent — re-run after any code/env change before submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(repo: Path) -> dict:
    if not (repo / ".git").exists():
        return {"present": False, "path": str(repo)}
    def run(args: list[str]) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=15
            ).stdout.strip()
        except Exception as exc:
            return f"<git error: {exc}>"
    return {
        "present": True,
        "path": str(repo),
        "head": run(["rev-parse", "HEAD"]),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run(["status", "--porcelain"])),
        "dirty_files": [
            line.strip()
            for line in run(["status", "--porcelain"]).splitlines()
            if line.strip()
        ][:200],
    }


def pip_freeze(python: str) -> list[str]:
    try:
        out = subprocess.run(
            [python, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120
        ).stdout
        return [line for line in out.splitlines() if line.strip()]
    except Exception as exc:
        return [f"<pip freeze error: {exc}>"]


def hash_tree(roots: list[Path], globs: list[str], limit: int = 5000) -> dict:
    out: dict[str, str] = {}
    count = 0
    for root in roots:
        if not root.exists():
            continue
        seen: set[Path] = set()
        for pattern in globs:
            for path in root.rglob(pattern):
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                if any(
                    part in {"__pycache__", ".git", "checkpoints"}
                    or part.endswith(".egg-info")
                    for part in path.parts
                ):
                    continue
                try:
                    out[str(path)] = sha256(path)
                    count += 1
                except OSError:
                    pass
                if count >= limit:
                    return out
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("repro_manifest.json"),
    )
    parser.add_argument("--cleanmarl", type=Path, default=Path("F:/cleanmarl"))
    parser.add_argument(
        "--fixtures",
        type=Path,
        nargs="*",
        default=None,
        help="fixture dirs/files (ns-3, TLE) to hash; omitted = none",
    )
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    python = sys.executable

    manifest = {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": python,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pip_freeze": pip_freeze(python),
        "git": {
            "project": git_info(project),
            "cleanmarl": git_info(args.cleanmarl),
        },
        "source_hashes": hash_tree(
            [project, args.cleanmarl / "cleanmarl"], ["*.py"]
        ),
    }
    if args.fixtures:
        manifest["fixture_hashes"] = {
            str(p): sha256(p) if p.is_file() else "<dir>"
            for p in args.fixtures
            if p.exists()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    n_src = len(manifest["source_hashes"])
    n_pkgs = len(manifest["pip_freeze"])
    print(f"wrote {args.output} ({n_src} source files, {n_pkgs} pip pkgs)")
    print(f"project HEAD: {manifest['git']['project'].get('head','?')[:12]}  dirty={manifest['git']['project'].get('dirty')}")
    print(f"cleanmarl HEAD: {manifest['git']['cleanmarl'].get('head','?')[:12]}  dirty={manifest['git']['cleanmarl'].get('dirty')}")
    if manifest["git"]["project"].get("dirty") or manifest["git"]["cleanmarl"].get("dirty"):
        print("WARNING: working tree has uncommitted changes — commit before submission for a clean manifest.")


if __name__ == "__main__":
    main()
