"""Build the trace viewer.

    python -m viewer --all
    python -m viewer vendor/tau2-bench/data/simulations/pinned_telecom
    python -m viewer results/baseline_test -o /tmp/site

Writes one self-contained HTML page per run plus an index. Nothing is fetched
at view time, so the output directory can be opened straight from disk.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from viewer import adapters, payload, render

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "viewer" / "out"
SEARCH_DIRS = [
    REPO_ROOT / "vendor" / "tau2-bench" / "data" / "simulations",
    REPO_ROOT / "results",
]


def discover() -> list[Path]:
    found: list[Path] = []
    for root in SEARCH_DIRS:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and adapters.detect(child):
                found.append(child)
    return found


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "run"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, help="run directories or results.json files")
    ap.add_argument("--all", action="store_true", help="build every run found on disk")
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--name", help="override the run name (single path only)")
    ap.add_argument("--json", action="store_true", help="also write the payload as JSON")
    args = ap.parse_args()

    paths = list(args.paths)
    if args.all:
        paths += [p for p in discover() if p not in paths]
    if not paths:
        raise SystemExit("nothing to build: pass a run path or --all")
    if args.name and len(paths) > 1:
        raise SystemExit("--name only makes sense with a single run")

    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in paths:
        run = adapters.load(path, name=args.name)
        blob = payload.build(run)
        href = f"{slug(run.name)}.html"
        (args.out / href).write_text(render.render_run(blob))
        if args.json:
            (args.out / f"{slug(run.name)}.json").write_text(json.dumps(blob))
        entries.append(payload.index_entry(blob, href))
        summary = blob["summary"]
        print(
            f"{run.name:24s} {run.family}/{run.domain:9s} "
            f"tasks={summary['num_tasks']:4d} scored={summary['scored']:4d} "
            f"err={summary['errors']:3d} missing={summary['missing']:3d} "
            f"pass@1={_pct(summary['pass1_macro'])} -> {href}"
        )

    entries.sort(key=lambda e: (e["domain"], e["name"]))
    (args.out / "index.html").write_text(render.render_index(entries))
    print(f"\nwrote {len(entries)} run page(s) -> {args.out / 'index.html'}")


def _pct(value) -> str:
    return "  n/a" if value is None else f"{value:5.1%}"


if __name__ == "__main__":
    main()
