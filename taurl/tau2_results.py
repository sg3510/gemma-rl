"""Read tau2-bench results into the same shape as our tau1 metrics.

tau2 stores one `results.json` per run: a dict with a `simulations` list, each
entry carrying `task_id` (a string, not an int), `trial`, `reward_info.reward`
and `termination_reason`. Mapping it onto `TaskStats` lets pass@k / pass^k be
computed identically across tau1 retail, tau1 airline and tau2 telecom, so the
three numbers are actually comparable.

tau2 also records `termination_reason`, which is the built-in analogue of our
tau1 failure taxonomy -- `max_steps` in particular is a harness limit rather
than a capability limit, and tau2's default cap is 200 steps versus tau1's 30.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from taurl.metrics import TaskStats, format_summary, summarize


def load(path: Path) -> tuple[dict[str, TaskStats], Counter, dict]:
    """Return (per-task stats, termination reasons, run info)."""
    if path.is_dir():
        path = path / "results.json"
    blob = json.loads(path.read_text())
    sims = blob.get("simulations") or []

    tasks: dict[str, TaskStats] = {}
    reasons: Counter = Counter()
    seen: set[tuple[str, int]] = set()

    for sim in sims:
        tid = str(sim.get("task_id"))
        trial = int(sim.get("trial", 0))
        if (tid, trial) in seen:
            continue
        seen.add((tid, trial))
        reward = float((sim.get("reward_info") or {}).get("reward", 0.0))
        tasks.setdefault(tid, TaskStats(task_id=tid, rewards=[])).rewards.append(reward)
        reasons[str(sim.get("termination_reason"))] += 1

    return tasks, reasons, blob.get("info") or {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path, help="simulations dir or results.json")
    ap.add_argument("--title", default="tau2 results")
    args = ap.parse_args()

    tasks, reasons, info = load(args.results)
    if not tasks:
        raise SystemExit(f"no simulations found in {args.results}")

    print(format_summary(summarize(tasks), args.title))
    total = sum(reasons.values())
    print("## Termination reasons\n")
    print("| reason | count | % |")
    print("|---|---|---|")
    for reason, n in reasons.most_common():
        print(f"| `{reason}` | {n} | {n / total:.0%} |")
    print(
        "\n> `max_steps` is a harness limit, not a capability limit "
        "(tau2 default is 200 steps; tau1 uses 30)."
    )

if __name__ == "__main__":
    main()
