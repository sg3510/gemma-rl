"""pass@k and pass^k from tau-bench result files.

THE METRIC TRAP: tau-bench's own `display_metrics()` reports **pass^k**, the
probability that *all* k sampled rollouts succeed:

    pass^k = mean_over_tasks[ C(c, k) / C(n, k) ]

That is a reliability metric, and it is what the published leaderboard shows.
It is NOT what we want for RL curriculum work. We want **pass@k**, the
probability that *at least one* of k rollouts succeeds:

    pass@k = mean_over_tasks[ 1 - C(n-c, k) / C(n, k) ]

pass@k is the standard unbiased estimator (Chen et al., 2021). It answers "can
the model do this task at all", which is exactly the question that decides
whether a task belongs in the training set. Confusing the two makes the model
look catastrophically worse than it is: for a task solved 3 times out of 8,
pass@8 = 1.00 while pass^8 = 0.00.

Both are computed here from the same raw per-trial rewards, and every report
labels which is which. tau-bench itself is never modified.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

SUCCESS_EPS = 1e-6


def is_success(reward: float) -> bool:
    """tau-bench's own success test: reward within epsilon of 1.0."""
    return reward > 1 - SUCCESS_EPS


# Exceptions that surface inside `agent.solve` but are the MODEL's fault, not
# the provider's. The big one: tau-bench does
# `json.loads(tool_call["function"]["arguments"])` in `message_to_action` with no
# error handling, so a model that emits malformed JSON in a tool call blows up
# the whole rollout. That is a genuine agent failure and must be scored 0, not
# excluded. tau-bench catches every exception identically, so the traceback is
# the only thing that distinguishes them.
MODEL_FAULT_MARKERS = (
    "json.decoder.JSONDecodeError",
    "message_to_action",
    "MalformedToolArguments",
)


def is_model_fault(entry: dict) -> bool:
    """True if the rollout raised because of the model's own output."""
    info = entry.get("info")
    if not isinstance(info, dict) or "error" not in info:
        return False
    blob = f"{info.get('error', '')}\n{info.get('traceback', '')}"
    return any(m in blob for m in MODEL_FAULT_MARKERS)


def is_errored(entry: dict) -> bool:
    """True if this rollout crashed on INFRASTRUCTURE rather than failing the task.

    tau-bench's `run._run` wraps `agent.solve` in a bare `except Exception` and
    records the crash as `reward=0.0` with an `error` key in `info`. That makes
    an exhausted retry budget or a 500 from the provider look exactly like the
    agent failing the task, which would silently deflate pass@1. We exclude
    those from the denominator and re-run them.

    But we must NOT do that for failures the model itself caused -- excluding
    those would inflate pass@1 and make the runner retry them forever.
    """
    info = entry.get("info")
    if not isinstance(info, dict) or "error" not in info:
        return False
    return not is_model_fault(entry)


def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one success in k draws) -- unbiased estimator over n trials."""
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} trials, have {n}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """P(all k draws succeed) -- tau-bench's leaderboard metric."""
    if k > n:
        raise ValueError(f"pass^{k} needs at least {k} trials, have {n}")
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


@dataclass
class TaskStats:
    task_id: int | str
    rewards: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rewards)

    @property
    def c(self) -> int:
        return sum(1 for r in self.rewards if is_success(r))

    @property
    def mean_reward(self) -> float:
        return sum(self.rewards) / self.n if self.n else 0.0


def scan_cells(path: Path) -> tuple[dict[tuple[int, str], float], set[tuple[int, str]]]:
    """Scan a sweep dir into (completed cells, errored cells).

    A "cell" is one (task_id, trial) slot. Returns the reward for every cell
    that completed cleanly, plus the set of cells that crashed and therefore
    need re-running. Both `load_results` and the resumable runner build on this.

    Deduplication: files are read oldest-first so the MOST RECENT write wins.
    tau-bench appends to a checkpoint as it goes and a resumed sweep can
    legitimately rewrite a slot; last-write-wins is the only ordering that is
    deterministic and not a function of the filename.

    Trial identity comes from the `trial_<i>` directory when present, because
    our runner invokes tau-bench once per trial and upstream would otherwise
    stamp every result as trial 0.
    """
    if path.is_dir():
        files = sorted(path.rglob("*.json"), key=lambda f: (f.stat().st_mtime, str(f)))
    elif path.exists():
        files = [path]
    else:
        return {}, set()

    cells: dict[tuple[int, str], float] = {}
    errored: set[tuple[int, str]] = set()

    for f in files:
        try:
            entries = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(entries, list):
            continue
        trial_dir = next(
            (q.name for q in f.parents if q.name.startswith("trial_")), None
        )
        for e in entries:
            if not isinstance(e, dict) or "task_id" not in e or "reward" not in e:
                continue
            key = (int(e["task_id"]), trial_dir or f"stored_{e.get('trial', 0)}")
            if is_errored(e):
                errored.add(key)
                cells.pop(key, None)
            else:
                errored.discard(key)
                cells[key] = float(e["reward"])

    return cells, errored


def load_results(path: Path) -> dict[int, TaskStats]:
    """Load a sweep directory (or a single tau-bench results JSON) into per-task
    stats. Crashed rollouts are excluded from the denominator -- see `is_errored`.
    """
    cells, _errored = scan_cells(path)
    tasks: dict[int, TaskStats] = {}
    for (tid, _trial), reward in sorted(cells.items(), key=lambda kv: kv[0]):
        tasks.setdefault(tid, TaskStats(task_id=tid, rewards=[])).rewards.append(reward)
    return dict(sorted(tasks.items()))


def summarize(tasks: dict[int, TaskStats], ks: list[int] | None = None) -> dict:
    """Headline metrics over a set of tasks."""
    if not tasks:
        return {"num_tasks": 0}
    n_min = min(t.n for t in tasks.values())
    # Every k up to the smallest trial count -- skipping values (e.g. only
    # reporting 1,2,4,8) hides the shape of the curve when n is not a power of 2.
    ks = ks or list(range(1, n_min + 1))

    out = {
        "num_tasks": len(tasks),
        "trials_per_task_min": n_min,
        "trials_per_task_max": max(t.n for t in tasks.values()),
        "total_rollouts": sum(t.n for t in tasks.values()),
        "pass_at_k": {},
        "pass_hat_k": {},
    }
    for k in ks:
        eligible = [t for t in tasks.values() if t.n >= k]
        out["pass_at_k"][k] = sum(pass_at_k(t.n, t.c, k) for t in eligible) / len(
            eligible
        )
        out["pass_hat_k"][k] = sum(pass_hat_k(t.n, t.c, k) for t in eligible) / len(
            eligible
        )
    # pass@1 is identical to mean reward for a binary reward; assert it holds.
    out["mean_reward"] = sum(t.mean_reward for t in tasks.values()) / len(tasks)
    return out


def per_task_rows(tasks: dict[int, TaskStats]) -> list[dict]:
    rows = []
    for t in tasks.values():
        rows.append(
            {
                "task_id": t.task_id,
                "n": t.n,
                "c": t.c,
                "success_rate": round(t.c / t.n, 4) if t.n else 0.0,
                "pass@1": round(t.mean_reward, 4),
                "pass@n": round(pass_at_k(t.n, t.c, t.n), 4) if t.n else 0.0,
            }
        )
    return rows


def format_summary(s: dict, title: str) -> str:
    if not s.get("num_tasks"):
        return f"# {title}\n\nNo results found.\n"
    lines = [
        f"# {title}",
        "",
        f"- tasks: **{s['num_tasks']}**",
        f"- rollouts: **{s['total_rollouts']}** "
        f"({s['trials_per_task_min']}-{s['trials_per_task_max']} per task)",
        "",
        "| k | pass@k (>=1 of k succeeds) | pass^k (all k succeed) |",
        "|---|---|---|",
    ]
    for k in s["pass_at_k"]:
        lines.append(
            f"| {k} | **{s['pass_at_k'][k]:.1%}** | {s['pass_hat_k'][k]:.1%} |"
        )
    lines += [
        "",
        f"pass@1 = mean reward = **{s['mean_reward']:.1%}**",
        "",
        "> pass@k is our target metric. pass^k is what tau-bench's own",
        "> `display_metrics()` and the published leaderboard report.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute pass@k and pass^k.")
    ap.add_argument("results", type=Path, help="sweep dir or results JSON")
    ap.add_argument("--title", default="tau-retail results")
    ap.add_argument("--csv", type=Path, help="write per-task CSV here")
    args = ap.parse_args()

    tasks = load_results(args.results)
    print(format_summary(summarize(tasks), args.title))

    if args.csv:
        import csv

        rows = per_task_rows(tasks)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
