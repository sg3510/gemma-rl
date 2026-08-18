"""Turn measured per-task success rates into a curriculum.

The output is the training frontier: tasks the model can *sometimes* solve.

Selection criterion is the empirical success fraction over n sampled rollouts,
`c/n`, which is an estimate of that task's **pass@1**. It is NOT pass@8 --
pass@8 is 1.0 for any task with at least one success, so it cannot discriminate
between tasks at all.

Why the middle of the range: GRPO computes advantage relative to the group mean,
A_i = r_i - r_bar. A group of [0,0,0,0,0,0,0,0] gives every sample zero
advantage; so does [1,1,1,1,1,1,1,1]. Only a mixed group like [0,1,0,0,1,0,1,0]
has anything to amplify.

The Level 1-5 labels are a *static heuristic* over the ground-truth actions and
instruction text, kept because they make the curriculum legible to a human.
The empirical success fraction is what actually selects tasks.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from taurl import config
from taurl.metrics import TaskStats, load_results

# Frontier band on per-task pass@1 (inclusive).
FRONTIER_LO = 0.25
FRONTIER_HI = 0.75

# Kept for backwards compatibility; the per-env sets live in config.
WRITE_TOOLS = config.WRITE_TOOLS["retail"]

# Instruction phrasing that signals the customer withholds information until
# asked -- the "conversational uncertainty" category.
UNCERTAINTY_MARKERS = (
    "unsure",
    "indecisive",
    "not sure",
    "don't remember",
    "do not remember",
    "figure out",
    "don't know",
    "if asked",
    "only if",
    "insist",
    "reluctant",
)

# Actions that require resolving a product variant before the write can happen.
LOOKUP_TOOLS = {
    "exchange_delivered_order_items",
    "modify_pending_order_items",
}


def load_tasks(split: str, env: str = "retail"):
    """Import the raw tau-bench task objects for a split."""
    module = f"tau_bench.envs.{env}.tasks_{split}"
    mod = __import__(module, fromlist=["*"])
    for attr in ("TASKS_TEST", "TASKS_TRAIN", "TASKS_DEV", "TASKS"):
        if hasattr(mod, attr):
            return getattr(mod, attr)
    raise RuntimeError(f"no task list found in {module}")


@dataclass
class TaskFeatures:
    task_id: int
    n_write_actions: int
    n_distinct_write_tools: int
    has_output_assertions: bool
    instruction_words: int
    uncertainty_markers: int
    needs_variant_lookup: bool

    @property
    def level(self) -> int:
        """Heuristic difficulty label. Precedence is deliberate: compound beats
        everything, then conversational uncertainty, then policy, then lookup."""
        if self.n_write_actions >= 3 or self.n_distinct_write_tools >= 2:
            return 4  # compound: several distinct mutations in one conversation
        if self.uncertainty_markers >= 2:
            return 5  # customer withholds; agent must ask the right questions
        if self.has_output_assertions:
            return 3  # agent must also report specific information back
        if self.needs_variant_lookup or self.n_write_actions >= 2:
            return 2  # lookup + decision before the write
        return 1  # single action


def extract_features(task_id: int, task) -> TaskFeatures:
    writes = [a for a in task.actions if a.name in WRITE_TOOLS]
    instruction = (task.instruction or "").lower()
    return TaskFeatures(
        task_id=task_id,
        n_write_actions=len(writes),
        n_distinct_write_tools=len({a.name for a in writes}),
        has_output_assertions=len(task.outputs) > 0,
        instruction_words=len(instruction.split()),
        uncertainty_markers=sum(m in instruction for m in UNCERTAINTY_MARKERS),
        needs_variant_lookup=any(a.name in LOOKUP_TOOLS for a in writes),
    )


def build_rows(tasks: dict[int, TaskStats], split: str) -> list[dict]:
    raw = load_tasks(split)
    rows = []
    for tid, stats in sorted(tasks.items()):
        f = extract_features(tid, raw[tid])
        p1 = stats.c / stats.n if stats.n else 0.0
        rows.append(
            {
                "task_id": tid,
                "n": stats.n,
                "c": stats.c,
                "pass@1": round(p1, 4),
                "frontier": int(FRONTIER_LO <= p1 <= FRONTIER_HI),
                "level": f.level,
                "n_write_actions": f.n_write_actions,
                "n_distinct_write_tools": f.n_distinct_write_tools,
                "has_output_assertions": int(f.has_output_assertions),
                "needs_variant_lookup": int(f.needs_variant_lookup),
                "uncertainty_markers": f.uncertainty_markers,
                "instruction_words": f.instruction_words,
            }
        )
    return rows


def report(rows: list[dict]) -> str:
    n = len(rows)
    frontier = [r for r in rows if r["frontier"]]
    too_easy = [r for r in rows if r["pass@1"] > FRONTIER_HI]
    too_hard = [r for r in rows if r["pass@1"] < FRONTIER_LO]
    dead = [r for r in rows if r["c"] == 0]
    solved = [r for r in rows if r["n"] and r["c"] == r["n"]]

    lines = [
        "## Training frontier",
        "",
        f"Criterion: per-task pass@1 (= c/n) in [{FRONTIER_LO:.2f}, {FRONTIER_HI:.2f}].",
        "",
        f"- **frontier: {len(frontier)}/{n} tasks ({len(frontier)/n:.0%})** "
        "-- usable RL signal",
        f"- too easy (pass@1 > {FRONTIER_HI:.2f}): {len(too_easy)}",
        f"- too hard (pass@1 < {FRONTIER_LO:.2f}): {len(too_hard)}",
        f"- degenerate all-fail groups (c=0): {len(dead)} -- zero advantage",
        f"- degenerate all-pass groups (c=n): {len(solved)} -- zero advantage",
        "",
        "### By difficulty level",
        "",
        "| level | tasks | mean pass@1 | in frontier |",
        "|---|---|---|---|",
    ]
    for lvl in sorted({r["level"] for r in rows}):
        sub = [r for r in rows if r["level"] == lvl]
        mean = sum(r["pass@1"] for r in sub) / len(sub)
        nf = sum(r["frontier"] for r in sub)
        lines.append(f"| {lvl} | {len(sub)} | {mean:.1%} | {nf} |")

    lines += [
        "",
        "### Success-count histogram",
        "",
        "| c/n | tasks |",
        "|---|---|",
    ]
    nmax = max((r["n"] for r in rows), default=0)
    for c in range(nmax + 1):
        cnt = sum(1 for r in rows if r["c"] == c)
        bar = "#" * cnt
        lines.append(f"| {c}/{nmax} | {cnt} {bar} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path, help="sweep dir")
    ap.add_argument("--split", choices=["dev", "test", "train"], required=True)
    ap.add_argument("--csv", type=Path, default=config.REPORTS_DIR / "frontier.csv")
    ap.add_argument("--md", type=Path, default=config.REPORTS_DIR / "frontier.md")
    args = ap.parse_args()

    tasks = load_results(args.results)
    if not tasks:
        raise SystemExit(f"no results found in {args.results}")
    rows = build_rows(tasks, args.split)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    text = report(rows)
    args.md.write_text(text)
    print(text)
    print(f"wrote {len(rows)} rows -> {args.csv}")

    ids = [r["task_id"] for r in rows if r["frontier"]]
    (args.csv.parent / "frontier_task_ids.json").write_text(json.dumps(ids))
    print(f"frontier task ids -> {args.csv.parent / 'frontier_task_ids.json'}")


if __name__ == "__main__":
    main()
