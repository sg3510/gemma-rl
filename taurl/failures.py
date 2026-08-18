"""Why did a rollout fail? A taxonomy computed post-hoc from saved trajectories.

Costs nothing extra -- tau-bench already stores the full message trajectory in
its results JSON. The point is to separate *mechanical* failures (malformed tool
calls, running out of steps) from *capability* failures (wrong tool, wrong
arguments, policy violation, incomplete compound request).

That distinction decides what to do next. If the model is at 12% because a third
of its rollouts ran out of steps, the fix is a harness setting. If it is at 12%
because it confidently issues the wrong mutation, that is the thing RL is for.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from taurl import config
from taurl.metrics import is_errored, is_success
from taurl.triage import WRITE_TOOLS

# Ordered by precedence: the first matching category wins, most-mechanical first.
CATEGORIES = [
    ("infra_error", "Rollout crashed (provider/API). Excluded from metrics."),
    ("step_cap", "Hit the step cap without the conversation ending."),
    ("unknown_tool", "Called a tool that does not exist."),
    ("tool_arg_error", "Called a real tool with arguments the env rejected."),
    ("no_mutation", "Never called any write tool, though the task required one."),
    ("missing_output", "Did the right mutations but never stated required info."),
    ("wrong_mutation", "Wrote to the database, but the final state was wrong."),
    ("other", "Failed for a reason not captured above."),
]


def classify(entry: dict) -> tuple[str, str]:
    """Return (category, a short piece of supporting evidence)."""
    info = entry.get("info") or {}
    traj = entry.get("traj") or []

    if is_errored(entry):
        return "infra_error", str(info.get("error"))[:200]

    # What the environment said back to the agent.
    tool_obs = [m for m in traj if m.get("role") == "tool"]
    unknown = [m for m in tool_obs if str(m.get("content", "")).startswith("Unknown action")]
    errors = [m for m in tool_obs if str(m.get("content", "")).startswith("Error:")]

    # Did the agent ever attempt a state mutation?
    called = []
    for m in traj:
        for tc in m.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name")
            if name:
                called.append(name)
    wrote = [c for c in called if c in WRITE_TOOLS and c != "transfer_to_human_agents"]

    conversation_ended = any(
        "###STOP###" in str(m.get("content") or "")
        for m in traj
        if m.get("role") == "user"
    )
    agent_steps = sum(1 for m in traj if m.get("role") == "assistant")

    if not conversation_ended and agent_steps >= config.MAX_NUM_STEPS:
        return "step_cap", f"{agent_steps} agent steps, conversation never ended"
    if unknown:
        return "unknown_tool", str(unknown[0]["content"])[:200]
    if errors:
        return "tool_arg_error", str(errors[0]["content"])[:200]

    task = info.get("task") or {}
    required_writes = [
        a for a in (task.get("actions") or []) if a.get("name") in WRITE_TOOLS
    ]
    if required_writes and not wrote:
        return "no_mutation", f"task needed {len(required_writes)} write(s), agent made 0"

    reward_info = info.get("reward_info") or {}
    ri = reward_info.get("info") or {}
    if "outputs" in ri:
        missed = [k for k, v in (ri.get("outputs") or {}).items() if not v]
        if missed:
            return "missing_output", f"never stated: {missed[:3]}"
    if wrote:
        return "wrong_mutation", f"called {sorted(set(wrote))}"
    return "other", ""


def analyse(sweep_dir: Path) -> tuple[Counter, dict[str, list], int, int]:
    files = sorted(sweep_dir.rglob("*.json"))
    counts: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)
    total = 0
    passed = 0
    seen: set[tuple] = set()

    for f in files:
        if f.name == "meta.json":
            continue
        try:
            entries = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(entries, list):
            continue
        trial_dir = next(
            (q.name for q in f.parents if q.name.startswith("trial_")), "trial_?"
        )
        for e in entries:
            if not isinstance(e, dict) or "task_id" not in e:
                continue
            key = (e["task_id"], trial_dir)
            if key in seen:
                continue
            seen.add(key)
            total += 1
            if is_success(float(e.get("reward", 0))) and not is_errored(e):
                passed += 1
                continue
            cat, evidence = classify(e)
            counts[cat] += 1
            if len(examples[cat]) < 3:
                examples[cat].append((e["task_id"], trial_dir, evidence))
    return counts, examples, total, passed


def report(counts: Counter, examples: dict, total: int, passed: int) -> str:
    failed = total - passed
    if not failed:
        return (
            "# Failure taxonomy\n\n"
            f"{total} rollouts, **all {passed} passed**. Nothing to classify.\n"
        )
    lines = [
        "# Failure taxonomy",
        "",
        f"{total} rollouts: **{passed} passed**, {failed} did not.",
        "",
        "| category | count | % of failures | what it means |",
        "|---|---|---|---|",
    ]
    for cat, desc in CATEGORIES:
        n = counts.get(cat, 0)
        if not n:
            continue
        lines.append(f"| `{cat}` | {n} | {n / failed:.0%} | {desc} |")

    mech = sum(counts.get(c, 0) for c in ("step_cap", "unknown_tool", "tool_arg_error"))
    cap = sum(counts.get(c, 0) for c in ("no_mutation", "missing_output", "wrong_mutation"))
    lines += [
        "",
        f"**Mechanical** (harness/format problems): {mech} "
        f"({mech / failed:.0%} of failures)",
        f"**Capability** (the thing RL is for): {cap} ({cap / failed:.0%} of failures)",
        "",
        "## Evidence",
        "",
    ]
    for cat, _ in CATEGORIES:
        if not examples.get(cat):
            continue
        lines.append(f"**`{cat}`**")
        lines.append("")
        for tid, trial, ev in examples[cat]:
            lines.append(f"- task {tid} ({trial}): `{ev}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--md", type=Path, default=config.REPORTS_DIR / "failures.md")
    args = ap.parse_args()

    counts, examples, total, passed = analyse(args.results)
    if not total:
        raise SystemExit(f"no rollouts found in {args.results}")
    text = report(counts, examples, total, passed)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
