"""Metrics over a normalised Run.

pass@k and pass^k come from `taurl.metrics` rather than being reimplemented, so
the viewer and the reports can never quietly disagree about the headline number.

Two conventions are reported side by side, because this project has already
published both and they differ by 4 points on `pinned_telecom`:

* **macro** -- mean of per-task success rates. This is what the paired
  before/after analysis in `reports/eval_protocol.md` commits us to, and what
  `taurl.tau2_results` prints (43.1%).
* **micro** -- successes / scored rollouts, ignoring which task they came from
  (47.1%, the README headline). Differs whenever tasks have unequal trial
  counts, which is precisely our situation.

pass@k is also reported with its **eligible task count**, because "pass@4" over
the 17 of 40 tasks that actually have 4 scored rollouts is a different claim
from pass@4 over all of them.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from taurl.metrics import is_success, pass_at_k, pass_hat_k

from viewer.model import ERROR, MISSING, PASS, Attempt, Run

# The RL frontier band: per-task pass@1 in [0.25, 0.75] produces mixed rollout
# groups, which is the condition GRPO needs. Tasks at 0/n or n/n give no
# gradient. Kept identical to taurl/triage.py.
FRONTIER_LO = 0.25
FRONTIER_HI = 0.75
FRONTIER_MIN_TRIALS = 2


def quality(attempt: Attempt) -> dict:
    """Tier-1 signals from `reports/tiered_reward.md`, computed on the trace.

    Exact duplicate calls (same tool, same arguments) are unambiguous waste --
    the agent re-fetching what it already has -- in a trace the verifier still
    scores perfect. Counts the AGENT's calls only: the customer simulator also
    calls tools (device checks), and those are neither the agent's behaviour nor
    something we train. 50 of 74 successes did it, mean 4.2 per trace.
    """
    seen: set[str] = set()
    duplicates = 0
    tool_calls = 0
    agent_turns = 0
    user_turns = 0
    prev_key = None
    run = max_run = 0
    for msg in attempt.messages:
        if msg.role == "assistant":
            agent_turns += 1
        elif msg.role == "user":
            user_turns += 1
        for call in msg.tool_calls:
            # Agent calls only -- customer device calls are not the agent's
            # behaviour and must not count toward its duplicate/loop metrics.
            if (call.get("requestor") or "assistant") != "assistant":
                continue
            tool_calls += 1
            key = f"{call.get('name')}::{_stable(call.get('arguments'))}"
            if key in seen:
                duplicates += 1
            seen.add(key)
            run = run + 1 if key == prev_key else 1
            max_run = max(max_run, run)
            prev_key = key
    return {
        "messages": len(attempt.messages),
        "agent_turns": agent_turns,
        "user_turns": user_turns,
        "tool_calls": tool_calls,
        "duplicate_calls": duplicates,
        # Longest run of back-to-back identical calls. No passing telecom trace
        # exceeds 4; >=10 total repeats is the "looped" criterion (taurl/loops.py).
        "max_consecutive_repeat": max_run,
    }


def _stable(args) -> str:
    import json

    try:
        return json.dumps(args, sort_keys=True)
    except TypeError:
        return str(args)


def task_stats(run: Run) -> dict[str, dict]:
    by_task: dict[str, list[Attempt]] = defaultdict(list)
    for attempt in run.attempts:
        by_task[attempt.task_id].append(attempt)

    stats: dict[str, dict] = {}
    for task in run.tasks:
        attempts = by_task.get(task.id, [])
        scored = [a for a in attempts if a.scored]
        n = len(scored)
        c = sum(1 for a in scored if is_success(a.reward or 0.0))
        rate = c / n if n else None
        succ_dups = [quality(a)["duplicate_calls"] for a in scored
                     if is_success(a.reward or 0.0) and a.messages]
        spread = (max(succ_dups) - min(succ_dups)) if len(succ_dups) >= 2 else 0
        max_consec = max((quality(a)["max_consecutive_repeat"] for a in attempts
                          if a.messages), default=0)
        stats[task.id] = {
            "n": n,
            "c": c,
            # Tier-1 axis: efficiency variance. A task solved on every rollout has
            # no reward gradient for GRPO but often makes the cleanest DPO pair.
            "max_consecutive": max_consec,
            "has_loop": bool(max_consec >= 10),
            "duplicate_spread": spread,
            "dpo_pairable": bool(len(succ_dups) >= 2 and spread >= 3),
            "errors": sum(1 for a in attempts if a.status == ERROR),
            "missing": sum(1 for a in attempts if a.status == MISSING),
            "pass1": rate,
            "frontier": bool(
                n >= FRONTIER_MIN_TRIALS
                and rate is not None
                and FRONTIER_LO <= rate <= FRONTIER_HI
            ),
        }
    return stats


def summarize(run: Run) -> dict:
    stats = task_stats(run)
    scored_tasks = {tid: s for tid, s in stats.items() if s["n"]}

    max_n = max((s["n"] for s in stats.values()), default=0)
    curve = []
    for k in range(1, max_n + 1):
        eligible = [s for s in scored_tasks.values() if s["n"] >= k]
        if not eligible:
            continue
        curve.append(
            {
                "k": k,
                "eligible_tasks": len(eligible),
                "coverage": len(eligible) / len(run.tasks) if run.tasks else 0.0,
                "pass_at_k": sum(pass_at_k(s["n"], s["c"], k) for s in eligible)
                / len(eligible),
                "pass_hat_k": sum(pass_hat_k(s["n"], s["c"], k) for s in eligible)
                / len(eligible),
            }
        )

    total_scored = sum(s["n"] for s in stats.values())
    total_success = sum(s["c"] for s in stats.values())
    frontier = [tid for tid, s in stats.items() if s["frontier"]]
    gradeable = [s for s in stats.values() if s["n"] >= FRONTIER_MIN_TRIALS]

    return {
        "num_tasks": len(run.tasks),
        "cells": len(run.attempts),
        "scored": total_scored,
        "successes": total_success,
        "errors": sum(1 for a in run.attempts if a.status == ERROR),
        "missing": sum(1 for a in run.attempts if a.status == MISSING),
        "pass1_macro": (
            sum(s["pass1"] for s in scored_tasks.values()) / len(scored_tasks)
            if scored_tasks
            else None
        ),
        "pass1_micro": total_success / total_scored if total_scored else None,
        "curve": curve,
        "full_depth_tasks": sum(
            1 for s in stats.values() if s["n"] >= run.trials_planned
        ),
        "frontier_tasks": len(frontier),
        "frontier_share": len(frontier) / len(gradeable) if gradeable else None,
        "termination": dict(
            Counter(
                a.termination_reason or "unknown"
                for a in run.attempts
                if a.status != MISSING
            ).most_common()
        ),
        "groups": group_breakdown(run, stats),
        "quality": quality_summary(run),
    }


def group_breakdown(run: Run, stats: dict[str, dict]) -> dict:
    """Per-group rollups -- telecom's scenario ladder runs 9% to 57%, and a
    single aggregate hides that. Any group key an adapter emits is rolled up."""
    keys = sorted({k for t in run.tasks for k in t.groups})
    out: dict[str, dict] = {}
    for key in keys:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for task in run.tasks:
            value = task.groups.get(key)
            if value is None:
                continue
            buckets[str(value)].append(stats[task.id])
        rows = {}
        for value, group in sorted(buckets.items()):
            scored = [s for s in group if s["n"]]
            gradeable = [s for s in group if s["n"] >= FRONTIER_MIN_TRIALS]
            rows[value] = {
                "tasks": len(group),
                "pass1_macro": (
                    sum(s["pass1"] for s in scored) / len(scored) if scored else None
                ),
                "frontier_tasks": sum(1 for s in group if s["frontier"]),
                "frontier_share": (
                    sum(1 for s in group if s["frontier"]) / len(gradeable)
                    if gradeable
                    else None
                ),
            }
        out[key] = rows
    return out


def quality_summary(run: Run) -> dict:
    """Tier-1 spread across *successes* only -- quality is a tie-break among
    successes, never a substitute for the success gate."""
    rows = [
        quality(a)
        for a in run.attempts
        if a.status == PASS and a.messages
    ]
    if not rows:
        return {}
    def med(key: str) -> float:
        vals = sorted(r[key] for r in rows)
        mid = len(vals) // 2
        return float(vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2)

    return {
        "successes_with_traces": len(rows),
        "median_messages": med("messages"),
        "median_tool_calls": med("tool_calls"),
        "median_duplicate_calls": med("duplicate_calls"),
        "traces_with_duplicates": sum(1 for r in rows if r["duplicate_calls"]),
    }
