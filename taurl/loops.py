"""Tool-call loop detection and taxonomy for tau2 runs.

The finding this measures (reports/tool_call_looping.md, D29): a large share of
telecom failures are not capability failures but lock-ups -- the agent repeats
one identical read call until the step cap kills the episode. This module makes
that measurable and classifiable so the claim survives re-runs.

Definitions, chosen against the data rather than by taste:

* Only the AGENT's tool calls are counted. Customer device calls are excluded
  -- see `analyse_sim`.
* A trace LOOPS if one exact agent call signature (tool name + canonicalised
  arguments) occurs >= LOOP_TOTAL times (default 10). Passing traces top out at
  8 total repeats, so 10 is conservative for "this trace died of repetition".
* MAX_CONSECUTIVE is reported separately because it is the actionable number:
  no passing trace exceeds 6 consecutive identical calls, which is what makes a
  runtime breaker at >=7 outcome-neutral. Recalibrate this whenever the trace
  pool grows -- it moved from 4 to 6 when 22 repaired rollouts landed.

Taxonomy per looping trace -- the "passive polling" question:

* prior_write: did the agent make ANY state-mutating call before the loop began?
  (If not, it is polling for a change it did nothing to cause.)
* needs_user_action: does the task's ideal solution require the CUSTOMER to act
  on their device? The agent cannot see device state directly (user-side tool
  results never enter its context) and `get_details_by_id` reads the carrier DB,
  which device actions do not update -- so polling it can never observe the
  change the agent is waiting for.
* missing_agent_write: expected agent-side writes that never happened.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

LOOP_TOTAL = 10

# The RL frontier band: per-task pass@1 in [0.25, 0.75] gives mixed rollout
# groups, which is the condition GRPO needs. Kept identical to taurl/triage.py.
FRONTIER_LO = 0.25
FRONTIER_HI = 0.75
FRONTIER_MIN_TRIALS = 2
# Smallest run length no passing trace reaches. Calibrated on the REPAIRED
# pinned_telecom grid (141 scored): passes top out at 6 consecutive identical
# calls, and >=7 catches every failure that >=5 would while aborting none of
# them. An earlier value of 5 was set against 56 passes and was too tight.
BREAKER_CONSECUTIVE = 7

# Agent-side tools that mutate carrier state (from tau2 telecom's TelecomTools;
# everything else the agent can call is a read).
# Customer device actions that change the world -- after one of these, an agent
# re-read is legitimate verification, not polling.
USER_WRITES = {
    "toggle_airplane_mode", "set_network_mode_preference", "reseat_sim_card",
    "toggle_data", "toggle_roaming", "toggle_data_saver_mode",
    "set_apn_settings", "reset_apn_settings", "toggle_wifi",
    "toggle_wifi_calling", "connect_vpn", "disconnect_vpn",
    "grant_app_permission", "reboot_device", "make_payment",
}

AGENT_WRITES = {
    "suspend_line",
    "resume_line",
    "send_payment_request",
    "enable_roaming",
    "disable_roaming",
    "refuel_data",
}


def call_signature(call: dict) -> str:
    try:
        args = json.dumps(call.get("arguments"), sort_keys=True)
    except TypeError:
        args = str(call.get("arguments"))
    return f"{call.get('name')}::{args}"


def unproductive_runs(messages: list[dict]) -> tuple[str | None, int]:
    """Longest run of identical AGENT calls with nothing meaningful in between.

    "Meaningful" = anyone speaking, or any state-changing call by either party.
    Re-reading a setting after the customer toggles it -- or after the agent has
    said something and is checking again on their behalf -- is correct
    behaviour. What is not correct is a silent chain: the same read fired
    back-to-back with nothing but its own identical results in between. Raw consecutive-repeat counts conflate the
    two, which is why passing traces reach 22 consecutive identical calls.
    """
    best_sig, best = None, 0
    cur_sig, cur = None, 0
    for msg in messages:
        role = msg.get("role")
        calls = msg.get("tool_calls") or []
        if role in ("user", "assistant") and not calls and msg.get("content"):
            cur_sig, cur = None, 0  # somebody spoke: the chain is broken
            continue
        for call in calls:
            requestor = call.get("requestor") or "assistant"
            name = call.get("name")
            if requestor != "assistant":
                # A customer device action changes the world; a customer *read*
                # does not, so only writes reset the run.
                if name in USER_WRITES:
                    cur_sig, cur = None, 0
                continue
            if name in AGENT_WRITES:
                cur_sig, cur = None, 0
                continue
            sig = call_signature(call)
            cur = cur + 1 if sig == cur_sig else 1
            cur_sig = sig
            if cur > best:
                best_sig, best = sig, cur
    return best_sig, best


@dataclass
class TraceLoopStats:
    task_id: str
    trial: int
    outcome: str  # pass | termination_reason
    total_calls: int = 0
    unique_calls: int = 0
    max_total_repeat: int = 0
    max_consecutive: int = 0
    max_unproductive: int = 0
    unproductive_tool: str | None = None
    looped_tool: str | None = None
    prior_write: bool = False
    needs_user_action: bool = False
    missing_agent_writes: list[str] = field(default_factory=list)
    scenario: str | None = None

    @property
    def looped(self) -> bool:
        return self.max_total_repeat >= LOOP_TOTAL


def _scenario(task_id: str) -> str | None:
    if task_id.startswith("[") and "]" in task_id:
        return task_id[1 : task_id.index("]")]
    return None


def analyse_sim(sim: dict, task: dict | None) -> TraceLoopStats:
    reward = float((sim.get("reward_info") or {}).get("reward") or 0.0)
    stats = TraceLoopStats(
        task_id=str(sim.get("task_id")),
        trial=int(sim.get("trial", 0)),
        outcome="pass" if reward > 1 - 1e-6 else str(sim.get("termination_reason")),
        scenario=_scenario(str(sim.get("task_id"))),
    )

    # AGENT calls only. Telecom lets the CUSTOMER call device tools
    # (check_sim_status, can_send_mms, ...), and roughly half of all tool calls
    # in a trace are theirs. Counting those as agent duplicates is wrong twice
    # over: re-checking a setting after changing it is correct behaviour, and
    # the customer is a model we deliberately do not control or train.
    calls: list[dict] = [
        c
        for m in sim.get("messages") or []
        for c in (m.get("tool_calls") or [])
        if (c.get("requestor") or "assistant") == "assistant"
    ]
    sigs = [call_signature(c) for c in calls]
    stats.total_calls = len(sigs)
    counts = Counter(sigs)
    stats.unique_calls = len(counts)
    if counts:
        top_sig, top_n = counts.most_common(1)[0]
        stats.max_total_repeat = top_n
        run = best = 1
        for prev, cur in zip(sigs, sigs[1:]):
            run = run + 1 if cur == prev else 1
            best = max(best, run)
        stats.max_consecutive = best
        usig, ubest = unproductive_runs(sim.get("messages") or [])
        stats.max_unproductive = ubest
        stats.unproductive_tool = usig.split("::", 1)[0] if usig else None
        if stats.looped:
            stats.looped_tool = top_sig.split("::", 1)[0]
            onset = sigs.index(top_sig)
            stats.prior_write = any(
                c.get("name") in AGENT_WRITES
                and (c.get("requestor") or "assistant") == "assistant"
                for c in calls[:onset]
            )

    if task is not None:
        expected = (task.get("evaluation_criteria") or {}).get("actions") or []
        stats.needs_user_action = any(a.get("requestor") == "user" for a in expected)
        done_writes = {
            c.get("name")
            for c in calls
            if (c.get("requestor") or "assistant") == "assistant"
            and c.get("name") in AGENT_WRITES
        }
        stats.missing_agent_writes = [
            a["name"]
            for a in expected
            if a.get("requestor") == "assistant"
            and a.get("name") in AGENT_WRITES
            and a.get("name") not in done_writes
        ]
    return stats


def analyse(path: Path) -> list[TraceLoopStats]:
    """All scored (non-infrastructure) simulations of a tau2 results file."""
    if path.is_dir():
        path = path / "results.json"
    blob = json.loads(path.read_text())
    tasks = {str(t.get("id")): t for t in blob.get("tasks") or []}
    return [
        analyse_sim(sim, tasks.get(str(sim.get("task_id"))))
        for sim in blob.get("simulations") or []
        if sim.get("termination_reason") != "infrastructure_error"
    ]


def task_tags(rows: list[TraceLoopStats], rewards: dict[tuple[str, int], float],
              pair_gap: int = 3) -> dict[str, dict]:
    """Two-axis tags per task: reward variance (what GRPO needs) and efficiency
    variance (what DPO needs).

    They are maximised in *different* places. A task solved on every rollout has
    zero reward variance and is useless to GRPO -- but its successes still differ
    in length, which makes it an ideal preference pair: Tier-0 is held equal by
    construction, so the only thing being compared is efficiency.
    """
    by_task: dict[str, list[TraceLoopStats]] = {}
    for r in rows:
        by_task.setdefault(r.task_id, []).append(r)

    out: dict[str, dict] = {}
    for task_id, group in by_task.items():
        n = len(group)
        succ = [r for r in group if r.outcome == "pass"]
        c = len(succ)
        rate = c / n if n else None
        dups = [r.total_calls - r.unique_calls for r in succ]
        spread = (max(dups) - min(dups)) if len(dups) >= 2 else 0
        out[task_id] = {
            "n": n,
            "c": c,
            "pass1": rate,
            # Tier-0 axis
            "bucket": ("all-fail" if c == 0 else "all-pass" if c == n else "frontier"),
            "frontier": bool(n >= FRONTIER_MIN_TRIALS and rate is not None
                             and FRONTIER_LO <= rate <= FRONTIER_HI),
            # Tier-1 axis
            "has_loop": any(r.looped for r in group),
            "loops": sum(1 for r in group if r.looped),
            "max_consecutive": max((r.max_consecutive for r in group), default=0),
            "duplicate_spread": spread,
            "dpo_pairable": bool(len(dups) >= 2 and spread >= pair_gap),
        }
    return out


def summarize(rows: list[TraceLoopStats]) -> dict:
    loops = [r for r in rows if r.looped]
    passes = [r for r in rows if r.outcome == "pass"]
    clean_fail = [r for r in rows if r.outcome != "pass" and not r.looped]

    by_scenario = {}
    for scenario in sorted({r.scenario for r in rows if r.scenario}):
        sub = [r for r in rows if r.scenario == scenario]
        sub_nl = [r for r in sub if not r.looped]
        by_scenario[scenario] = {
            "scored": len(sub),
            "looped": sum(1 for r in sub if r.looped),
            "nonloop_pass_rate": (
                sum(1 for r in sub_nl if r.outcome == "pass") / len(sub_nl)
                if sub_nl
                else None
            ),
        }

    return {
        "scored": len(rows),
        "loops": len(loops),
        "loop_rate": len(loops) / len(rows) if rows else 0.0,
        "passes": len(passes),
        "clean_failures": len(clean_fail),
        "looped_tools": Counter(r.looped_tool for r in loops),
        "prior_write": sum(1 for r in loops if r.prior_write),
        "needs_user_action": sum(1 for r in loops if r.needs_user_action),
        "missing_agent_write": sum(1 for r in loops if r.missing_agent_writes),
        "max_consecutive_pass": max((r.max_consecutive for r in passes), default=0),
        "max_total_repeat_pass": max((r.max_total_repeat for r in passes), default=0),
        "median_calls": {
            "pass": median([r.total_calls for r in passes]) if passes else None,
            "loop": median([r.total_calls for r in loops]) if loops else None,
        },
        "by_scenario": by_scenario,
        "tags": _tag_rollup(rows),
        "breaker_would_abort_passes": sum(
            1 for r in passes if r.max_consecutive >= BREAKER_CONSECUTIVE
        ),
        "breaker_would_catch": sum(
            1
            for r in rows
            if r.outcome != "pass" and r.max_consecutive >= BREAKER_CONSECUTIVE
        ),
    }


def _tag_rollup(rows: list[TraceLoopStats]) -> dict:
    tags = task_tags(rows, {})
    buckets: dict[str, dict] = {}
    for t in tags.values():
        b = buckets.setdefault(t["bucket"], {"tasks": 0, "with_loop": 0, "pairable": 0})
        b["tasks"] += 1
        b["with_loop"] += int(t["has_loop"])
        b["pairable"] += int(t["dpo_pairable"])
    return {
        "buckets": buckets,
        "grpo_tasks": sum(1 for t in tags.values() if t["frontier"]),
        "dpo_tasks": sum(1 for t in tags.values() if t["dpo_pairable"]),
    }


def format_summary(s: dict, title: str) -> str:
    lines = [
        f"# Loop analysis: {title}",
        "",
        f"- scored rollouts: **{s['scored']}** "
        f"(pass {s['passes']}, looped {s['loops']}, clean failures {s['clean_failures']})",
        f"- **loop rate: {s['loop_rate']:.0%}** (>= {LOOP_TOTAL} identical calls)",
        f"- looped tools: "
        + ", ".join(f"`{t}` ×{n}" for t, n in s["looped_tools"].most_common()),
        f"- loops preceded by ANY agent-side write: **{s['prior_write']}/{s['loops']}** "
        "(the rest poll a state they did nothing to change)",
        f"- loops on tasks whose solution needs a CUSTOMER device action: "
        f"**{s['needs_user_action']}/{s['loops']}** "
        "(device state is invisible to the polled DB read)",
        f"- loops missing a required agent-side write: {s['missing_agent_write']}/{s['loops']}",
        "",
        "| scenario | scored | looped | non-loop pass rate |",
        "|---|---|---|---|",
    ]
    for scenario, g in s["by_scenario"].items():
        rate = "n/a" if g["nonloop_pass_rate"] is None else f"{g['nonloop_pass_rate']:.0%}"
        lines.append(f"| `{scenario}` | {g['scored']} | {g['looped']} | {rate} |")
    tg = s.get("tags") or {}
    if tg:
        lines += ["", "| bucket | tasks | with a loop | DPO-pairable |", "|---|---|---|---|"]
        for b, d in sorted(tg["buckets"].items()):
            lines.append(f"| {b} | {d['tasks']} | {d['with_loop']} | {d['pairable']} |")
        lines += [
            "",
            f"**GRPO set** (reward variance): {tg['grpo_tasks']} tasks. "
            f"**DPO set** (efficiency variance among successes): {tg['dpo_tasks']} tasks. "
            "They overlap only partly -- a task solved every time has no reward "
            "gradient but often makes the cleanest preference pair.",
        ]
    lines += [
        "",
        f"Breaker calibration: no passing trace exceeds "
        f"**{s['max_consecutive_pass']} consecutive** identical calls "
        f"({s['max_total_repeat_pass']} total), so a breaker at "
        f">={BREAKER_CONSECUTIVE} consecutive aborts "
        f"**{s['breaker_would_abort_passes']} passes** and catches "
        f"**{s['breaker_would_catch']}** failing traces early.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path, help="tau2 results.json or its directory")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    rows = analyse(args.results)
    print(format_summary(summarize(rows), args.title or args.results.name))


if __name__ == "__main__":
    main()
