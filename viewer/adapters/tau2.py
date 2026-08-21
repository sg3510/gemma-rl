"""Adapter for tau2-bench runs (`.../simulations/<run>/results.json`).

tau2 packs everything into one file: the run's serving stack under `info`, the
task definitions (including `evaluation_criteria`, i.e. the ideal end state)
under `tasks`, and every rollout under `simulations`. So one file in, one Run
out -- the only external lookup is split membership.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from viewer import splits
from viewer.model import ERROR, FAIL, PASS, Attempt, Check, Message, Run, Task

SUCCESS_EPS = 1e-6
MAX_CONTENT_CHARS = 20_000  # tool results can be whole DB dumps

# tau2 gates the reward on the product of everything in `reward_basis`; checks
# outside it still run, for diagnostics only. Telecom's basis is ENV_ASSERTION,
# which is why `db_check` is False on plenty of traces that scored 1.0.
BASIS_FOR_KIND = {
    "action": ("ACTION", "DB"),  # DB targets are produced by replaying actions
    "env_assertion": ("ENV_ASSERTION",),
    "nl_assertion": ("NL_ASSERTION",),
    "communicate": ("COMMUNICATE",),
    "db": ("DB",),
}

# tau2 telecom ids look like:
#   [mms_issue]airplane_mode_on|bad_wifi_calling|...[PERSONA:Hard]
# retail and airline just use "0", "1", ... -- the regex simply misses and we
# fall back to the raw id.
TELECOM_ID = re.compile(r"^\[(?P<scenario>[^\]]+)\](?P<faults>[^\[]*)"
                        r"(?:\[PERSONA:(?P<persona>[^\]]+)\])?$")


def detect(path: Path) -> bool:
    return _results_file(path) is not None


def _results_file(path: Path) -> Path | None:
    if path.is_dir():
        candidate = path / "results.json"
        return candidate if candidate.exists() else None
    if path.name.endswith(".json") and path.exists():
        try:
            head = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return path if isinstance(head, dict) and "simulations" in head else None
    return None


def load(path: Path, name: str | None = None) -> Run:
    results = _results_file(path)
    if results is None:
        raise ValueError(f"not a tau2 results file: {path}")
    blob = json.loads(results.read_text())
    info = blob.get("info") or {}
    env_info = info.get("environment_info") or {}
    agent_info = info.get("agent_info") or {}
    user_info = info.get("user_info") or {}
    domain = env_info.get("domain_name") or "unknown"

    sims = blob.get("simulations") or []
    trials_planned = max(
        [int(info.get("num_trials") or 1)]
        + [int(s.get("trial", 0)) + 1 for s in sims]
    )

    run = Run(
        name=name or results.parent.name,
        family="tau2",
        domain=domain,
        source=str(results),
        timestamp=blob.get("timestamp"),
        trials_planned=trials_planned,
        stack={
            "agent_model": agent_info.get("llm"),
            "agent_args": agent_info.get("llm_args") or {},
            "user_model": user_info.get("llm"),
            "user_args": user_info.get("llm_args") or {},
            "max_steps": info.get("max_steps"),
            "max_errors": info.get("max_errors"),
            "seed": info.get("seed"),
            "git_commit": info.get("git_commit"),
        },
    )

    run.tasks = [_task(raw, domain) for raw in blob.get("tasks") or []]
    known = {t.id for t in run.tasks}

    best: dict[tuple[str, int], Attempt] = {}
    for sim in sims:
        attempt = _attempt(sim)
        if attempt.task_id not in known:
            continue
        key = (attempt.task_id, attempt.trial)
        prior = best.get(key)
        # A scored rollout always beats an errored one for the same cell;
        # otherwise the later write wins.
        if prior is None or attempt.scored or not prior.scored:
            best[key] = attempt
    run.attempts = list(best.values())
    run.fill_missing()

    if info.get("num_trials") and trials_planned != info["num_trials"]:
        run.notes.append(
            f"run info says num_trials={info['num_trials']} but trials 0-"
            f"{trials_planned - 1} are present (resumed or re-run)."
        )
    return run


def _task(raw: dict, domain: str) -> Task:
    task_id = str(raw.get("id"))
    scenario = raw.get("user_scenario") or {}
    instructions = scenario.get("instructions") or {}
    criteria = raw.get("evaluation_criteria") or {}
    basis = [str(b) for b in (criteria.get("reward_basis") or [])]

    groups: dict = {}
    match = TELECOM_ID.match(task_id)
    short = task_id
    if match:
        groups["scenario"] = match.group("scenario")
        if match.group("persona"):
            groups["persona"] = match.group("persona")
        faults = [f for f in (match.group("faults") or "").split("|") if f]
        if faults:
            groups["faults"] = len(faults)
        short = " + ".join(faults) or task_id
    elif raw.get("description", {}).get("purpose"):
        groups["purpose"] = raw["description"]["purpose"][:60]

    goal = "\n\n".join(
        part
        for part in (
            instructions.get("reason_for_call"),
            instructions.get("known_info"),
            instructions.get("task_instructions"),
        )
        if part
    )

    return Task(
        id=task_id,
        short=short,
        split=splits.primary(domain, task_id),
        splits=splits.memberships(domain, task_id),
        groups=groups,
        goal=goal,
        purpose=(raw.get("description") or {}).get("purpose"),
        ideal=_ideal(criteria, basis),
        reward_basis=basis,
    )


def _ideal(criteria: dict, basis: list[str]) -> list[Check]:
    out: list[Check] = []
    for action in criteria.get("actions") or []:
        out.append(
            Check(
                kind="action",
                label=_action_label(action),
                counts=_counts("action", basis),
                detail={
                    "requestor": action.get("requestor"),
                    "arguments": action.get("arguments"),
                },
            )
        )
    for assertion in criteria.get("env_assertions") or []:
        out.append(
            Check(
                kind="env_assertion",
                label=_assertion_label(assertion),
                counts=_counts("env_assertion", basis),
                detail={"env_type": assertion.get("env_type")},
            )
        )
    for assertion in criteria.get("nl_assertions") or []:
        out.append(
            Check(
                kind="nl_assertion",
                label=str(assertion),
                counts=_counts("nl_assertion", basis),
            )
        )
    for item in criteria.get("communicate_info") or []:
        out.append(
            Check(
                kind="communicate",
                label=f"communicates {item!r}",
                counts=_counts("communicate", basis),
            )
        )
    if "DB" in basis:
        out.append(
            Check(
                kind="db",
                label="final DB state matches the target",
                counts=True,
                detail={"note": "target = replaying the expected actions"},
            )
        )
    return out


def _counts(kind: str, basis: list[str]) -> bool:
    return any(b in basis for b in BASIS_FOR_KIND.get(kind, ()))


def _action_label(action: dict) -> str:
    args = action.get("arguments") or {}
    rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    who = action.get("requestor")
    prefix = f"{who}: " if who and who != "assistant" else ""
    return f"{prefix}{action.get('name')}({rendered})"


def _assertion_label(assertion: dict) -> str:
    args = assertion.get("arguments") or {}
    rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{assertion.get('func_name')}({rendered})"


def _attempt(sim: dict) -> Attempt:
    reward_info = sim.get("reward_info") or {}
    reward = float(reward_info.get("reward") or 0.0)
    reason = sim.get("termination_reason")
    if reason == "infrastructure_error":
        status = ERROR
    elif reward > 1 - SUCCESS_EPS:
        status = PASS
    else:
        status = FAIL

    cost = None
    if sim.get("agent_cost") is not None or sim.get("user_cost") is not None:
        cost = float(sim.get("agent_cost") or 0.0) + float(sim.get("user_cost") or 0.0)

    # A run that exhausts the step cap is never handed to the verifier at all:
    # tau2 records reward 0.0 with a note and every check list null. That is a
    # different thing from failing the checks, and the viewer says so.
    note = (reward_info.get("info") or {}).get("note")

    return Attempt(
        task_id=str(sim.get("task_id")),
        trial=int(sim.get("trial", 0)),
        status=status,
        reward=reward,
        termination_reason=reason,
        duration=sim.get("duration"),
        cost=cost,
        checks=_checks(reward_info),
        reward_breakdown=reward_info.get("reward_breakdown") or {},
        messages=[_message(m) for m in sim.get("messages") or []],
        note=note,
    )


def _checks(reward_info: dict) -> list[Check]:
    basis = [str(b) for b in (reward_info.get("reward_basis") or [])]
    out: list[Check] = []
    for item in reward_info.get("action_checks") or []:
        action = item.get("action") or {}
        out.append(
            Check(
                kind="action",
                label=_action_label(action),
                met=item.get("action_match"),
                counts=_counts("action", basis),
                detail={
                    "tool_type": item.get("tool_type"),
                    "requestor": action.get("requestor"),
                    "arguments": action.get("arguments"),
                },
            )
        )
    for item in reward_info.get("env_assertions") or []:
        assertion = item.get("env_assertion") or {}
        out.append(
            Check(
                kind="env_assertion",
                label=_assertion_label(assertion),
                met=item.get("met"),
                counts=_counts("env_assertion", basis),
                detail={"env_type": assertion.get("env_type")},
            )
        )
    for item in reward_info.get("nl_assertions") or []:
        out.append(
            Check(
                kind="nl_assertion",
                label=str(item.get("nl_assertion") or item),
                met=item.get("met"),
                counts=_counts("nl_assertion", basis),
                detail={"justification": item.get("justification")},
            )
        )
    for item in reward_info.get("communicate_checks") or []:
        out.append(
            Check(
                kind="communicate",
                label=f"communicates {item.get('info')!r}",
                met=item.get("met"),
                counts=_counts("communicate", basis),
                detail={"justification": item.get("justification")},
            )
        )
    db_check = reward_info.get("db_check")
    if db_check:
        out.append(
            Check(
                kind="db",
                label="final DB state matches the target",
                met=db_check.get("db_match"),
                counts=_counts("db", basis),
                detail={"db_reward": db_check.get("db_reward")},
            )
        )
    return out


def _message(msg: dict) -> Message:
    content = msg.get("content")
    if isinstance(content, str) and len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + "\n... [truncated by the viewer]"
    calls = []
    for call in msg.get("tool_calls") or []:
        calls.append(
            {
                "id": call.get("id"),
                "name": call.get("name"),
                "arguments": call.get("arguments"),
                "requestor": call.get("requestor"),
            }
        )
    return Message(
        role=str(msg.get("role")),
        content=content,
        tool_calls=calls,
        requestor=msg.get("requestor"),
        turn=msg.get("turn_idx"),
        tool_call_id=msg.get("id") if msg.get("role") == "tool" else None,
    )
