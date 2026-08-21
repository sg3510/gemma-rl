"""Adapter for tau1 sweeps produced by `taurl.run_sweep` (`results/<name>/`).

Layout is one directory per trial, each holding tau-bench's own results JSON:
a list of `{task_id, reward, info, traj, trial}`. Trial identity comes from the
`trial_<i>` directory, not from the `trial` field -- upstream stamps 0 on every
result when `num_trials=1`, which is how our runner invokes it.

Error classification is imported from `taurl.metrics` rather than re-derived:
an infrastructure crash is excluded from the denominator, but a rollout that
died on the model's own malformed tool JSON is a genuine failure and scores 0.
"""
from __future__ import annotations

import json
from pathlib import Path

from taurl.metrics import is_errored, is_success

from viewer.model import ERROR, FAIL, PASS, Attempt, Check, Message, Run, Task

MAX_CONTENT_CHARS = 20_000


def detect(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "meta.json").exists() and any(path.glob("trial_*"))


def load(path: Path, name: str | None = None) -> Run:
    if not detect(path):
        raise ValueError(f"not a tau1 sweep directory: {path}")
    meta = json.loads((path / "meta.json").read_text())

    run = Run(
        name=name or meta.get("sweep") or path.name,
        family="tau1",
        domain=meta.get("env") or "unknown",
        source=str(path),
        trials_planned=int(meta.get("trials") or 1),
        stack={
            "agent_model": meta.get("agent_model"),
            "agent_args": {"temperature": meta.get("agent_temperature")},
            "user_model": meta.get("user_model"),
            "user_args": {"temperature": meta.get("user_temperature")},
            "max_steps": meta.get("max_num_steps"),
            "tier": meta.get("tier"),
            "provider": meta.get("agent_provider"),
        },
        notes=[],
    )
    split = meta.get("split")

    # Oldest first so the most recent write wins, matching `metrics.scan_cells`.
    files = sorted(path.rglob("*.json"), key=lambda f: (f.stat().st_mtime, str(f)))
    cells: dict[tuple[str, int], Attempt] = {}
    tasks: dict[str, Task] = {}

    for file in files:
        if file.name == "meta.json":
            continue
        trial = _trial_of(file)
        if trial is None:
            continue
        try:
            entries = json.loads(file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "task_id" not in entry:
                continue
            task_id = str(entry["task_id"])
            attempt = _attempt(entry, task_id, trial)
            cells[(task_id, trial)] = attempt
            if task_id not in tasks or not tasks[task_id].ideal:
                task = _task(entry, task_id, split)
                if task is not None:
                    tasks[task_id] = task

    run.tasks = [tasks[k] for k in sorted(tasks, key=_numeric)]
    run.attempts = list(cells.values())
    run.fill_missing()
    return run


def _numeric(task_id: str):
    return (0, int(task_id)) if task_id.lstrip("-").isdigit() else (1, task_id)


def _trial_of(file: Path) -> int | None:
    for parent in file.parents:
        if parent.name.startswith("trial_"):
            try:
                return int(parent.name.split("_", 1)[1])
            except ValueError:
                return None
    return None


def _task(entry: dict, task_id: str, split: str | None) -> Task | None:
    info = entry.get("info")
    if not isinstance(info, dict):
        return None
    spec = info.get("task")
    if not isinstance(spec, dict):
        # Errored rollouts carry no task spec; a later trial will supply one.
        return Task(id=task_id, short=task_id, split=split, splits=[split] if split else [])
    ideal = [
        Check(
            kind="action",
            label=f"{action.get('name')}({_kwargs(action.get('kwargs'))})",
            detail={"kwargs": action.get("kwargs")},
        )
        for action in spec.get("actions") or []
    ]
    ideal += [
        Check(kind="communicate", label=f"states {out!r}")
        for out in spec.get("outputs") or []
    ]
    return Task(
        id=task_id,
        short=task_id,
        split=split,
        splits=[split] if split else [],
        groups={"user": spec.get("user_id")} if spec.get("user_id") else {},
        goal=spec.get("instruction") or "",
        ideal=ideal,
        # tau1 always gates on the DB hash plus required outputs.
        reward_basis=["DB", "COMMUNICATE"],
    )


def _kwargs(kwargs) -> str:
    if not isinstance(kwargs, dict):
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in kwargs.items())


def _attempt(entry: dict, task_id: str, trial: int) -> Attempt:
    reward = float(entry.get("reward") or 0.0)
    info = entry.get("info") if isinstance(entry.get("info"), dict) else {}
    if is_errored(entry):
        status = ERROR
    elif is_success(reward):
        status = PASS
    else:
        status = FAIL

    reward_info = info.get("reward_info") or {}
    detail = reward_info.get("info") or {}
    checks: list[Check] = []
    if "r_actions" in detail:
        checks.append(
            Check(
                kind="db",
                label="final DB state matches the target",
                met=bool(detail["r_actions"] == 1.0),
                detail={"gt_data_hash": detail.get("gt_data_hash")},
            )
        )
    for output, met in (detail.get("outputs") or {}).items():
        checks.append(
            Check(kind="communicate", label=f"states {output!r}", met=bool(met))
        )

    return Attempt(
        task_id=task_id,
        trial=trial,
        status=status,
        reward=reward,
        termination_reason="error" if status == ERROR else None,
        cost=info.get("user_cost"),
        checks=checks,
        messages=[_message(m) for m in entry.get("traj") or []],
        note=str(info.get("error"))[:400] if status == ERROR else None,
    )


def _message(msg: dict) -> Message:
    content = msg.get("content")
    if isinstance(content, str) and len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + "\n... [truncated by the viewer]"
    calls = []
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        arguments = fn.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass  # malformed tool JSON is itself a finding -- show it raw
        calls.append(
            {
                "id": call.get("id"),
                "name": fn.get("name"),
                "arguments": arguments,
                "requestor": "assistant",
            }
        )
    return Message(
        role=str(msg.get("role")),
        content=content,
        tool_calls=calls,
        tool_call_id=msg.get("tool_call_id"),
    )
