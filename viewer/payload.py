"""Turn a Run into the JSON the page reads.

Everything the browser needs is embedded in the HTML: no server, no fetch, so
a built page works over file:// and can be handed to somebody as one file.
Pruning tau2's per-message audio/raw fields takes `pinned_telecom` from 18.5 MB
to about 4 MB, which is comfortable to inline.
"""
from __future__ import annotations

import hashlib

from viewer import splits
from viewer.metrics import quality, summarize, task_stats
from viewer.model import Run

SCHEMA = 1


def task_ref(run_name: str, task_id: str) -> str:
    """Short, stable handle for a task within a run.

    tau2 task ids are 80-character fault lists -- unquotable in conversation.
    This gives every task a 4-char tag derived from its real id, so a human can
    say "T-a3f9" and the reference resolves to exactly one task, in this run and
    in any future run containing the same task.
    """
    digest = hashlib.sha1(task_id.encode()).hexdigest()[:4]
    return f"T-{digest}"


def attempt_ref(run_name: str, task_id: str, trial: int) -> str:
    """Handle for one rollout: <run>/<task-ref>#<trial>."""
    return f"{run_name}/{task_ref(run_name, task_id)}#{trial}"

# The group the grid organises itself by, when a run has several. Scenario is
# telecom's difficulty ladder (9% to 57% pass@1), which is the split that
# actually changes how you read the numbers.
GROUP_PREFERENCE = ["scenario", "purpose", "user"]


def primary_group(run: Run) -> str | None:
    keys = {k for t in run.tasks for k in t.groups}
    for preferred in GROUP_PREFERENCE:
        if preferred in keys:
            return preferred
    return sorted(keys)[0] if keys else None


def build(run: Run) -> dict:
    stats = task_stats(run)
    split_counts: dict[str, int] = {}
    for task in run.tasks:
        if task.split:
            split_counts[task.split] = split_counts.get(task.split, 0) + 1

    tasks = []
    for task in run.tasks:
        blob = task.as_json()
        blob["stats"] = stats[task.id]
        blob["ref"] = task_ref(run.name, task.id)
        tasks.append(blob)

    attempts = []
    for attempt in run.attempts:
        blob = attempt.as_json()
        blob["ref"] = attempt_ref(run.name, attempt.task_id, attempt.trial)
        if attempt.messages:
            blob["quality"] = quality(attempt)
        attempts.append(blob)

    return {
        "schema": SCHEMA,
        "run": {
            "name": run.name,
            "family": run.family,
            "domain": run.domain,
            "source": run.source,
            "timestamp": run.timestamp,
            "trials_planned": run.trials_planned,
            "stack": run.stack,
            "splits": split_counts,
            "domain_splits": splits.known_splits(run.domain),
            "notes": run.notes,
            "primary_group": primary_group(run),
        },
        "summary": summarize(run),
        "tasks": tasks,
        "attempts": attempts,
    }


def index_entry(payload: dict, href: str) -> dict:
    """The slim per-run record the index page uses -- no traces, so the index
    stays small however many runs accumulate."""
    return {
        "name": payload["run"]["name"],
        "family": payload["run"]["family"],
        "domain": payload["run"]["domain"],
        "timestamp": payload["run"]["timestamp"],
        "trials_planned": payload["run"]["trials_planned"],
        "splits": payload["run"]["splits"],
        "stack": payload["run"]["stack"],
        "summary": payload["summary"],
        "href": href,
        "tasks": {
            t["id"]: {
                "short": t["short"],
                "split": t["split"],
                "groups": t["groups"],
                **t["stats"],
            }
            for t in payload["tasks"]
        },
    }
