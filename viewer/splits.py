"""Which split does a task belong to?

Answered from tau2's own `split_tasks.json`, never from the run's name. That
matters for one reason: once we start training, the question "did this eval
touch the training set" has to be answerable from the data, not from whether
somebody named the directory honestly.

tau2 telecom ships small=20, train=74, test=40, base=114, full=2285. train and
test are disjoint and both sit inside base, so a task can legitimately carry
several labels; `primary` picks the most specific one.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS_DIR = REPO_ROOT / "vendor" / "tau2-bench" / "data" / "tau2" / "domains"

# Most specific first. `full` and `base` are supersets and only used as a
# fallback label when nothing narrower applies.
PRECEDENCE = ["test", "train", "small", "base", "full"]


@lru_cache(maxsize=None)
def _splits(domain: str) -> dict[str, frozenset[str]]:
    path = DOMAINS_DIR / domain / "split_tasks.json"
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    return {k: frozenset(str(t) for t in v) for k, v in blob.items()}


def memberships(domain: str, task_id: str) -> list[str]:
    """Every split containing this task, most specific first."""
    splits = _splits(domain)
    found = [name for name in PRECEDENCE if task_id in splits.get(name, ())]
    # Any split we do not know the precedence of still gets reported.
    found += sorted(
        name
        for name, ids in splits.items()
        if name not in PRECEDENCE and task_id in ids
    )
    return found


def primary(domain: str, task_id: str) -> str | None:
    found = memberships(domain, task_id)
    return found[0] if found else None


def known_splits(domain: str) -> dict[str, int]:
    return {k: len(v) for k, v in _splits(domain).items()}
