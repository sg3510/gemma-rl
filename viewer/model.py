"""The one data shape the viewer knows about.

Every eval we might want to look at -- tau2 telecom today, tau2 airline/retail,
tau1 retail/airline, and whatever a trained policy produces later -- gets an
adapter that lowers it into these dataclasses. The renderer, the metrics and
the HTML never see a benchmark-specific field, so adding an eval means writing
one adapter, not touching the viewer.

The deliberate choices here:

* An `Attempt` is a *cell* -- one (task, trial) slot -- and it may be MISSING.
  tau2 wrote 160 rollouts for `pinned_telecom` but 41 died on infrastructure and
  were never retried, so 23 of 40 tasks have fewer than 4 scored attempts. A
  viewer that only renders the rows it found would hide exactly the gap that
  stops us reporting pass@4. Missing cells are first-class.

* `Check` covers both what the task ASKED for (`Task.ideal`) and what the
  verifier FOUND (`Attempt.checks`), because the interesting view is the two
  side by side. tau2's reward is a checklist, not a database diff -- `db_check`
  is routinely False on a task that scored 1.0, because telecom's `reward_basis`
  is `ENV_ASSERTION`. `Check.counts` records whether a check gated the reward.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PASS = "pass"
FAIL = "fail"
ERROR = "error"
MISSING = "missing"


@dataclass
class Message:
    """One turn. `requestor` matters: telecom lets the *customer* call tools on
    their own phone, so 'who acted' is not recoverable from `role` alone."""

    role: str
    content: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    requestor: str | None = None
    turn: int | None = None
    tool_call_id: str | None = None

    def as_json(self) -> dict:
        out = {"role": self.role}
        if self.content:
            out["content"] = self.content
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        if self.requestor:
            out["requestor"] = self.requestor
        if self.turn is not None:
            out["turn"] = self.turn
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        return out


@dataclass
class Check:
    """A verifier criterion, either expected (on a Task) or evaluated (on an
    Attempt). `met` is None when the check was defined but never scored."""

    kind: str  # action | env_assertion | nl_assertion | communicate | db
    label: str
    met: bool | None = None
    counts: bool = True  # did this gate the reward (in reward_basis)?
    detail: dict = field(default_factory=dict)

    def as_json(self) -> dict:
        out = {"kind": self.kind, "label": self.label, "counts": self.counts}
        if self.met is not None:
            out["met"] = self.met
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass
class Attempt:
    task_id: str
    trial: int
    status: str = MISSING
    reward: float | None = None
    termination_reason: str | None = None
    duration: float | None = None
    cost: float | None = None
    checks: list[Check] = field(default_factory=list)
    reward_breakdown: dict = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    note: str | None = None  # error text, for ERROR cells

    @property
    def scored(self) -> bool:
        """Scored cells are the denominator. An infrastructure crash is not a
        task failure -- counting it as one is bug B1, which this project has now
        hit twice (once in tau-bench's runner, once in its own tau2 reader)."""
        return self.status in (PASS, FAIL)

    def as_json(self) -> dict:
        out = {
            "task_id": self.task_id,
            "trial": self.trial,
            "status": self.status,
        }
        for key in ("reward", "termination_reason", "duration", "cost", "note"):
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        if self.checks:
            out["checks"] = [c.as_json() for c in self.checks]
        if self.reward_breakdown:
            out["reward_breakdown"] = self.reward_breakdown
        if self.messages:
            out["messages"] = [m.as_json() for m in self.messages]
        return out


@dataclass
class Task:
    id: str
    short: str  # display label; the full tau2 id is unreadable in a grid header
    split: str | None = None
    splits: list[str] = field(default_factory=list)  # every split it belongs to
    groups: dict = field(default_factory=dict)  # scenario, persona, ...
    goal: str = ""  # what the customer wants, in their words
    purpose: str | None = None
    ideal: list[Check] = field(default_factory=list)
    reward_basis: list[str] = field(default_factory=list)

    def as_json(self) -> dict:
        return {
            "id": self.id,
            "short": self.short,
            "split": self.split,
            "splits": self.splits,
            "groups": self.groups,
            "goal": self.goal,
            "purpose": self.purpose,
            "ideal": [c.as_json() for c in self.ideal],
            "reward_basis": self.reward_basis,
        }


@dataclass
class Run:
    """One eval run: a set of tasks, each attempted `trials_planned` times."""

    name: str
    family: str  # tau2 | tau1 | ...
    domain: str
    source: str
    tasks: list[Task] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    trials_planned: int = 1
    timestamp: str | None = None
    stack: dict = field(default_factory=dict)  # the pinned serving stack
    notes: list[str] = field(default_factory=list)

    def fill_missing(self) -> None:
        """Materialise a cell for every (task, trial) the run intended to run.

        Without this the grid silently shrinks to whatever completed, and a run
        that lost a quarter of its rollouts looks identical to one that did not.
        """
        seen = {(a.task_id, a.trial) for a in self.attempts}
        for task in self.tasks:
            for trial in range(self.trials_planned):
                if (task.id, trial) not in seen:
                    self.attempts.append(Attempt(task_id=task.id, trial=trial))
        self.attempts.sort(key=lambda a: (a.task_id, a.trial))
