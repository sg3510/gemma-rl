"""Tests for the trace viewer.

The pure-model tests always run. The ones that read real runs are skipped when
the data is not on disk -- `results/` is gitignored and the tau2 simulations
live inside the submodule's data directory, so a fresh clone has neither.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from viewer import adapters, payload, splits
from viewer.metrics import quality, summarize, task_stats
from viewer.model import ERROR, FAIL, MISSING, PASS, Attempt, Message, Run, Task

REPO_ROOT = Path(__file__).resolve().parent.parent
TAU2_RUN = REPO_ROOT / "vendor/tau2-bench/data/simulations/pinned_telecom"
# Numeric pins use the frozen pre-repair snapshot: the live results.json is a
# resumable checkpoint that later runs legitimately rewrite.
TAU2_SNAPSHOT = REPO_ROOT / "results/_snapshots/pinned_telecom_prerepair.json"
TAU1_RUN = REPO_ROOT / "results/prov_NextBit"


def _run(rewards: dict[str, list[str]], trials: int = 4) -> Run:
    """Build a Run from {task_id: [status, ...]}."""
    run = Run(name="t", family="tau2", domain="telecom", source="-", trials_planned=trials)
    run.tasks = [Task(id=tid, short=tid) for tid in rewards]
    for tid, statuses in rewards.items():
        for trial, status in enumerate(statuses):
            run.attempts.append(
                Attempt(
                    task_id=tid,
                    trial=trial,
                    status=status,
                    reward=1.0 if status == PASS else 0.0,
                )
            )
    run.fill_missing()
    return run


def test_missing_cells_are_materialised():
    run = _run({"a": [PASS, FAIL]}, trials=4)
    assert len(run.attempts) == 4
    assert [a.status for a in run.attempts] == [PASS, FAIL, MISSING, MISSING]


def test_errors_leave_the_denominator_but_missing_is_visible():
    run = _run({"a": [PASS, ERROR, FAIL]}, trials=4)
    stats = task_stats(run)["a"]
    # Tier-0 fields; the Tier-1 tags (loops, duplicate spread) are asserted
    # separately since they depend on trace content, not outcomes.
    assert {k: stats[k] for k in ("n", "c", "errors", "missing", "pass1", "frontier")} == {
        "n": 2, "c": 1, "errors": 1, "missing": 1, "pass1": 0.5, "frontier": True}
    assert stats["has_loop"] is False and stats["dpo_pairable"] is False


def test_pass_at_k_reports_how_many_tasks_were_eligible():
    # One task with 4 scored rollouts, one with 2: pass@4 covers half the run.
    run = _run({"a": [PASS, FAIL, PASS, FAIL], "b": [PASS, FAIL]}, trials=4)
    curve = {c["k"]: c for c in summarize(run)["curve"]}
    assert curve[1]["eligible_tasks"] == 2
    assert curve[4]["eligible_tasks"] == 1
    assert curve[4]["coverage"] == 0.5


def test_macro_and_micro_differ_when_trial_counts_are_uneven():
    run = _run({"a": [PASS, PASS, PASS, FAIL], "b": [FAIL]}, trials=4)
    s = summarize(run)
    assert s["pass1_macro"] == pytest.approx(0.375)  # mean of 0.75 and 0.0
    assert s["pass1_micro"] == pytest.approx(0.6)  # 3 successes / 5 rollouts


def test_frontier_needs_at_least_two_scored_rollouts():
    # 1/1 is a 100% success rate, not a frontier task -- binning by success
    # count alone would wrongly call it mixed.
    solo = task_stats(_run({"a": [PASS]}, trials=4))["a"]
    assert solo["frontier"] is False
    mixed = task_stats(_run({"a": [PASS, FAIL]}, trials=4))["a"]
    assert mixed["frontier"] is True


def test_quality_counts_exact_duplicate_calls():
    attempt = Attempt(
        task_id="a",
        trial=0,
        status=PASS,
        messages=[
            Message(role="assistant", tool_calls=[{"name": "get", "arguments": {"id": 1}}]),
            Message(role="tool", content="{}"),
            Message(role="assistant", tool_calls=[{"name": "get", "arguments": {"id": 1}}]),
            Message(role="assistant", tool_calls=[{"name": "get", "arguments": {"id": 2}}]),
            Message(role="user", content="thanks"),
        ],
    )
    q = quality(attempt)
    assert q == {"messages": 5, "agent_turns": 3, "user_turns": 1,
                 "tool_calls": 3, "duplicate_calls": 1,
                 "max_consecutive_repeat": 2}


def test_telecom_splits_are_disjoint_and_labelled_most_specific_first():
    sizes = splits.known_splits("telecom")
    assert sizes["train"] == 74 and sizes["test"] == 40 and sizes["base"] == 114
    test_ids = set(splits._splits("telecom")["test"])
    train_ids = set(splits._splits("telecom")["train"])
    assert not test_ids & train_ids
    any_test = next(iter(test_ids))
    assert splits.primary("telecom", any_test) == "test"
    assert splits.memberships("telecom", any_test)[0] == "test"
    assert splits.primary("telecom", "not-a-task") is None


@pytest.mark.skipif(not TAU2_SNAPSHOT.exists(), reason="snapshot not on disk")
def test_tau2_adapter_matches_the_numbers_taurl_reports():
    run = adapters.load(TAU2_SNAPSHOT)
    blob = payload.build(run)
    s = blob["summary"]
    assert run.family == "tau2" and run.domain == "telecom"
    assert s["num_tasks"] == 40
    assert s["cells"] == 160  # 40 tasks x 4 trials, nothing lost from the grid
    assert s["errors"] == 41  # infrastructure, excluded from the denominator
    assert s["scored"] == 119
    assert s["pass1_macro"] == pytest.approx(0.431, abs=5e-4)
    assert s["pass1_micro"] == pytest.approx(0.471, abs=5e-4)
    # Every task is on the held-out test split, and none leaks into train.
    assert blob["run"]["splits"] == {"test": 40}


@pytest.mark.skipif(not TAU2_SNAPSHOT.exists(), reason="snapshot not on disk")
def test_tau2_adapter_separates_unevaluated_from_failed():
    run = adapters.load(TAU2_SNAPSHOT)
    capped = [a for a in run.attempts if a.termination_reason == "max_steps"]
    assert capped, "expected step-cap terminations in this run"
    # tau2 never runs the verifier on a premature termination: no checks, and a
    # note saying why. Rendering that as "failed every check" would be a lie.
    assert all(not a.checks for a in capped)
    assert all(a.note and "prematurely" in a.note for a in capped)


@pytest.mark.skipif(not TAU1_RUN.exists(), reason="tau1 sweeps not on disk")
def test_tau1_adapter_loads_a_sweep():
    run = adapters.load(TAU1_RUN)
    assert run.family == "tau1"
    assert run.tasks and run.attempts
    assert all(a.trial >= 0 for a in run.attempts)
    blob = payload.build(run)
    assert blob["summary"]["num_tasks"] == len(run.tasks)
