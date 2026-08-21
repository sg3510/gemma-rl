"""Regression tests for taurl.loops.

The numeric pins come from the pre-repair snapshot of pinned_telecom and are the
figures quoted in reports/tool_call_looping.md -- if a refactor moves any of
them, the report is wrong or the code is.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from taurl.loops import (
    BREAKER_CONSECUTIVE,
    TraceLoopStats,
    analyse,
    analyse_sim,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = REPO_ROOT / "results/_snapshots/pinned_telecom_prerepair.json"


def _sim(calls, reward=0.0, reason="max_steps", task_id="[mms_issue]x[PERSONA:None]"):
    return {
        "task_id": task_id,
        "trial": 0,
        "termination_reason": reason,
        "reward_info": {"reward": reward},
        "messages": [
            {"role": "assistant", "tool_calls": [c]} for c in calls
        ],
    }


def call(name, requestor="assistant", **args):
    return {"name": name, "arguments": args, "requestor": requestor}


def test_loop_detection_threshold():
    looped = analyse_sim(_sim([call("get_details_by_id", id="L1")] * 10), None)
    assert looped.looped and looped.looped_tool == "get_details_by_id"
    not_looped = analyse_sim(_sim([call("get_details_by_id", id="L1")] * 9), None)
    assert not not_looped.looped


def test_different_arguments_are_different_signatures():
    calls = [call("get_details_by_id", id=f"L{i}") for i in range(12)]
    stats = analyse_sim(_sim(calls), None)
    assert not stats.looped
    assert stats.unique_calls == 12 and stats.max_consecutive == 1


def test_prior_write_only_counts_agent_side_mutations():
    # A user-side device toggle before the loop is NOT an agent write; the
    # agent polling after it is still passive polling of the carrier DB.
    calls = [call("toggle_data", requestor="user")] + [call("get_details_by_id", id="L1")] * 10
    assert analyse_sim(_sim(calls), None).prior_write is False
    calls = [call("enable_roaming", customer_id="C1")] + [call("get_details_by_id", id="L1")] * 10
    assert analyse_sim(_sim(calls), None).prior_write is True


def test_needs_user_action_comes_from_expected_requestors():
    task = {"evaluation_criteria": {"actions": [
        {"name": "toggle_airplane_mode", "requestor": "user"},
        {"name": "enable_roaming", "requestor": "assistant"},
    ]}}
    stats = analyse_sim(_sim([call("get_details_by_id", id="L1")] * 10), task)
    assert stats.needs_user_action is True
    assert stats.missing_agent_writes == ["enable_roaming"]


@pytest.mark.skipif(not SNAPSHOT.exists(), reason="pre-repair snapshot not on disk")
def test_pinned_telecom_prerepair_pins():
    s = summarize(analyse(SNAPSHOT))
    assert s["scored"] == 119
    assert s["loops"] == 49 and s["passes"] == 56 and s["clean_failures"] == 14
    assert s["looped_tools"]["get_details_by_id"] == 44
    assert s["prior_write"] == 4
    assert s["needs_user_action"] == 38
    # Breaker calibration: >=5 consecutive aborts no passing trace.
    # On the smaller pre-repair pool passes topped out at 4 consecutive; the
    # threshold is calibrated against the larger repaired pool (max 6), so it
    # is comfortably clear of this snapshot.
    assert s["max_consecutive_pass"] == 4
    assert s["breaker_would_abort_passes"] == 0
    assert s["breaker_would_catch"] == 48
    assert BREAKER_CONSECUTIVE == 7
    assert s["by_scenario"]["mms_issue"]["looped"] == 27
