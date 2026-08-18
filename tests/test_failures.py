"""Every taxonomy branch must be reachable and correctly prioritised.

The precedence order matters: a rollout that both hit the step cap AND made a
wrong mutation is a *mechanical* failure, because the wrong mutation may simply
be where it happened to run out of room.
"""
import pytest

from taurl.failures import classify

CASES = {
    "infra_error": {"reward": 0.0, "info": {"error": "RateLimitError: 429"}, "traj": []},
    "step_cap": {
        "reward": 0.0,
        "info": {"task": {"actions": []}},
        "traj": [{"role": "assistant", "content": "x"}] * 30,
    },
    "unknown_tool": {
        "reward": 0.0,
        "info": {"task": {"actions": []}},
        "traj": [
            {"role": "tool", "content": "Unknown action frobnicate"},
            {"role": "user", "content": "###STOP###"},
        ],
    },
    "tool_arg_error": {
        "reward": 0.0,
        "info": {"task": {"actions": []}},
        "traj": [
            {"role": "tool", "content": "Error: order not found"},
            {"role": "user", "content": "###STOP###"},
        ],
    },
    "no_mutation": {
        "reward": 0.0,
        "info": {"task": {"actions": [{"name": "cancel_pending_order"}]}},
        "traj": [{"role": "user", "content": "###STOP###"}],
    },
    "missing_output": {
        "reward": 0.0,
        "info": {
            "task": {"actions": []},
            "reward_info": {"info": {"outputs": {"$150.00": False}}},
        },
        "traj": [{"role": "user", "content": "###STOP###"}],
    },
    "wrong_mutation": {
        "reward": 0.0,
        "info": {"task": {"actions": [{"name": "cancel_pending_order"}]}},
        "traj": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "cancel_pending_order"}}]},
            {"role": "user", "content": "###STOP###"},
        ],
    },
}


@pytest.mark.parametrize("expected,entry", list(CASES.items()))
def test_each_category_is_reachable(expected, entry):
    got, evidence = classify(entry)
    assert got == expected, f"expected {expected}, got {got} ({evidence})"


def test_infra_error_outranks_everything():
    """A crash must never be attributed to the model's behaviour."""
    entry = dict(CASES["wrong_mutation"], info={"error": "boom", "task": {"actions": []}})
    assert classify(entry)[0] == "infra_error"


def test_completed_conversation_is_not_step_cap():
    """30 steps is only a cap failure if the conversation never ended."""
    entry = {
        "reward": 0.0,
        "info": {"task": {"actions": [{"name": "cancel_pending_order"}]}},
        "traj": [{"role": "assistant", "content": "x"}] * 30
        + [{"role": "user", "content": "###STOP###"}],
    }
    assert classify(entry)[0] != "step_cap"
