"""Verify pass@k / pass^k against hand-computed closed forms.

If these are wrong, every number in the project is wrong, so they are checked
against values derived by hand rather than against the implementation.
"""
import json
import os
from pathlib import Path

import pytest

from taurl.metrics import (
    load_results,
    pass_at_k,
    pass_hat_k,
    summarize,
)


def test_the_headline_case_n8_c3():
    """3 successes out of 8 -- the canonical 'training frontier' task.

    This single task is the whole point of the project: pass@1 is 37.5% (in the
    30-40% sweet spot), while pass@8 is 100% -- the model demonstrably CAN solve
    it, just not reliably. tau-bench's pass^8 would report 0.0 and make it look
    hopeless.
    """
    n, c = 8, 3
    assert pass_at_k(n, c, 1) == pytest.approx(0.375)   # == c/n, the pass@1 estimate
    assert pass_at_k(n, c, 8) == pytest.approx(1.0)     # C(5,8)=0 -> 1 - 0
    assert pass_hat_k(n, c, 8) == pytest.approx(0.0)    # C(3,8)=0
    # 1 - C(5,2)/C(8,2) = 1 - 10/28
    assert pass_at_k(n, c, 2) == pytest.approx(1 - 10 / 28)
    # C(3,2)/C(8,2) = 3/28
    assert pass_hat_k(n, c, 2) == pytest.approx(3 / 28)


def test_pass_at_1_equals_success_fraction():
    """pass@1 must always equal c/n. This is the frontier selection criterion."""
    for n in range(1, 13):
        for c in range(n + 1):
            assert pass_at_k(n, c, 1) == pytest.approx(c / n)


def test_degenerate_groups_have_no_rl_signal():
    """The two useless cases for GRPO: all-fail and all-succeed."""
    assert pass_at_k(8, 0, 8) == 0.0 and pass_hat_k(8, 0, 8) == 0.0
    assert pass_at_k(8, 8, 8) == 1.0 and pass_hat_k(8, 8, 8) == 1.0


def test_pass_at_k_is_monotonic_and_dominates_pass_hat_k():
    for k in range(1, 9):
        assert pass_at_k(8, 3, k) >= pass_hat_k(8, 3, k)
    vals = [pass_at_k(8, 3, k) for k in range(1, 9)]
    assert vals == sorted(vals)                      # pass@k rises with k
    hats = [pass_hat_k(8, 3, k) for k in range(1, 9)]
    assert hats == sorted(hats, reverse=True)        # pass^k falls with k


def test_k_greater_than_n_is_an_error_not_a_silent_wrong_number():
    with pytest.raises(ValueError):
        pass_at_k(4, 2, 8)


def test_load_dedupes_and_uses_trial_dirs(tmp_path: Path):
    """Resumed sweeps write one dir per trial; upstream stamps every result
    trial=0, so directory identity is what makes dedup correct."""
    for trial in range(3):
        d = tmp_path / f"trial_{trial}" / "sub"
        d.mkdir(parents=True)
        (d / "res.json").write_text(
            json.dumps([{"task_id": 7, "reward": 1.0 if trial == 0 else 0.0, "trial": 0}])
        )
    # A stale duplicate for a (task, trial) slot must lose to the newer file.
    stale = tmp_path / "trial_0" / "stale.json"
    stale.write_text(json.dumps([{"task_id": 7, "reward": 0.0, "trial": 0}]))
    os.utime(stale, (0, 0))  # explicitly older than the real result

    tasks = load_results(tmp_path)
    assert tasks[7].n == 3, "duplicate within a trial must not inflate n"
    assert tasks[7].c == 1

    s = summarize(tasks)
    assert s["pass_at_k"][1] == pytest.approx(1 / 3)
    assert s["total_rollouts"] == 3
