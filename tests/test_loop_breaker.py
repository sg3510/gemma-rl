"""The breaker must fire exactly where the data says it is safe."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from taurl.loops import BREAKER_CONSECUTIVE, analyse  # noqa: E402

SNAPSHOT = Path(__file__).resolve().parent.parent / "results/_snapshots/pinned_telecom_prerepair.json"


class _C:
    def __init__(self, name, args):
        self.name, self.arguments = name, args


class _M:
    def __init__(self, calls):
        self.tool_calls = calls


def _consecutive(seq):
    from tau2probe.loop_breaker import consecutive_repeats  # noqa: WPS433

    return consecutive_repeats([_M([_C(n, a)]) for n, a in seq])


@pytest.mark.skipif("tau2" not in sys.modules and not Path(".venv-tau2").exists(),
                    reason="tau2 env not available")
def test_counts_trailing_run_only():
    pytest.importorskip("tau2")
    sig, run = _consecutive([("a", {}), ("a", {}), ("b", {}), ("b", {}), ("b", {})])
    assert run == 3 and sig.startswith("b")


@pytest.mark.skipif(not SNAPSHOT.exists(), reason="snapshot not on disk")
def test_threshold_is_above_every_passing_trace():
    """The safety property: no passing trace may reach the threshold."""
    rows = analyse(SNAPSHOT)
    worst_pass = max(r.max_consecutive for r in rows if r.outcome == "pass")
    assert worst_pass < BREAKER_CONSECUTIVE, (
        f"threshold {BREAKER_CONSECUTIVE} would abort a passing trace that "
        f"reached {worst_pass} consecutive identical calls -- recalibrate"
    )
