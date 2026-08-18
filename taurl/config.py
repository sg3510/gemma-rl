"""Central configuration. Every knob that affects a measured number lives here.

Rule: if you change something in this file, log it in rl_product_managers.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"

load_dotenv(REPO_ROOT / ".env")

PROVIDER = "openrouter"

# The policy under evaluation (and, later, under training).
AGENT_MODEL_PAID = "google/gemma-4-26b-a4b-it"
AGENT_MODEL_FREE = "google/gemma-4-26b-a4b-it:free"

# The simulated customer. FIXED FOREVER across every run and every training step.
# Deliberately not a Gemma: the customer must never be the policy we are training,
# or we cannot tell whether a failure came from the agent or from the customer.
USER_MODEL = "google/gemini-3.7-flash"

# tau-bench interpolates the user-model string straight into its checkpoint
# filename (run.py:30) and, unlike the agent model, never strips the vendor
# prefix. A "/" in the name therefore becomes a directory separator, `open()`
# raises OUTSIDE the per-task try/except, and the entire trial's results are
# lost. We hand tau-bench a slash-free alias for pathname purposes; the wrapper
# in patches.py rewrites it back to USER_MODEL at the API boundary.
USER_MODEL_ALIAS = "gemini-3.7-flash"

# Sampling temperature for the agent. pass@k is meaningless at temp 0, and the
# baseline must be measured at the temperature RL rollouts will actually use.
AGENT_TEMPERATURE = 1.0

# The customer is pinned to greedy so the agent policy is the only thing that
# varies across the k rollouts of a task. Upstream tau-bench passes no
# temperature at all, inheriting the provider default (~1.0).
USER_TEMPERATURE = 0.0

# OpenRouter per-token prices (fetched from /api/v1/models, 2026-08-18).
# litellm 1.97.0 does not know these models, so we register them ourselves to
# get real cost accounting instead of a KeyError.
MODEL_PRICES = {
    AGENT_MODEL_PAID: {"input": 0.00000007, "output": 0.00000034},
    AGENT_MODEL_FREE: {"input": 0.0, "output": 0.0},
    USER_MODEL: {"input": 0.000000375, "output": 0.000001875},
}

# tau-bench's agent step cap. Retail trajectories that exhaust this are counted
# as "step-cap exhaustion" failures, which is a harness problem, not a
# capability problem -- see taurl/failures.py.
MAX_NUM_STEPS = 30


@dataclass(frozen=True)
class Tier:
    """Free vs paid OpenRouter serving for the agent."""

    name: str
    agent_model: str
    max_concurrency: int
    num_retries: int


FREE = Tier(name="free", agent_model=AGENT_MODEL_FREE, max_concurrency=2, num_retries=8)
PAID = Tier(name="paid", agent_model=AGENT_MODEL_PAID, max_concurrency=16, num_retries=5)

TIERS = {"free": FREE, "paid": PAID}


def get_tier(name: str | None = None) -> Tier:
    """Resolve the serving tier. Defaults to free; override with TAURL_TIER."""
    return TIERS[(name or os.environ.get("TAURL_TIER", "free")).lower()]


def require_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put it in .env at the repo root "
            "(already gitignored)."
        )
    return key
