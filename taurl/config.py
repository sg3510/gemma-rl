"""Central configuration. Every knob that affects a measured number lives here.

Rule: if you change something in this file, log it in rl_product_managers.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
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

# Serving path note (2026-08-20): OpenRouter serves this model from six
# endpoints, ALL Google (3x Vertex, 3x AI Studio, differing only in price tier),
# so calling the Gemini API directly is the same first-party serving minus a
# proxy hop -- 0.9s per call instead of 1.82s. It also escapes OpenRouter's
# GLOBAL per-model RPM cap on gemini-3.7-flash, which is shared across all their
# users and was the sole throughput bottleneck for our evals (the agent backend
# never rate-limited). See reports/eval_protocol.md.
USER_MODEL_DIRECT = "gemini/gemini-3.7-flash"

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

# OpenRouter routes a single model id across many backend providers, and they do
# NOT all serve tool calls correctly. Measured 2026-08-18 with a prompt that
# invites two parallel tool calls (3 samples each):
#
#   DeepInfra, NextBit, SiliconFlow, Novita, Venice, Parasail   6/6 valid JSON
#   Google                                                      0/6 -- MERGED
#   Cloudflare                                                  rate-limited, untested
#
# Google's endpoint concatenates parallel tool calls into ONE tool_call whose
# `arguments` string holds two JSON objects back to back:
#
#   {"order_id": "#W5765741"}{"order_id": "#W7736983"}
#
# tau-bench then does a bare json.loads on that and the whole rollout dies. This
# is a serving-side tool-call parsing bug -- the same class of failure seen
# self-hosting Gemma 4 on vLLM without `--enable-auto-tool-choice
# --tool-call-parser gemma4`. We cannot pass parser flags to someone else's
# server, so we exclude the provider instead.
#
# This mattered: it killed ~11% of all retail rollouts, every one scored as a
# task failure, understating pass@1.
BLOCKED_PROVIDERS = ["Google"]

# allow_fallbacks stays on so we keep throughput across the remaining backends.
PROVIDER_ROUTING = {"ignore": BLOCKED_PROVIDERS, "allow_fallbacks": True}

# THE pinned stack. Measured per-provider on 10 retail train tasks (2026-08-18):
#
#   DeepInfra   fp8   10/10 complete  70%  2.1 min
#   NextBit     bf16  10/10 complete  70%  2.0 min   <-- chosen
#   Novita      bf16  10/10 complete  70%  2.2 min
#   Venice      bf16  10/10 complete  80%  2.7 min
#   SiliconFlow fp8    0/10 -- does not serve `tools` (NotFoundError)
#   Parasail    bf16   0/10 -- does not serve `tools` (NotFoundError)
#
# The four working backends agree within sampling noise, which retires the
# earlier "fp8 beats bf16 by 31 points" scare: that gap was SiliconFlow and
# Parasail failing 100% of calls and biasing both group means.
#
# NextBit is chosen for bf16 (matching the precision we will fine-tune in),
# full 262K context, and the best throughput of the bf16 options.
#
# ^ This reasoning is now known to be insufficient: it never measured loop rate,
#   empty-completion rate, or whether the backend can serve the model's own
#   sampling config. All three go against NextBit. See D30.
#
# SUPERSEDED 2026-08-20 (D30): NextBit is the WRONG choice for tau2 telecom.
# On identical tasks and weights it locks into tool-call loops at 32% against
# Novita's 6% (bf16 too, p=0.003), passes 60% against Novita's 91%, and emits
# empty completions that killed 19 of 160 cells in one run against Novita's 5
# of 40. It also cannot serve `top_k`, so it is structurally incapable of
# Gemma's own generation_config (top_k=64, top_p=0.95).
#
# The comparison below that chose it was 10 tasks on tau1 RETAIL scored by pass
# rate; it could not see either pathology. See reports/provider_loop_rates.md.
# Plan of record: re-baseline on Novita with sampling parameters pinned
# explicitly. This constant stays as-is only until that re-baseline lands, so
# existing numbers remain reproducible.
#
# `require_parameters` makes OpenRouter refuse any backend that cannot serve our
# `tools` payload, rather than silently degrading. NOTE the interaction with the
# fix: adding `top_k` to the sampling args would make OpenRouter refuse NextBit
# outright, since NextBit does not expose it.
PINNED_PROVIDER = "NextBit"
PINNED_ROUTING = {
    "only": [PINNED_PROVIDER],
    "allow_fallbacks": False,
    "require_parameters": True,
}

# Task counts per environment and split, and which splits actually exist.
# NOTE: tau-bench's airline env ships a TEST SPLIT ONLY -- `airline/env.py`
# raises ValueError for "train" and "dev". So airline can serve as an
# evaluation target but cannot host an RL curriculum without us authoring a
# split ourselves. Retail is the only tau1 domain with a real training pool.
SPLIT_SIZES = {
    "retail": {"dev": 20, "test": 115, "train": 500},
    "airline": {"test": 50},
}

# Tools that mutate the database, per environment. Everything else is a read.
WRITE_TOOLS = {
    "retail": {
        "cancel_pending_order",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
        "transfer_to_human_agents",
    },
    "airline": {
        "book_reservation",
        "cancel_reservation",
        "send_certificate",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
        "transfer_to_human_agents",
    },
}


@dataclass(frozen=True)
class Tier:
    """A fully-pinned serving stack for the agent.

    The whole point of naming this is that a benchmark number is a property of
    the model AND the stack that served it (see D13/D14 in the decision log).
    OpenRouter fans one model id across 8 backends at mixed precision with
    different tool-call parsers, and we measured ~30-point swings between them.
    So every run records exactly which stack produced it.
    """

    name: str
    agent_model: str
    agent_provider: str
    max_concurrency: int
    num_retries: int
    # OpenRouter-only routing controls; empty for single-backend stacks.
    extra_body: dict = field(default_factory=dict)
    # Env var holding the key, and the env var litellm expects it in.
    key_env: str = "OPENROUTER_API_KEY"
    key_target: str = "OPENROUTER_API_KEY"


FREE = Tier(
    name="free",
    agent_model=AGENT_MODEL_FREE,
    agent_provider="openrouter",
    max_concurrency=2,
    num_retries=8,
    extra_body={"provider": PROVIDER_ROUTING},
)
PAID = Tier(
    name="paid",
    agent_model=AGENT_MODEL_PAID,
    agent_provider="openrouter",
    max_concurrency=16,
    num_retries=5,
    extra_body={"provider": PINNED_ROUTING},
)
# Google AI Studio: ONE backend, one serving config, one precision. Verified
# 2026-08-18 to emit correctly separated parallel tool calls (unlike
# OpenRouter's "Google" endpoint, which merges them -- see B6). Gemma is free
# here and showed no throttling at concurrency 12. This is the stack that makes
# a before/after RL comparison meaningful, because nothing about it varies
# between runs.
AISTUDIO = Tier(
    name="aistudio",
    agent_model="gemma-4-26b-a4b-it",
    agent_provider="gemini",
    max_concurrency=12,
    num_retries=5,
    key_env="AISTUDIO_GOOGLE_KEY",
    key_target="GEMINI_API_KEY",
)

TIERS = {"free": FREE, "paid": PAID, "aistudio": AISTUDIO}


def get_tier(name: str | None = None) -> Tier:
    """Resolve the serving stack. Override with TAURL_TIER."""
    return TIERS[(name or os.environ.get("TAURL_TIER", "aistudio")).lower()]


def require_api_key(tier: "Tier | None" = None) -> str:
    """Ensure the agent's key is present and exported where litellm expects it.

    The customer simulator always runs on OpenRouter (D6 pinned it to
    gemini-3.7-flash, which AI Studio does not serve), so that key is required
    regardless of which stack the agent uses.
    """
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set (needed for the customer simulator). "
            "Put it in .env at the repo root (already gitignored)."
        )
    if tier is not None and tier.key_env != "OPENROUTER_API_KEY":
        key = os.environ.get(tier.key_env)
        if not key:
            raise RuntimeError(f"{tier.key_env} is not set (needed for {tier.name}).")
        os.environ.setdefault(tier.key_target, key)
    return os.environ["OPENROUTER_API_KEY"]
