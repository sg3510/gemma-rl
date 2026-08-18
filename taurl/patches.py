"""Runtime patches over tau-bench. We do NOT fork it.

`vendor/tau-bench` is a submodule pinned to a SHA and must stay byte-identical to
upstream so `git diff` against Sierra's repo is always empty. Everything we need
to change is monkey-patched here at import time.

Both `tau_bench.agents.tool_calling_agent` and `tau_bench.envs.user` do
`from litellm import completion`, which binds the name into the module at import.
So we rebind the module attribute, not `litellm.completion`.

What we change and why:
  1. Retries with backoff on 429/5xx. The OpenRouter free tier rate-limits hard
     and upstream passes no retry config at all, so a sweep dies partway.
  2. A temperature for the customer simulator. Upstream passes none, inheriting
     the provider default (~1.0), which means the customer varies across the k
     rollouts of a task and pollutes the pass@k signal we are trying to measure.
  3. A guaranteed `response_cost` in `_hidden_params`. litellm does not know
     these OpenRouter models, and `envs/user.py` does a bare subscript that
     would raise KeyError.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Any

import litellm

from taurl import config

# These registrations are intentional; litellm warns only about cache-cost
# fields we do not use.
litellm.suppress_debug_info = True

_TRANSIENT = (
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.APIConnectionError,
    litellm.Timeout,
)

_applied = False
_lock = threading.Lock()

# Accumulated spend, since tau-bench's own cost tracking is unreliable here.
_spend = {"agent": 0.0, "user": 0.0}
_spend_lock = threading.Lock()


def register_model_prices() -> None:
    """Teach litellm the OpenRouter prices so response_cost is real, not None."""
    for model, price in config.MODEL_PRICES.items():
        litellm.register_model(
            {
                f"openrouter/{model}": {
                    "input_cost_per_token": price["input"],
                    "output_cost_per_token": price["output"],
                    "litellm_provider": "openrouter",
                    "mode": "chat",
                }
            }
        )


def _ensure_cost(res: Any, role: str) -> Any:
    """Guarantee `_hidden_params['response_cost']` is a float, and tally spend."""
    hidden = getattr(res, "_hidden_params", None)
    if hidden is None:
        hidden = {}
        try:
            res._hidden_params = hidden
        except Exception:
            return res
    cost = hidden.get("response_cost")
    if cost is None:
        try:
            cost = litellm.completion_cost(completion_response=res)
        except Exception:
            cost = 0.0
    cost = float(cost or 0.0)
    hidden["response_cost"] = cost
    with _spend_lock:
        _spend[role] += cost
    return res


def _make_completion(
    role: str,
    num_retries: int,
    force_temperature: float | None,
    force_model: str | None = None,
):
    """Build a `completion` replacement for one call site (agent or customer)."""

    def _completion(**kwargs):
        if force_model is not None:
            # The customer is addressed by a slash-free alias so tau-bench can
            # build a valid checkpoint path; restore the real id here.
            kwargs["model"] = force_model
        if force_temperature is not None:
            kwargs.setdefault("temperature", force_temperature)
        kwargs.setdefault("timeout", 180)

        last_exc = None
        for attempt in range(num_retries + 1):
            try:
                res = litellm.completion(**kwargs)
                return _ensure_cost(res, role)
            except _TRANSIENT as exc:
                last_exc = exc
                if attempt == num_retries:
                    break
                # Exponential backoff with jitter. The free tier is per-minute,
                # so the tail waits need to be long enough to clear a window.
                delay = min(60.0, 2.0 * (2**attempt)) * (0.5 + random.random())
                time.sleep(delay)
        raise last_exc

    return _completion


def apply(tier: config.Tier) -> None:
    """Install all patches. Idempotent; safe to call from multiple entrypoints."""
    global _applied
    with _lock:
        if _applied:
            return

        import logging

        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        register_model_prices()

        import tau_bench.agents.tool_calling_agent as agent_mod
        import tau_bench.envs.user as user_mod

        # Upstream hardcodes `max_num_steps: int = 30` as a default argument, and
        # `run.py` never passes it. Trajectories that hit the cap are a harness
        # limit, not a capability limit, so we need to be able to raise it.
        _orig_solve = agent_mod.ToolCallingAgent.solve

        def _solve(self, env, task_index=None, max_num_steps=None):
            return _orig_solve(
                self,
                env=env,
                task_index=task_index,
                max_num_steps=max_num_steps or config.MAX_NUM_STEPS,
            )

        agent_mod.ToolCallingAgent.solve = _solve

        # The agent's temperature is passed explicitly by tau-bench, so we do not
        # force one here -- that would silently override the run config.
        agent_mod.completion = _make_completion(
            role="agent", num_retries=tier.num_retries, force_temperature=None
        )
        # The customer gets pinned to greedy. See module docstring, reason 2.
        user_mod.completion = _make_completion(
            role="user",
            num_retries=tier.num_retries,
            force_temperature=config.USER_TEMPERATURE,
            force_model=config.USER_MODEL,
        )
        _applied = True


def spend() -> dict[str, float]:
    with _spend_lock:
        return dict(_spend, total=_spend["agent"] + _spend["user"])
