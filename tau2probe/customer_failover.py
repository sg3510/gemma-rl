"""Failover the customer simulator between its two serving paths.

`gemini-3.7-flash` is reachable two ways, and OpenRouter's own endpoint listing
shows all six of its providers for this model are Google (3x Vertex, 3x AI
Studio, differing only in price tier). So the two paths are the same first-party
serving; one just has a proxy in front. Failing over between them changes the
route, not the model -- which is what makes this safe under D6.

Why it is needed: each path has a *different* limiter.

* OpenRouter enforces a GLOBAL per-model RPM cap (`model_limit_rpm`) shared
  across all its users. We hit it at concurrency 12 and lost 59 of 160 cells at
  concurrency 32.
* The direct Gemini API enforces our own Tier-2 quota, which is far larger but
  not infinite.

Because a rollout is a strictly serial conversation, ONE throttled customer call
stalls the whole episode -- which is why 12 requested concurrency delivered only
4.2x actual. Falling over to the other path instead of backing off keeps the
episode moving, and the two limiters are independent.

Only the customer model is redirected. The agent's calls are untouched: its
`extra_body` provider pinning is the whole basis of our serving-stack control.
"""
from __future__ import annotations

import os
import threading

import litellm
from litellm.exceptions import RateLimitError, ServiceUnavailableError, Timeout

DIRECT = "gemini/gemini-3.7-flash"
PROXIED = "openrouter/google/gemini-3.7-flash"
ALTERNATE = {DIRECT: PROXIED, PROXIED: DIRECT}

_stats = {"direct": 0, "proxied": 0, "failovers": 0, "both_failed": 0}
_lock = threading.Lock()


def stats() -> dict:
    with _lock:
        return dict(_stats)


def _bump(key: str) -> None:
    with _lock:
        _stats[key] += 1


def install() -> None:
    """Wrap litellm.completion so customer calls fail over between paths."""
    if getattr(litellm, "_customer_failover_installed", False):
        return
    original = litellm.completion

    def completion(*args, **kwargs):
        model = kwargs.get("model")
        if model not in ALTERNATE:
            return original(*args, **kwargs)  # agent call -- never touch it
        try:
            result = original(*args, **kwargs)
            _bump("direct" if model == DIRECT else "proxied")
            return result
        except (RateLimitError, ServiceUnavailableError, Timeout):
            alt = ALTERNATE[model]
            retry = dict(kwargs, model=alt)
            # OpenRouter-only routing controls are invalid against the direct
            # API, and the direct API needs its own key present.
            if alt == DIRECT:
                retry.pop("extra_body", None)
                if not os.environ.get("GEMINI_API_KEY"):
                    key = os.environ.get("AISTUDIO_GOOGLE_KEY")
                    if key:
                        os.environ["GEMINI_API_KEY"] = key
            _bump("failovers")
            try:
                result = original(*args, **retry)
                _bump("direct" if alt == DIRECT else "proxied")
                return result
            except Exception:
                # Both limiters are saying no. Let the caller's own retry
                # policy handle the backoff rather than spinning here.
                _bump("both_failed")
                raise

    litellm.completion = completion
    litellm._customer_failover_installed = True
