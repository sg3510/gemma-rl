"""Multi-route serving pools with failover, for both agent and customer.

The point is throughput WITHOUT changing what serves the model. Each pool holds
routes that are the same backend reached through different proxies, so failing
over between them is a route change, not a stack change -- which is the only
form of failover D30 permits.

**Agent pool** -- both routes pin Novita:
  * OpenRouter -> `provider.only = ["Novita"]`
  * Vercel AI Gateway -> `providerOptions.gateway.only = ["novita"]`
  Pinning is enforced by both (verified: pinning to a provider that cannot serve
  the model returns "No available providers match the 'only' filter"). The
  gateway's providers for this model are gmicloud, novita, parasail, vertex.

**Customer pool** -- every route is Google:
  * Gemini API direct (Tier-2 quota, ~0.9s/call)
  * OpenRouter (all six of its endpoints for this model are Google)
  * Vercel AI Gateway

Why it matters: a rollout is a strictly serial conversation, so one throttled
call stalls the whole episode. That is why concurrency 12 delivered only 4.2x
actual. Each proxy has an INDEPENDENT limiter -- OpenRouter enforces a global
per-model RPM cap shared across all its users, while the direct API enforces our
own quota -- so spreading across them multiplies usable concurrency.
"""
from __future__ import annotations

import os
import threading

import litellm
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

VERCEL_BASE = "https://ai-gateway.vercel.sh/v1"
AGENT_MODEL = "google/gemma-4-26b-a4b-it"
CUSTOMER_MODEL = "google/gemini-3.7-flash"

TRANSIENT = (RateLimitError, ServiceUnavailableError, Timeout, APIConnectionError)


def _vercel_key() -> str | None:
    return os.environ.get("VERCEL_API_GATEWAY") or os.environ.get("AI_GATEWAY_API_KEY")


def agent_routes(provider: str = "Novita") -> list[dict]:
    routes = [{
        "model": f"openrouter/{AGENT_MODEL}",
        "extra_body": {"provider": {"only": [provider], "allow_fallbacks": False,
                                    "require_parameters": True}},
    }]
    if _vercel_key():
        routes.append({
            "model": f"openai/{AGENT_MODEL}",
            "api_base": VERCEL_BASE,
            "api_key": _vercel_key(),
            "extra_body": {"providerOptions": {"gateway": {"only": [provider.lower()]}}},
        })
    return routes


def customer_routes() -> list[dict]:
    routes = []
    if os.environ.get("AISTUDIO_GOOGLE_KEY") or os.environ.get("GEMINI_API_KEY"):
        os.environ.setdefault("GEMINI_API_KEY", os.environ.get("AISTUDIO_GOOGLE_KEY", ""))
        routes.append({"model": f"gemini/{CUSTOMER_MODEL.split('/')[-1]}"})
    routes.append({"model": f"openrouter/{CUSTOMER_MODEL}"})
    if _vercel_key():
        routes.append({"model": f"openai/{CUSTOMER_MODEL}", "api_base": VERCEL_BASE,
                       "api_key": _vercel_key()})
    return routes


_pools: dict[str, list[dict]] = {}
_next = {"agent": 0, "customer": 0}
_stats: dict[str, int] = {}
_lock = threading.Lock()


def stats() -> dict:
    with _lock:
        return dict(_stats)


def _bump(key: str) -> None:
    with _lock:
        _stats[key] = _stats.get(key, 0) + 1


def _pool_for(model: str | None) -> str | None:
    if not model:
        return None
    if AGENT_MODEL in model:
        return "agent"
    if CUSTOMER_MODEL.split("/")[-1] in model:
        return "customer"
    return None


def install(provider: str = "Novita", spread: bool = True) -> None:
    """Patch litellm.completion to spread and fail over across each pool.

    `spread` round-robins the starting route so load is shared across limiters
    rather than hammering one until it 429s. Safe here only because every route
    within a pool is the same backend.
    """
    _pools["agent"] = agent_routes(provider)
    _pools["customer"] = customer_routes()
    if getattr(litellm, "_routes_installed", False):
        return
    original = litellm.completion

    def completion(*args, **kwargs):
        pool_name = _pool_for(kwargs.get("model"))
        if not pool_name:
            return original(*args, **kwargs)
        pool = _pools[pool_name]
        with _lock:
            start = _next[pool_name] % len(pool) if spread else 0
            _next[pool_name] += 1
        last = None
        for i in range(len(pool)):
            route = pool[(start + i) % len(pool)]
            attempt = {k: v for k, v in kwargs.items()
                       if k not in ("model", "api_base", "api_key", "extra_body")}
            attempt.update(route)
            try:
                out = original(*args, **attempt)
                _bump(f"{pool_name}:{route['model'].split('/')[0]}")
                return out
            except TRANSIENT as exc:
                last = exc
                _bump(f"{pool_name}:failover")
                continue
        _bump(f"{pool_name}:exhausted")
        raise last

    litellm.completion = completion
    litellm._routes_installed = True
