"""Terminate a rollout once it is provably lost to a tool-call loop.

Measured justification (reports/tool_call_looping.md):

* No passing trace ever reaches 7 consecutive identical tool calls -- the
  maximum observed across 62 passing rollouts is 6. So cutting at 7 is
  **outcome-neutral**: every trajectory it ends was going to score 0 at the
  step cap anyway.
* Replaying the rule over real runs saves **55%** of all tokens on the NextBit
  grid (61 of 141 rollouts cut) and **12%** on Novita (16 of 156). Agent input
  grows with context length because every turn resends the conversation, so a
  loop is quadratically expensive.
* Repeat probability at depth 5 is 87% and at depth 8 is 97% -- by seven
  repetitions the trajectory is effectively absorbed.

This is a BREAKER, not a repair guard. The rollout still scores 0, exactly as
its `max_steps` fate would have been. It removes wasted tokens, never the
consequence of looping -- which is what keeps the GRPO gradient that teaches the
policy to stop. For training there is a positive argument too: 7 repetitions of
negatively-advantaged tokens already carry the signal, while 50 of them swamp
the gradient mass that belongs to the earlier turns where the agent should have
acted instead of re-reading.

Recalibrate BREAKER_CONSECUTIVE whenever the trace pool grows: it moved from 5
to 7 when the repaired grid added 22 rollouts.
"""
from __future__ import annotations

import json

from tau2.agent.llm_agent import LLMAgent

from taurl.loops import BREAKER_CONSECUTIVE


class LoopBrokenError(RuntimeError):
    """Raised to end a rollout that has entered a repetition attractor."""

    def __init__(self, signature: str, count: int):
        self.signature = signature
        self.count = count
        super().__init__(
            f"loop breaker: {signature} repeated {count} times consecutively "
            f"(threshold {BREAKER_CONSECUTIVE}); no passing trace has ever "
            f"exceeded {BREAKER_CONSECUTIVE - 1}. Scoring 0."
        )


def signature(name: str, arguments) -> str:
    try:
        return f"{name}::{json.dumps(arguments, sort_keys=True)}"
    except TypeError:
        return f"{name}::{arguments}"


def consecutive_repeats(messages) -> tuple[str | None, int]:
    """Length of the trailing run of one identical tool-call signature."""
    sigs = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            sigs.append(signature(tc.name, tc.arguments))
    if not sigs:
        return None, 0
    run = 1
    for i in range(len(sigs) - 1, 0, -1):
        if sigs[i] == sigs[i - 1]:
            run += 1
        else:
            break
    return sigs[-1], run


class LoopBreakingAgent(LLMAgent):
    """LLMAgent that stops once the trailing identical-call run hits the threshold."""

    def __init__(self, *args, threshold: int = BREAKER_CONSECUTIVE, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.broken: dict | None = None

    def _generate_next_message(self, message, state):
        sig, run = consecutive_repeats(state.messages)
        if sig and run >= self.threshold:
            self.broken = {"signature": sig, "count": run,
                           "turn": len(state.messages)}
            raise LoopBrokenError(sig, run)
        return super()._generate_next_message(message, state)


def make_factory(threshold: int = BREAKER_CONSECUTIVE):
    def factory(tools, domain_policy, **kwargs):
        for key in ("task", "audio_native_config", "audio_taps_dir"):
            kwargs.pop(key, None)
        return LoopBreakingAgent(tools=tools, domain_policy=domain_policy,
                                 threshold=threshold, **kwargs)
    return factory
