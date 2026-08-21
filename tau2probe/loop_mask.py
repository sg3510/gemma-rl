"""Dynamic tool masking: hide a tool while the agent is looping on it.

The measured facts this is built against (reports/tool_call_looping.md):

* Repetition is self-reinforcing: 23% -> 87% -> 97% repeat probability as 2, 5
  and 8 identical results accumulate. Intervening EARLY is everything.
* Annotating the repetition backfires (a "you already called this" note raised
  the repeat rate to 40%) -- restating the call re-primes it. Removing the tool
  from the schema cannot re-prime: there is no token to copy.
* No passing trace exceeds 6 consecutive identical calls, so masking at 3 sits
  well inside the loop and clear of legitimate behaviour.

Design rules, from the same report:

* **Reversible.** The tool comes back as soon as the agent does anything else,
  so a task that legitimately needs a later re-read is not crippled.
* **Cascading.** If the loop migrates to another read, that one gets masked too
  (the static ablation exists to tell us whether this is needed).
* **Never in evaluation.** This is a training-time exploration aid -- action
  masking, as used for illegal moves in RL -- not a fix. Evaluation runs
  unmasked and reports loop rate, which is the only way to show the policy
  actually learned something rather than leaning on the masker.
* **Costed, not free.** `mask_events` is recorded on the agent so a training
  reward can subtract a small penalty per activation; without that the policy
  gets no gradient away from the attractor and learns the masker will save it.
"""
from __future__ import annotations

import json

from tau2.agent.llm_agent import LLMAgent

MASK_AFTER = 3  # consecutive identical calls before the tool disappears
RESTORE_AFTER = 2  # non-offending agent turns before it returns

# Every activation is recorded so a training reward can subtract a penalty per
# firing, and so the mechanism is auditable after the fact. Appends from worker
# threads are fine on a plain list.
MASK_LOG: list[dict] = []


def call_signature(name: str, arguments) -> str:
    try:
        return f"{name}::{json.dumps(arguments, sort_keys=True)}"
    except TypeError:
        return f"{name}::{arguments}"


class LoopMaskingAgent(LLMAgent):
    """LLMAgent that withholds tools the agent is currently looping on."""

    def __init__(self, *args, mask_after: int = MASK_AFTER,
                 restore_after: int = RESTORE_AFTER, cascade: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_after = mask_after
        self.restore_after = restore_after
        self.cascade = cascade
        self._all_tools = list(self.tools)
        self.mask_events: list[dict] = []

    def _recent_signatures(self, state) -> list[str]:
        sigs = []
        for msg in state.messages:
            for tc in getattr(msg, "tool_calls", None) or []:
                sigs.append((tc.name, call_signature(tc.name, tc.arguments)))
        return sigs

    def _masked_tool_names(self, state) -> set[str]:
        """Tools whose exact call has repeated `mask_after` times in a row, and
        that the agent has not since moved away from."""
        sigs = self._recent_signatures(state)
        if len(sigs) < self.mask_after:
            return set()
        masked: set[str] = set()
        # Trailing run of one identical signature -> mask that tool.
        run = 1
        for i in range(len(sigs) - 1, 0, -1):
            if sigs[i][1] == sigs[i - 1][1]:
                run += 1
            else:
                break
        if run >= self.mask_after:
            masked.add(sigs[-1][0])
        if not self.cascade:
            return masked
        # Cascade: any tool still dominating the recent window stays masked
        # until `restore_after` calls have gone by without it.
        window = sigs[-(self.mask_after + self.restore_after):]
        for name, sig in set(window):
            occurrences = [s for s in sigs if s[1] == sig]
            if len(occurrences) >= self.mask_after:
                tail = [s[1] for s in sigs[-self.restore_after:]]
                if sig in tail:
                    masked.add(name)
        return masked

    def _generate_next_message(self, message, state):
        masked = self._masked_tool_names(state)
        if masked:
            kept = [t for t in self._all_tools if t.name not in masked]
            event = {"task_id": getattr(self, "task_id", None),
                     "turn": len(state.messages), "masked": sorted(masked),
                     "tools_left": len(kept)}
            self.mask_events.append(event)
            MASK_LOG.append(event)
            self.tools = kept
        else:
            self.tools = list(self._all_tools)
        return super()._generate_next_message(message, state)


def make_factory(mask_after: int = MASK_AFTER, restore_after: int = RESTORE_AFTER,
                 cascade: bool = True):
    def factory(tools, domain_policy, **kwargs):
        task = kwargs.pop("task", None)
        for key in ("audio_native_config", "audio_taps_dir"):
            kwargs.pop(key, None)
        agent = LoopMaskingAgent(
            tools=tools, domain_policy=domain_policy,
            mask_after=mask_after, restore_after=restore_after,
            cascade=cascade, **kwargs,
        )
        agent.task_id = getattr(task, "id", None)
        return agent
    return factory
