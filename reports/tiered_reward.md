# Adding human preference: a tiered reward

τ-bench's reward is **binary**. A 20-message success and a 72-message success on
the same task both score 1.0. That discards a lot of information, and recovering
it is what would turn this project from RLVR (RL from *verifiable* rewards) into
RLHF proper.

## The discarded signal is real, and we measured it

Across 74 successful traces on the pinned stack:

| observation | value |
|---|---|
| tasks with ≥2 successful traces | 22 → **50 comparable pairs** |
| pairs whose two successes differ in length | **21 of 22** |
| median length gap | 8 messages (**max 52**) |
| most extreme pair | **20 vs 72 messages, both reward 1.0** |
| traces making an **exact** duplicate call (same tool, same arguments) | **50 of 74**, mean 4.2 per trace |
| message count | mean 47, **sd 24** |

Exact duplicate calls are unambiguous waste: the agent re-fetching information it
already has, in a trace the verifier still scores perfect.

## Design: three lexicographic tiers

**Tier 0 — hard gate.** Binary task success from the τ-bench verifier. Never
overridden; quality only ever breaks ties *among* successes.

This ordering is not a stylistic choice. Reward brevity without gating on
success and the optimal policy is to say nothing and terminate immediately. The
gate is what makes the rest safe.

**Tier 1 — programmatic, zero marginal cost.**

| signal | status |
|---|---|
| exact duplicate tool calls (same name + same args) | **strong** — 50/74 traces, mean 4.2 |
| message / tool-call count | **strong** — sd 24 on a mean of 47 |
| authenticated before first write | **useless here** — 73/74 already comply, no discrimination |

Recording the third is the point of measuring: we would otherwise have built a
policy-compliance term into the reward that carries no information for this
model.

**Tier 2 — human preference, expensive.** The subjective residue: politeness,
question quality, clarity of explanation.

## What the measurement changed about the plan

We expected humans to be needed mostly for tie-breaking. They are not:

| of the 21 task-pairs that differ in length | |
|---|---|
| Tier-1 metrics **agree** (shorter trace is also the cleaner one) | **20** — auto-rankable, no human |
| Tier-1 metrics **conflict** | **1** — genuine judgement call |

**Tier 1 resolves ~95% of comparisons for free.** So the human budget should go
to *dimensions Tier 1 cannot see at all* — was the agent polite, did it ask the
right clarifying question — rather than to arbitrating efficiency disputes.
That is a cheaper and more useful labelling task than we had assumed.

## The UI

Two traces for the same task side by side, with Tier-1 metrics pre-computed and
displayed, and the human asked only for Tier-2 judgements plus the rare Tier-1
conflict. Pairs are sampled by **active learning** — prioritise conflicts, and
pairs where a preliminary preference model is uncertain — rather than at random.

## Where the signal goes

- **Tier 1** folds into the GRPO reward directly as a small shaped bonus among
  successes: `r = 1.0 − λ·(duplicate_calls) − μ·(excess_turns)`, with λ and μ
  small enough that no combination can outrank a task failure.
- **Tier 2** trains either a small preference model or feeds DPO directly. 50
  pairs is thin for a reward model and workable for DPO.
- **The held-out evaluation stays the untouched binary τ-bench reward.** Shaped
  training reward must never become the metric, or we grade ourselves on our own
  curve.

## Caveats

1. **Shorter is not automatically better.** A brief trace may have skipped a
   verification step the policy document requires. Tier 1 must score brevity
   against the policy, not in a vacuum, or we would train away compliance.
2. **Politeness may trade off against success.** Measure the correlation on
   existing traces before rewarding it.
3. **τ-bench already encodes part of Tier 2.** The required-output substrings
   capture "did the agent actually state the refund amount", so some communication
   quality is in Tier 0 already.
4. **Multi-objective RL is harder to reason about.** The lexicographic gate keeps
   it tractable, but any positive result needs checking against reward hacking —
   inspect trajectories of newly-successful tasks by hand.

## Plan slot

- **Stage B2 (free, before training):** implement Tier 1, report the quality
  distribution across the existing 74 successes. Pure analysis of traces we have.
- **Stage F (after the first RL run):** trace-comparison UI, collect Tier-2
  labels on pairs the new policy generates, and run a DPO or preference-model
  variant against the Tier-0-only baseline.
