# Our evaluation protocol, and how it differs from Sierra's

**We are running toy experiments, not preparing a leaderboard submission.** This
document states every way we deviate from τ-bench's standard protocol, why, and
why the deviations are acceptable *for the question we are actually asking*.

The question matters, because it is a different one from Sierra's.

| | Sierra's τ-bench | This project |
|---|---|---|
| Question | How do **different models** rank? | Does **one model** improve after RL? |
| Comparison | across models, unpaired | within model, **paired** on the same tasks |
| Effect size | 20–40 points (frontier vs open-weights) | 5–15 points |
| Output | a leaderboard position | a before/after delta |

A benchmark tuned for the first question is not automatically adequate for the
second — and, as it turns out, the second is easier to power than we first
thought, for a reason that has nothing to do with task count.

---

## Every deviation, and why

| Sierra standard | What we do | Why |
|---|---|---|
| `base` task split (114 telecom) | **`test` split (40)** | `base` *contains* the 74 training tasks. Train on them and evaluate on `base` and you are evaluating on your training set. |
| 4+ trials per domain | 4 for baselines, **8** for before/after | Power. See below. |
| Default user simulator | **`gemini-3.7-flash` @ temp 0** | Cheaper than the model under test, and deliberately not a Gemma so a failure is attributable. Upstream passes *no* temperature to the customer, inheriting ~1.0, which would mix customer variance into our pass@k. |
| `--max-steps 200` | **140** | Across 199 observed simulations the longest *successful* trajectory was 136 messages. 140 costs nothing measurable and cuts 24% of spend. |
| Any available backend | **pinned NextBit, bf16** (superseded → Novita, see [provider_loop_rates.md](provider_loop_rates.md)) | We measured ~30-point swings between OpenRouter backends for the same model id, and one backend corrupts tool calls outright. |
| Greedy decoding (implied) | **temperature 1.0** | pass@k is meaningless at temperature 0, and the baseline must be measured at the temperature RL rollouts are generated at. |
| Model not trained on τ-bench | **We will train on the τ-bench train split** | That is the entire experiment. |
| Rollout ends only at `max_steps` | **Loop breaker: terminate at 7 consecutive identical tool calls, score 0** | Outcome-neutral by measurement, and saves 12–55% of tokens. See below. |

Any one of these makes a submission non-standard. The last one alone makes it
**custom** by Sierra's own definition — their guide requires "a model or system
that was not trained specifically on τ-bench tasks, rewards, or evaluation
data". They anticipated this use case and gave it its own category.

We are not submitting. These numbers are "Gemma 4 26B A4B on τ-retail/telecom
**under this harness**", never a leaderboard position.

---

## The loop breaker, and why it is not a thaw of the reward

**Rule:** if the agent issues the same tool call — same name, same arguments —
**7 times consecutively**, the rollout terminates immediately with
`termination_reason = loop_breaker` and **reward 0**.

**Why this is not a protocol cheat.** Across 62 passing rollouts on the repaired
grid, the largest run of consecutive identical calls in any *successful* trace is
**6**. Every trajectory the breaker ends was already going to reach the step cap
and score 0; it changes when we stop paying, not what we score. Repeat
probability measured by prefix replay is 87% at five identical results and 97%
at eight — by seven the trajectory is effectively absorbed.

**What it saves.** Replaying the rule over completed runs: **55%** of all tokens
on the NextBit grid (61 of 141 rollouts cut) and **12%** on Novita (16 of 156).
Agent input grows with context length because every turn resends the whole
conversation, so a loop is quadratically expensive.

**Why it also helps training, not just cost.** Seven repetitions of
negatively-advantaged tokens already carry the "do not repeat" signal; fifty of
them put most of the trajectory's gradient mass on repeated tokens and drown the
earlier turns where the agent should have acted instead of re-reading. Long
looping trajectories are also the ones that blow up KV cache and cap group size,
so cutting them buys a larger G at the same memory.

**The line it must not cross.** This is a *breaker*, never a *repair guard*. It
never returns a cached result, never annotates the repetition, never rescues the
episode — all of which would delete the reward signal that teaches the policy to
stop looping and produce a policy that only works inside our scaffold. A dedup
note was measured and made things **worse** (40% repeat rate against a 23%
control): restating the call re-primes it.

**Conditions of use.**

1. **Both arms, always.** Before and after must run with the breaker enabled and
   at the same threshold, or part of the delta is a change of harness.
2. **Recalibrate as the corpus grows.** The threshold moved from 5 to 7 when 22
   repaired rollouts landed and the worst passing trace went from 4 to 6
   consecutive. `tests/test_loop_breaker.py` asserts the safety property — no
   passing trace may reach the threshold — and fails loudly if it ever does.
3. **Report `loop_breaker` separately** from `max_steps` and from genuine task
   failures, so the loop rate stays visible rather than being folded into a
   single failure bucket.
4. **Evaluation runs unmasked.** The breaker is compatible with dynamic tool
   masking during *training* rollouts, but masking never enters evaluation; see
   [tool_call_looping.md](tool_call_looping.md).

Implementation: `tau2probe/loop_breaker.py`, threshold from
`taurl.loops.BREAKER_CONSECUTIVE`.

---

## Customer simulator: served direct, not through OpenRouter

**Change (2026-08-20):** the customer simulator now runs against the Gemini API
directly (`gemini/gemini-3.7-flash`) instead of `openrouter/google/gemini-3.7-flash`.
The model, version and temperature (0.0) are unchanged, so D6 still holds.

**Why it is safe.** OpenRouter serves this model from **six endpoints, all
Google** -- three "Google" (Vertex) and three "Google AI Studio", differing only
in price tier. Routing direct is therefore not a new serving stack; it is one of
the two paths OpenRouter was already using, minus a proxy hop. That also
explains the latency: **0.9s per call direct against 1.82s via OpenRouter** is
proxy overhead, not a different model.

**Why it was necessary.** OpenRouter applies a *global* per-model RPM cap to
`gemini-3.7-flash` under high demand -- shared across all their users, not a
property of our key:

```
Rate limit exceeded: model_limit_rpm/google/gemini-3.7-flash-20260813.
High demand for google/gemini
```

Each rollout makes ~20 sequential customer calls, so this capped useful
concurrency at ~12. At 32 we lost 59 of 160 cells to 429s in a single run. The
**agent** backend (Novita) never rate-limited at any concurrency we tried, so the
pinned customer was the sole throughput bottleneck -- an awkward position given
[rl_run_plan.md](rl_run_plan.md) needs 3,840+ rollouts and already counts
customer latency as ~45% of wall-clock.

| | via OpenRouter | direct |
|---|---|---|
| per-call latency | 1.82s | **0.9s** |
| median rollout | 121s | **108s** |
| usable concurrency | ~12 | higher (ceiling not yet probed) |
| rate limits at c=32 | 1,306 | n/a |

**Caveat kept on the record:** runs before this change used the OpenRouter path.
Any paired before/after must not straddle the switch -- both arms on the same
customer path, like every other pinned knob.

---

## Why 40 tasks is still enough for our question

The instinct is that a 40-task evaluation is too small. For *ranking models* it
would be marginal. For *detecting a within-model change* it is fine, because a
paired design cancels the dominant source of variance.

Task difficulty is enormously heterogeneous — telecom tasks range from 0% to
100% success. In a cross-model comparison that heterogeneity is noise you must
overwhelm with sample size. In a before/after comparison on the *same tasks*, it
subtracts out: a task that is hard before is hard after, and only the per-task
*delta* enters the test.

Simulated power to detect a **+10 point** improvement:

| design | sims per arm | unpaired | **paired** |
|---|---|---|---|
| our `test` split, 40 × 4 | 160 | 36% | 63% |
| **our `test` split, 40 × 8** | **320** | 74% | **89%** |
| Sierra's `base`, 114 × 4 | 456 | 91% | 97% |
| Sierra's `base`, 114 × 8 | 912 | 100% | 100% |

**40 tasks × 8 trials, analysed paired, gives 89% power** — above the
conventional 80% bar — at roughly **$49** for both arms at $0.076/simulation.

For contrast, Sierra's actual use case — separating a 55% frontier model from a
25% open-weights one — reaches **100% power** at their standard 114 × 4. Their
benchmark is not underpowered; we were simply asking a harder question with a
smaller slice and analysing it the conservative way.

### The analysis this commits us to

Power of 89% depends on doing the paired test properly:

1. Same task ids, same trial count, before and after.
2. Same fixed customer simulator, same temperature, same pinned backend, same
   step cap, **same sampling parameters** (`top_p`, `top_k` — pinned explicitly
   since D30; they were previously left to undocumented per-host defaults) and
   **the same loop-breaker threshold**. Any change to the serving stack between
   arms contaminates the delta — the whole lesson of our provider investigation.
3. Test on **per-task deltas** (`c_after/n − c_before/n`), not on pooled totals.
4. Report the confidence interval on the delta, not just two point estimates.

### What we still cannot claim

- **An absolute number comparable to the leaderboard.** Different split,
  different customer, different step cap, different temperature.
- **A +5 point improvement.** Power at that effect size is 39% even at 40 × 16.
  If the true gain is small, this evaluation cannot prove it, and we will say so
  rather than publish a suggestive-but-underpowered result.
- **Generalisation from telecom alone.** Telecom has only three scenario
  archetypes, so we additionally evaluate on τ2 airline and τ1 retail to test
  transfer. If it does not transfer, that is a finding, not a failure.
