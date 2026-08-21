# Why our numbers differ from the published leaderboards

Artificial Analysis reports **44%** for Gemma 4 26B A4B on τ²-bench telecom.
Across this project our measurements of *the same model id* ranged from **20% to
54%**. This documents what caused that spread, what we fixed, and what remains a
legitimate difference.

The short version: **a benchmark score for a hosted open-weights model is not a
property of the model.** It is a property of the model, the anonymous backend
that served it, the sampling configuration, and the task subset — and three of
those four are usually unreported.

---

## Where we landed

| configuration | telecom pass@1 |
|---|---|
| Artificial Analysis (published) | **44%** |
| ours, unpinned routing, first read (n=14) | 28.6% |
| ours, unpinned routing, more data (n=38) | 23.7% |
| ours, **pinned to one backend** (n=10) | **40.0%** |

Pinning the serving stack moved us from ~24% to ~40% and largely closed the gap.
The residual 4 points sit well inside sampling error at these sample sizes.

---

## Cause 1 — The serving backend (dominant, ~30 points)

OpenRouter routes a single model id across **8 backend providers** at mixed
precision. They are not equivalent.

**One of them is outright broken for tool use.** Google's endpoint merges
parallel tool calls into a single `tool_call` whose `arguments` string contains
two concatenated JSON objects:

```
{"order_id": "#W5765741"}{"order_id": "#W7736983"}
```

τ-bench calls `json.loads()` on that with no error handling, the exception is
caught by a blanket `except Exception`, and the rollout is recorded as
`reward=0.0` — indistinguishable from the agent failing the task.

Measured impact on τ1 retail: **49 of 460 rollouts (11%), across 42 of 115
tasks**, every one silently scored as a task failure. Excluding the provider
moved retail pass@1 from 65.9% to ~73%.

A controlled probe (a prompt inviting two parallel tool calls, 3 samples each):

| backend | valid parallel tool calls |
|---|---|
| DeepInfra, NextBit, SiliconFlow, Novita, Venice, Parasail | 6/6 |
| **Google** | **0/6 — merged** |
| Cloudflare | rate-limited, untested |

This is the same failure class seen when self-hosting Gemma 4 on vLLM without
`--enable-auto-tool-choice --tool-call-parser gemma4`: a server whose tool-call
parser does not understand the model's output syntax. You cannot pass parser
flags to someone else's server, so we excluded the backend and now record which
one served every request.

**Two more backends cannot serve tools at all.** SiliconFlow and Parasail return
`NotFoundError` for any request carrying a `tools` payload — consistent with
their `tools=False` flag in OpenRouter's endpoint listing.

### The trap this set for us

Our first quantization comparison grouped backends by precision and reported
**fp8 53.8% vs bf16 22.2%** — a large, confident, publishable-looking number.

It was entirely an artifact. SiliconFlow (fp8) and Parasail (bf16) failed 100%
of their calls, and that non-random attrition dragged both group means apart.
Measured per backend instead of per precision group, on 10 identical tasks:

| backend | quant | completed | reward | infra errors |
|---|---|---|---|---|
| DeepInfra | fp8 | 10/10 | 70% | 0 |
| NextBit | bf16 | 10/10 | 70% | 0 |
| Novita | bf16 | 10/10 | 70% | 0 |
| Venice | bf16 | 10/10 | 80% | 0 |
| SiliconFlow | fp8 | 0/10 | — | 10 |
| Parasail | bf16 | 0/10 | — | 10 |

The four working backends agree within sampling noise. **fp8 and bf16 are
indistinguishable here; the 31-point gap was measurement error.**

What saved us was refusing to bank the result and looking at the attrition
column.

---

## Cause 2 — Sample size (large early, our own fault)

Our telecom estimate moved **28.6% (n=14) → 33.3% (n=18) → 23.7% (n=38)**.

At n≈15 the standard error is **±12 points**. The 44% reference was never more
than ~1.3 standard errors from our first reading. We should not have quoted a
14-simulation number as though it disagreed with anything.

---

## Cause 3 — Sampling temperature (real, deliberate, unresolved)

We measure at **temperature 1.0**. A leaderboard reports greedy decoding.

This is not a mistake we intend to fix. pass@k is meaningless at temperature 0 —
all k samples are identical — and the baseline has to be measured at the
temperature RL rollouts will actually be generated at, or it is not the baseline
we are trying to improve on.

It should cost us a few points relative to a greedy leaderboard number, in the
direction observed. We have not isolated its size.

---

## Cause 4 — Reasoning mode (tested, inconclusive)

Artificial Analysis marks this model with a reasoning-model icon. The model
exposes `reasoning` / `include_reasoning` parameters and we run with reasoning
off. A direct ablation on telecom:

| configuration | pass@1 | n |
|---|---|---|
| reasoning off, temp 1.0 | 23.7% | 38 |
| reasoning on, temp 1.0 | 20.0% | 15 |
| reasoning on, temp 0.0 | 27.3% | 11 |

No separable effect at these sample sizes. Worth revisiting with more data
before claiming anything.

---

## Cause 5 — Task subset (unresolved)

We run τ2's **40-task `test` split**. Artificial Analysis does not publish which
set it uses; τ2 telecom ships `small` (20), `test` (40), `train` (74),
`base` (114) and `full` (2,285). Different subsets have different difficulty and
this is simply not reconcilable from the outside.

---

## Cause 6 — Our own harness deviations (declared)

Two, both deliberate and both reported with every number:

1. **Customer simulator is `gemini-3.7-flash` at temperature 0**, not τ-bench's
   default `gpt-4o`. Cheaper than the model under test, and deliberately not a
   Gemma so a failure is attributable. Upstream passes *no* temperature to the
   customer at all, inheriting ~1.0, which would mix customer variance into our
   pass@k.
2. **τ2 step cap lowered from 200 to 140.** Across 199 observed simulations the
   longest *successful* trajectory was 136 messages (median 34, p95 74), while
   78 hopeless loops burned exactly 202 each — about 75% of all message volume.
   140 sits above every success on record, so it costs nothing measurable and
   cuts 24% of the bill. We declined a cap of 100 (39% cheaper) because it would
   have discarded one real success.

---

## What we would need to fully reconcile

- The exact task set and split Artificial Analysis runs
- Their decoding temperature and whether reasoning was enabled
- Their serving stack and precision
- Enough samples on both sides for the intervals to actually separate

None of that is published, which is the point.

---

## Leaderboards report no uncertainty, and the uncertainty they omit is not the one you would guess

Published agent leaderboards present scores as bare numbers. Applying the
binomial to Artificial Analysis's τ²-telecom chart (18 models), at 114 tasks
with one trial each:

| trials | 95% CI half-width @ p=.44 | adjacent pairs that are statistically indistinguishable |
|---|---|---|
| 1 (114 sims) | **±9.1 pts** | **15 of 17** |
| 4 (456 sims) | ±4.6 pts | 11 of 17 |

So "Qwen3.5 9B 87% beats Qwen3.5 35B A3B 86%" is a coin flip presented as a
ranking, and the 44% vs 37% gap below Gemma is not resolvable at that sample
size.

One fairness caveat, the same correction we had to apply to our own numbers:
every model runs the *same* tasks, so cross-model comparisons are **paired** and
more powerful than the unpaired arithmetic above. Their rankings are better than
these figures imply. They do not report that either.

### But sampling error is the *small* term

| source of uncertainty | magnitude |
|---|---|
| sampling (456 sims, p=.44) | ±4.6 pts |
| **serving backend, same model id** | **~30 pts** |
| our own readings of Gemma 4 26B A4B on this exact benchmark | **20% – 54%** |

An error bar would capture the first and entirely miss the second, which for a
hosted open-weights model is roughly six times larger. Publishing ±4.6% while
the dominant uncertainty is *which anonymous server answered the request* would
be worse than publishing nothing: it manufactures precision that does not exist.

**So the fix is not error bars. It is reporting the serving configuration** —
backend, precision, temperature, task split, step cap. For closed API models
most of that is determinate. For open-weights models routed through resellers it
is the single largest unreported variable, and essentially nobody publishes it.

In fairness: confidence intervals clutter a bar chart, the audience wants an
ordering, and this is an industry-wide norm rather than one publisher's lapse.
The practical reading is that a leaderboard position for an open-weights model
means "roughly this tier", not "beats the model on the next bar down".

This is also the strongest argument for the shape of our own claim: a
**before/after delta on a pinned stack** is more defensible than any absolute
number, because pinning eliminates precisely the variance that dwarfs the error
bars.

---

## The methodological takeaway

If you intend to make a before/after claim about a model — which is exactly what
an RL result is — every one of these must be pinned and reported, or the
"improvement" may be the serving stack moving underneath you. Our current pin:

| knob | value |
|---|---|
| backend | OpenRouter → **NextBit only**, `allow_fallbacks: false`, `require_parameters: true` — **superseded 2026-08-20, see [provider_loop_rates.md](provider_loop_rates.md)** |
| precision | bf16 (matches what we would fine-tune in) |
| agent temperature | 1.0 |
| customer | `gemini-3.7-flash` @ temp 0 |
| step cap | 140 (τ2) / 30 (τ1) |
| task split | `test`, held out |

The honest end state is self-hosting — vLLM at bf16 with
`--enable-auto-tool-choice --tool-call-parser gemma4` — because then the stack
is ours and cannot silently change between the "before" and the "after".
