# Serving backends do not agree on telecom, and our pinned one is the worst

**Same model id, same weights, same temperature, same 20 tasks, same customer
simulator.** Loop rate — the share of scored rollouts that repeat one identical
tool call ≥10 times — varies **5×** across OpenRouter backends, and the backend
we pinned every headline number to (D13/D14) is the outlier.

| backend | precision | scored | looped | **loop rate** | pass rate |
|---|---|---|---|---|---|
| **NextBit** *(our pinned stack)* | bf16 | 72 | 23 | **32%** | 60% |
| DeepInfra | fp8 | 40 | 4 | **10%** | 82% |
| **Novita** | bf16 | 34 | 2 | **6%** | **91%** |

NextBit vs Novita: Fisher exact two-sided **p = 0.003**.

**Precision is not the axis.** The two bf16 stacks sit at opposite extremes,
which also retires the fp8-vs-bf16 framing left over from D14's quantization
probe. Whatever differs is backend software: tool-call parser, chat template,
handling of `content: null` on tool-call turns, or whether the host applies the
model's `generation_config.json`.

## Are we comparing the same tests?

Yes, with two caveats that must travel with these numbers.

- **Same task ids.** Novita and DeepInfra ran an identical 20-task set
  (verified by set equality); the NextBit row is its repaired-grid rollouts
  **restricted to those same 20 ids**, not its full 40-task run.
- **Different trial depth.** NextBit has up to 4 trials on those tasks (72
  scored) against 2 trials each for Novita (34) and DeepInfra (40). Rates are
  per-rollout, so this affects precision of the estimate, not its centre.
- **Easier slice.** `--num-tasks 20` takes the first 20 of the `test` split:
  11 `service_issue` + 9 `mobile_data_issue`, and **no `mms_issue`** — the hard
  archetype, which on NextBit's full grid loops at 66% (35/53). So all three
  absolute rates here are optimistic relative to the domain as a whole.
- Novita lost 6 of 40 cells and NextBit's full grid lost 19 of 160 to the
  empty-completion pathology below; those are excluded, not counted as failures.

## What this does to earlier decisions

**D13/D14 chose NextBit on the wrong evidence.** The provider comparison behind
that decision was 10 tasks on **τ1 retail**, scored on pass rate, and concluded
"the four working backends agree within sampling noise". On τ2 telecom, with
enough rollouts and with loop rate as the metric, they demonstrably do not. The
earlier probe was too small, on the wrong domain, and measured the wrong thing.

**The headline telecom baseline is substantially a NextBit artifact.** On this
slice NextBit passes 60% where Novita passes 91%. Our pass@1 of 41.9% is not a
property of Gemma 4 26B A4B on telecom; it is a property of Gemma-served-by-
NextBit. Every downstream number inherits that — the frontier band, the
scenario ladder, the power analysis.

**Backend selection must be re-run** on telecom, at n≥4, with loop rate as a
first-class criterion alongside pass rate and tool-call validity.

## The second NextBit pathology: empty completions

Re-running the 41 excluded cells of `pinned_telecom` produced 19 fresh
`infrastructure_error`s with an identical signature: tau2 raises
`AssistantMessage must have either content or tool_calls` — the backend returned
a **completely empty completion**. They are task-correlated (the same, mostly
`mms_issue`, tasks fail again), so this is not random provider flakiness. It
costs us 12% of the grid and is why a run-level pass@4 still does not exist:
only 28 of 40 tasks reached 4 scored rollouts.

A retry-on-empty in the harness recovers these without touching semantics, and
should exist before any training run — an empty completion mid-rollout during
GRPO would otherwise poison a group.

## Sampling parameters were never pinned

We send `temperature: 1.0` and nothing else. `top_p`, `top_k`,
`repetition_penalty` and `min_p` fall back to each host's **undocumented**
defaults. OpenRouter's endpoints API publishes which parameters a backend
*accepts*, never what it applies when you omit them:

| backend | exposes `top_k` | exposes `min_p` |
|---|---|---|
| NextBit | **no** | no |
| Novita | yes | no |
| DeepInfra | yes | yes |

Meanwhile Gemma 4 ships its own canonical config
(`generation_config.json`: `temperature 1.0, top_k 64, top_p 0.95`), so a host
that applies it and one that does not are running visibly different samplers on
identical weights. **NextBit cannot apply `top_k` at all** — it does not expose
the parameter — which makes it structurally incapable of the vendor-recommended
configuration.

Tested directly (replay probe, depth 5, NextBit): `top_p=0.95` gives **87%**
repeat probability, identical to the unset control's 87%. So canonical `top_p`
alone does not explain or fix the gap. `top_k=64` remains untested on NextBit
for the reason above.

**Consequence:** whichever sampling config we adopt, it must be sent explicitly
and recorded in run metadata like every other pinned knob. Note the interaction
with `require_parameters: true` — sending `top_k` would make OpenRouter refuse
NextBit outright, so canonical sampling and our current pinned backend are
mutually exclusive.

## Not usable as a backend

| backend | status |
|---|---|
| Google (OpenRouter) | merges parallel tool calls into one malformed call (B6) — excluded since day one |
| SiliconFlow, Parasail | do not serve `tools` (NotFoundError) |
| Google AI Studio | quota is 16k input tokens/min for `gemma-4-26b`; one telecom turn is 5–10k. Not viable for a 40×4 run, which is why the strongest independent check went unrun |

## Decision: switch the pinned backend to Novita

| | NextBit | **Novita** | DeepInfra |
|---|---|---|---|
| precision | bf16 | **bf16** | fp8 |
| loop rate (same 20 tasks) | 32% | **6%** | 10% |
| pass rate (same 20 tasks) | 60% | **91%** | 82% |
| empty-completion errors | **112** (one run) | 2 | 3 |
| cells lost to infrastructure | 19/160 | 5/40 | 0/40 |
| median rollout | 103 s | 112 s | 125 s |
| median messages/trace | 94 | **58** | 60 |
| price in / out per M tokens | $0.10 / $0.40 | $0.13 / $0.40 | $0.07 / $0.34 |
| serves `top_k` (Gemma's config) | **no** | **yes** | yes |
| max completion tokens | 262 144 | 131 072 | 16 384 |

The median-messages column tells the same story from another angle: NextBit
traces run 94 messages to Novita's 58 because half of them are padded with loop
repetitions.

Novita costs ~30% more on input tokens — noise against the customer-simulator
bill — and its 131K context is irrelevant at a 140-step cap. It is the only bf16
backend that can serve `top_k`, hence the only one capable of Gemma's own
`generation_config`.

**Plan of record:** re-baseline 40 × 4 telecom on Novita with sampling pinned
explicitly (`temperature 1.0, top_p 0.95, top_k 64`), retain the NextBit numbers
as the cautionary record, and treat the switch as a deliberate re-baseline —
both eval arms on the new stack, never one arm each.

## Novita is not a substitute for AI Studio

Worth stating because the two get conflated. Novita is a third-party GPU cloud
serving the open weights, near-certainly via vLLM or SGLang with the community
chat template and tool-call parser. Google AI Studio is **first-party** serving:
different template application, different tool-call parsing, its own sampling
defaults, possible internal quantization Google does not publish, its own safety
filtering, behind a different API surface. Measuring Novita tells us nothing
about AI Studio, which remains quota-blocked and untested on telecom. Neither is
the eventual self-hosted vLLM either, so both stay provisional until Stage A.

## What we still do not know

- **Which backend difference causes it.** Parser, template, or sampling
  defaults — undiagnosable from the client side. Self-hosted vLLM (Stage A),
  where we control all three, is the definitive test.
- **Whether Novita's advantage holds on `mms_issue`**, the hard archetype absent
  from this slice, and at n≥4.
- Whether the low-loop backends also avoid the empty-completion failure.

Both gaps are cheap to close and should be closed before the baseline is quoted
again.

## Related

- [tool_call_looping.md](tool_call_looping.md) — the failure mode itself, and
  the replay evidence that the attractor is in the model
- [leaderboard_vs_ours.md](leaderboard_vs_ours.md) — the original pinned-stack
  argument, whose conclusion this report revises
