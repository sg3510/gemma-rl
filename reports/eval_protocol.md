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
| Any available backend | **pinned NextBit, bf16** | We measured ~30-point swings between OpenRouter backends for the same model id, and one backend corrupts tool calls outright. |
| Greedy decoding (implied) | **temperature 1.0** | pass@k is meaningless at temperature 0, and the baseline must be measured at the temperature RL rollouts are generated at. |
| Model not trained on τ-bench | **We will train on the τ-bench train split** | That is the entire experiment. |

Any one of these makes a submission non-standard. The last one alone makes it
**custom** by Sierra's own definition — their guide requires "a model or system
that was not trained specifically on τ-bench tasks, rewards, or evaluation
data". They anticipated this use case and gave it its own category.

We are not submitting. These numbers are "Gemma 4 26B A4B on τ-retail/telecom
**under this harness**", never a leaderboard position.

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
   step cap. Any change to the serving stack between arms contaminates the delta
   — the whole lesson of our provider investigation.
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
