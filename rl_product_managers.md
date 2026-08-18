# RL for Product Managers — decision log

A running record of every non-obvious decision made while building an RL demo on
τ-bench retail with Gemma 4 26B A4B. Written as we go, so the eventual blog post
is assembled from real decisions rather than reconstructed after the fact.

Each entry: **what we decided**, **what else we considered**, **why**, and
**what would change our mind**.

---

## The thesis

We want to show a small open model getting measurably better at a real agent
benchmark. Not a home-grown CRM toy — an established benchmark with an
objective verifier, so the improvement is not something we defined into
existence.

The pitch, if it works:

> Here is an established agent benchmark. Here is a small open model failing it.
> Here are its actual trajectories. Here is the environment's reward. Here is the
> training loop. And here is the same model becoming measurably more capable.

---

## D1 — Environment: τ-bench retail, not a synthetic toy

**Decided:** `sierra-research/tau-bench`, retail domain, pinned to commit
`59a200c6d575d595120f1cb70fea53cef0632f6b` (2026-03-18) as a git submodule.

**Considered:** a home-grown CRM environment; τ-bench airline; τ³ banking.

**Why:** retail gives us natural language + persistent state + written policy +
tools + multi-turn interaction + an objective verifier, all at once — and
somebody else already built and validated it. Airline is deliberately harder
(the original paper found removing the policy document barely dented retail but
crushed airline, because airline genuinely encodes arbitrary domain rules).
Banking is still hard for frontier models. Retail is where a 26B model has a
chance of landing in the band we need.

**Changes our mind:** if retail turns out too easy for Gemma 4 (pass@1 > 50%),
airline becomes the target instead.

---

## D2 — Split discipline: train / dev / test are not interchangeable

**Decided:**
- `dev` (20 tasks) — harness debugging only.
- `test` (115 tasks) — **held out**. Measured once for the headline number, then
  frozen. Only a sweep literally named `baseline_test` is permitted to touch it;
  the runner raises otherwise.
- `train` (500 tasks) — frontier discovery and, later, RL.

**Considered:** the original brief proposed picking ~20 retail tasks and using
them for both curriculum and evaluation.

**Why:** if we select training tasks by measured difficulty and then report
improvement on those same tasks, we have measured nothing except our own task
selection. τ-bench already ships the splits; using them costs nothing.

**Changes our mind:** nothing. This one is load-bearing.

---

## D3 — pass@k, not pass^k

**Decided:** compute our own metrics in `taurl/metrics.py`. Report both, always
labelled. tau-bench itself is never modified.

**The trap:** tau-bench's `display_metrics()` and the published leaderboard
report **pass^k** — the probability that *all* k rollouts succeed:

```
pass^k = mean_tasks[ C(c, k) / C(n, k) ]
```

We need **pass@k** — the probability that *at least one* of k succeeds:

```
pass@k = mean_tasks[ 1 - C(n-c, k) / C(n, k) ]
```

For a task solved 3 times out of 8: **pass@8 = 100%**, **pass^8 = 0%**. Same
task, same data. pass^k answers "is this agent reliable enough to ship";
pass@k answers "can the model do this at all", which is the question that
decides whether a task is worth training on.

**Why it matters here:** reading pass^k as pass@k would make the model look
hopeless on exactly the tasks that are most valuable for RL.

---

## D4 — The frontier criterion is per-task pass@1, not pass@8

**Decided:** a task enters the training frontier when its empirical success
fraction over 8 sampled rollouts satisfies `2/8 ≤ c/8 ≤ 6/8`.

**Clarification (this was initially stated loosely):** `c/n` is an estimate of
that task's **pass@1**, not its pass@8. pass@8 for any task with `c ≥ 1` is
1.0 by construction, so it cannot discriminate between tasks and is useless as
a selection criterion.

**Why:** GRPO computes advantage relative to the group mean, `A_i ≈ r_i - r̄`.
A rollout group of `[0,0,0,0,0,0,0,0]` gives every sample zero advantage. So
does `[1,1,1,1,1,1,1,1]`. The useful group looks like `[0,1,0,0,1,0,1,0]` — the
model demonstrably *can* solve the task but does not reliably, so there is a
gradient between the successes and the failures. Per-task pass@1 in the middle
of the range is precisely the condition for that.

The **aggregate** pass@8 over the selected set (target 80–95%) is a sanity check
on the set as a whole, not the selection rule.

---

## D5 — Agent served via OpenRouter, free tier first

**Decided:** `google/gemma-4-26b-a4b-it` through OpenRouter (litellm provider
`openrouter`). Start on the `:free` variant, flip to paid when rate limits hurt.
One `--tier` flag switches.

**Considered:** local 4-bit MLX on the M3 Pro / 36GB (the model is a 25.2B-total
/ 3.8B-active MoE, ~14GB at 4-bit, so it does fit); Google AI Studio.

**Why:** OpenRouter gives real native tool-calling, 262K context, and enough
concurrency to finish a sweep in about an hour instead of overnight. Paid is
$0.07/M in and $0.34/M out, so the entire study costs less than a sandwich.
Local is single-stream (10–20 hrs per sweep) and quantization would shift the
very number we are trying to measure.

**Changes our mind:** RL training will need local or rented weights anyway. This
decision is about the *measurement* phase only.

---

## D6 — The simulated customer is Gemini 3.7 Flash, fixed forever

**Decided:** `google/gemini-3.7-flash`, temperature 0, same OpenRouter key.
Pinned for every run and every future training step.

**Considered:** gpt-4o (tau-bench's default, hence leaderboard-comparable);
another Gemma; a fully scripted deterministic customer.

**Why not another Gemma:** if the customer and the agent are the same policy,
a failure is unattributable. *Did the agent fail, or did the customer behave
strangely?* The policy being optimized should be the only thing changing.

**Why not gpt-4o:** it costs more than the agent it is testing, and we would
need a second API key.

**The cost:** we are no longer strictly leaderboard-comparable. Every report has
to say so. Our number is "Gemma 4 26B A4B on τ-retail *under our harness*", not
a leaderboard entry.

---

## D7 — Pin the customer to temperature 0

**Decided:** inject `temperature=0.0` into the customer's completion calls.

**Why:** upstream `tau_bench/envs/user.py` calls `completion()` with no
temperature at all, inheriting the provider default (~1.0). That means even at
agent temperature 0 the run is non-deterministic — and worse, the variance
across our 8 rollouts would be a mixture of *agent* variance and *customer*
variance. We are trying to measure the former. Pinning the customer makes the
agent policy the only thing that varies.

**Changes our mind:** conversational-uncertainty tasks (the "Level 5" category,
where the customer withholds information until asked) may need customer
variation to be measured honestly. Revisit if the frontier is dominated by them.

---

## D8 — Measure at temperature 1.0

**Decided:** all measurement sweeps run the agent at temperature 1.0.
(Smoke tests use 0.0 for reproducibility.)

**Why:** pass@k is meaningless at temperature 0 — all k rollouts are the same
sample. And the baseline must be measured at the temperature RL rollouts will
actually be generated at, or the baseline is not the thing we are improving on.

---

## D9 — Keep the binary reward for now

**Decided:** the untouched τ-bench reward — replay ground-truth actions on a
fresh database, SHA-256 the resulting state, compare; plus every required output
string must appear verbatim. Any mismatch → 0.

**Considered:** immediately implementing the dense shaped reward
(+1 correct mutation / −2 incorrect / −0.5 omitted / −3 policy-violating).

**Why defer:** shaped reward is a training-time tool. The held-out evaluation
must stay the benchmark's own metric or we will fool ourselves — a model tuned
against our shaping would score well on our shaping. Get the honest baseline
first; add shaping when there is a training loop to feed.

---

## D10 — Patch tau-bench at runtime, never fork it

**Decided:** `vendor/tau-bench` is a submodule that stays byte-identical to
upstream (`git diff` against Sierra's repo is always empty). Everything we
change is monkey-patched in `taurl/patches.py`.

**Why:** we want to be able to say "we ran the real benchmark", and be able to
prove it. A fork makes that claim unauditable.

Both `tau_bench/agents/tool_calling_agent.py` and `tau_bench/envs/user.py` do
`from litellm import completion`, which binds the name into the module at import
time — so we rebind the module attribute rather than `litellm.completion`.

---

## Bugs found in tau-bench along the way

These are worth writing up: they are the kind of thing that silently corrupts a
benchmark number, and none of them are visible from the README.

**B1 — a crashed rollout is recorded as a legitimate failure.**
`run.py:89` wraps `agent.solve` in a bare `except Exception` and writes
`reward=0.0` with an `error` key in `info`. An exhausted retry budget or a
provider 500 is therefore indistinguishable from the agent failing the task, and
silently deflates pass@1. We detect the `error` key, exclude those rollouts from
the denominator, and re-run them.

**B2 — a "/" in the user-model name destroys the entire trial.**
`run.py:30` interpolates the user model into the checkpoint filename. For the
agent model it strips the vendor prefix (`.split('/')[-1]`); for the user model
it does not. `google/gemini-3.7-flash` becomes a directory separator, `open()`
raises — and it raises *outside* the per-task try/except, so `executor.map`
re-raises in the main thread and every completed result in that trial is thrown
away. Cost us one smoke run before we spotted it. We pass a slash-free alias and
restore the real id at the API boundary.

**B3 — the customer's cost is assigned, not accumulated.**
`envs/user.py` does `self.total_cost = res._hidden_params["response_cost"]` on
every turn, so it reports the last turn's cost rather than the conversation's.
It also bare-subscripts a key that is absent for models litellm does not know —
which is all three of ours. We register the OpenRouter prices with litellm and
guarantee the key exists.

**B5 — malformed tool arguments are indistinguishable from a provider outage.**
`agents/tool_calling_agent.py::message_to_action` does
`json.loads(tool_call["function"]["arguments"])` with no error handling. When the
model emits invalid JSON in a tool call the exception propagates into the same
bare `except Exception` as B1, and the rollout is recorded with an `error` key —
exactly like a rate-limit. This one bit us: our first pass treated it as
infrastructure, which both **inflated pass@1 by about 10 points** (real failures
were being dropped from the denominator) and would have made the runner retry a
deterministic failure forever. We now discriminate on the traceback: transport
faults are excluded and retried, model-output faults are scored 0 and given
their own `malformed_tool_args` taxonomy category. We also re-raise with the
offending argument string attached, which upstream discards.

**B6 — one backend provider silently destroys ~11% of all rollouts.**
Not a tau-bench bug so much as a bug tau-bench cannot survive. OpenRouter routes
a single model id across 8 backends, and Google's own endpoint merges parallel
tool calls into ONE `tool_call` whose `arguments` string holds two JSON objects
back to back:

```
{"order_id": "#W5765741"}{"order_id": "#W7736983"}
```

`message_to_action` does a bare `json.loads` on that (B5) and the rollout dies,
scored as a task failure. Measured impact on tau1 retail: **49 of 460 rollouts
(11%), touching 42 of 115 tasks**, every one recorded as the agent failing.
Retail pass@1 went from 65.9% to ~73% once the provider was excluded.

A controlled probe (a prompt inviting two parallel tool calls, 3 samples per
backend) isolated it cleanly:

```
DeepInfra, NextBit, SiliconFlow, Novita, Venice, Parasail   6/6 valid JSON
Google                                                      0/6 MERGED
Cloudflare                                                  rate-limited
```

This is the same failure class seen self-hosting Gemma 4 on vLLM without
`--enable-auto-tool-choice --tool-call-parser gemma4`: a server whose tool-call
parser does not understand the model's syntax. We cannot pass parser flags to
someone else's server, so we exclude the provider and record which backend
served every request.

We also added a repair net: if a merged payload arrives from an unblocked
backend, decode only the first JSON object. That is faithful rather than a
fudge — tau-bench itself keeps only the first tool call (`tool_calls[:1]`).
Repairs are counted and reported; unrepairable output still fails honestly.

**The lesson worth writing up:** a benchmark score for a hosted open model is
not a property of the model. It is a property of the model *and* whichever
anonymous backend served the request.

**B4 — no resume.**
`run()` writes a checkpoint but re-runs everything from scratch on restart.
Our runner scans completed `(task_id, trial)` cells and issues only the missing
ones.

---

## D11 — Retail is too easy; airline and telecom are the real targets

**Measured, held-out test splits, temperature 1.0, Gemini 3.7 Flash customer:**

| environment | pass@1 | verdict |
|---|---|---|
| τ1 retail | ~65% | **too easy** — far above the 20–50% band |
| τ1 airline | ~39% | **in band** |
| τ2 telecom | measuring | hardest domain by published error rates |

Retail landing at ~65% is the headline surprise. The original τ-bench authors
declined to test 7B/13B models because the benchmark was too hard, and GPT-4o
managed 61% on retail in 2024. A 2026 open 26B-A4B model (3.8B active) now
clears that. The capability frontier moved; the benchmark did not.

Airline at ~39% is exactly the band we wanted, which matches the original
finding that airline encodes far more arbitrary domain policy than retail —
removing the policy document barely dented retail scores but crushed airline.

**The catch:** τ1 airline ships a **test split only**. `airline/env.py` raises
for `train` and `dev`. So it is a fine evaluation target and a useless training
environment — selecting training tasks from those 50 and then reporting
improvement on the same 50 is precisely the D2 failure. Our split guard prints
a warning whenever airline touches `test` so this cannot happen silently.

That is what pushes us to τ2.

---

## D12 — Adopt τ2-bench for the training environment

**Decided:** add `sierra-research/tau2-bench` (pinned `a2c0247`) as a second
submodule in its own virtualenv.

**Why a second venv:** τ2 pins `litellm>=1.80.15,<1.82.7`; our τ1 harness runs
1.97.0. Rather than downgrade and risk changing τ1 numbers mid-study, the two
benchmarks get separate environments and a shared results reader
(`taurl/tau2_results.py`) so pass@k is computed identically for all three.

**Why τ2 telecom specifically:**

| domain | train / test | full pool |
|---|---|---|
| retail | 74 / 40 | 114 |
| airline | 30 / 20 | 50 |
| **telecom** | 74 / 40 | **2,285** |

Telecom is the only environment in the whole τ family with a task pool large
enough for real RL, it has a genuine train/test split, and published error
analysis marks it the hardest domain (98% duplicate-call, 53% max-steps failure
rates for baseline methods). It is also the most policy-dense: a 17KB
tech-support manual plus a 16KB workflow document, which is exactly the
policy-following axis the airline result says matters.

τ2 also adds *dual control* — the user has their own tools, not just speech.

**What we did not pick:** τ3 banking. Frontier models sit at 48–55% and the best
open-weight entry is 25%; Gemma would likely produce all-zero rollout groups.

**Bonus:** τ2 ships `--auto-resume`, `--max-retries` and `--max-concurrency`
natively, so most of the scaffolding we had to build for τ1 (B1–B4) is
unnecessary there.

---

## D13 — The serving backend moves the score more than anything we changed

**What happened:** OpenRouter fans one model id across 8 backend providers at
mixed precision — DeepInfra and SiliconFlow at **fp8**, NextBit/Novita/Parasail/
Venice at **bf16**, Google and Cloudflare undeclared. We were routing across all
of them, so every number we produced was a blend of serving stacks.

Then Google's endpoint turned out to be actively broken (see B6), and a
follow-up probe on τ2 telecom split the remaining backends by precision:

| group | pass@1 | n | `max_steps` | `infrastructure_error` |
|---|---|---|---|---|
| fp8 (DeepInfra, SiliconFlow) | 53.8% ± 13.8% | 13 | 0 | 5 |
| bf16 (NextBit, Novita, Parasail, Venice) | 22.2% ± 9.8% | 18 | 6 | 8 |

**This result is NOT trustworthy and we are not banking it.** Three confounds:

1. **Provider identity is not precision.** Each group mixes several vendors with
   different chat templates, tool parsers and sampling defaults. A 31-point gap
   between "fp8 vendors" and "bf16 vendors" says the vendors differ, not that
   fp8 beats bf16 — which would be backwards anyway, since fp8 is the lossy one.
2. **Non-random attrition.** Pinning providers with `allow_fallbacks: false`
   produced infrastructure errors on 5/18 and 8/26 attempts. Those dropouts are
   not random with respect to task difficulty, so both means are suspect.
3. **Tiny samples.** The two intervals overlap.

The real signal is the `max_steps` column: the bf16 vendor group loops until it
exhausts 200 steps a third of the time and the fp8 group never does. That is a
serving-configuration difference — sampling parameters or tool-call parsing —
not arithmetic precision.

**Decision:** stop treating "the model on OpenRouter" as one thing. Pin
measurement to a single named provider and record it with every number.

**Which precision for fine-tuning later: bf16.**
- The released Gemma 4 weights are bf16; that is the reference numerics.
- LoRA/GRPO trains adapters in bf16 (or a 4-bit quantized base with bf16
  adapters under QLoRA). fp8 is a *serving* optimization, not a training format.
- Once we self-host for RL we control the stack, and the eval must match what we
  train and serve or the improvement number is measuring the server.
- So the target stack is **self-hosted vLLM, bf16, with
  `--enable-auto-tool-choice --tool-call-parser gemma4`** — the flags whose
  absence caused B6. Hosted-API measurement is a stopgap until then.

---

## D14 — Our telecom number disagrees with the Artificial Analysis leaderboard

Artificial Analysis reports **44%** for Gemma 4 26B A4B on τ²-bench telecom.
Our runs have ranged **20%–54%** depending on configuration. Same model id.

Candidate explanations, ranked by how much evidence we actually have:

1. **Serving backend — demonstrated, large.** See D13/B6. Google's endpoint
   broke tool calls outright; provider groups differ by ~30 points. This alone
   spans the disagreement.
2. **Sample size — demonstrated.** Our telecom estimate moved 28.6% → 33.3% →
   23.7% as n grew from 14 to 38. Standard error at n≈15 is ±12pp, so 44% was
   never more than ~1.3 SE from our first reading. We should not have quoted a
   14-sim number.
3. **Reasoning mode — tested, inconclusive.** AA marks the model with a
   reasoning lightbulb; we run reasoning off. A direct ablation gave 20.0%
   (n=15, reasoning on, temp 1.0) vs 27.3% (n=11, reasoning on, temp 0) vs
   23.7% (n=38, reasoning off). No separable effect at these sample sizes.
4. **Temperature — untested, expected.** We measure at temp 1.0 because pass@k
   is meaningless greedy; a leaderboard reports greedy. This should cost us
   points relative to AA, in the observed direction.
5. **Task set — untested.** We use τ2's 40-task `test` split. AA does not
   publish which set it runs; `base` is 114 tasks and `full` is 2,285.

**What this means for the project:** a hosted-API pass@1 is not a stable
quantity. It is a measurement of a model *and* a serving stack *and* a sampling
config. For a before/after RL claim that is fatal unless every one of those is
pinned — which is the strongest argument yet for self-hosting before we train.

---

## D15 — What this has cost

OpenRouter spend through the baseline and diagnostic phase: **$22.40**.

| phase | cost |
|---|---|
| τ1 retail baseline (115 tasks × 4 trials, 460 rollouts) | $3.66 |
| τ1 retail re-run after the provider fix | $4.90 |
| τ1 airline, τ2 retail/airline/telecom, and all diagnostics | remainder |

Roughly **$0.015 per trajectory** — agent ~$0.007, customer simulator ~$0.008.
The customer costs about as much as the model under test, which is why we did
not use gpt-4o for it (D6).

Notably, **the diagnostics cost more than the baseline they corrected**, and
were worth it: without them we would have published a retail number understated
by ~8 points and a telecom number off by up to 30.

---

## Open questions

- Does Gemma 4 26B A4B actually land in the 20–50% pass@1 band on retail?
- If it lands low, how much of that is mechanical (malformed tool calls, step-cap
  exhaustion) rather than genuine capability? The failure taxonomy answers this.
- How many of the 500 train tasks sit in the frontier band? That number decides
  whether there is enough signal to train on at all.
