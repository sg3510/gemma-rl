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

**B4 — no resume.**
`run()` writes a checkpoint but re-runs everything from scratch on restart.
Our runner scans completed `(task_id, trial)` cells and issues only the missing
ones.

---

## Open questions

- Does Gemma 4 26B A4B actually land in the 20–50% pass@1 band on retail?
- If it lands low, how much of that is mechanical (malformed tool calls, step-cap
  exhaustion) rather than genuine capability? The failure taxonomy answers this.
- How many of the 500 train tasks sit in the frontier band? That number decides
  whether there is enough signal to train on at all.
