# The dominant failure mode is a tool-call loop, not missing capability

Found while building the trace viewer, by reading `pinned_telecom` attempt by
attempt. It changes what our headline number means.

**41% of scored rollouts repeat one identical tool call ten or more times until
the step cap kills them.** Not "sometimes re-fetches" — locks up.

## The measurement

τ2 telecom, `pinned_telecom`: 40 held-out `test` tasks × 4 trials, NextBit bf16,
agent temp 1.0, customer Gemini 3.7 Flash temp 0, step cap 140. 160 cells, 41
lost to infrastructure errors, **119 scored**.

A "loop" here means one exact call signature (same tool name, same arguments)
appearing ≥10 times in a trajectory.

| outcome | n | median calls | unique calls | most-repeated call | longest back-to-back run |
|---|---|---|---|---|---|
| pass | 56 | 10 | 8 | ×3 | ×1 |
| `user_stop` failure | 12 | 9 | 6 | ×4 | ×2 |
| **`max_steps` failure** | **50** | **62** | **7** | **×54** | **×51** |

The median looping trace makes 62 tool calls of which 7 are distinct, fires one
identical call 54 times with 51 of them consecutive, and burns 132 of its 140
messages doing it. Passing traces have no consecutive repeat at all.

## What is being looped, and why it looks like a stuck read-poll

- **One tool, overwhelmingly.** 44 of 49 loops are `get_details_by_id`; the rest
  are `get_customer_by_id` / `get_customer_by_phone`. All reads.
- **Nothing had changed the state.** Only **4 of 49** traces had made any
  agent-side write before the loop began (`taurl/loops.py`; an earlier count of
  6 also included customer-device toggles, which are not the agent's actions).
  The agent re-reads line details waiting for a status change it never caused.
- **The polled read cannot ever show the change.** 38 of 49 loops are on tasks
  whose solution requires a *customer device action* — and user-side tool
  results never enter the agent's context (tau2 filters them out), while
  `get_details_by_id` reads the carrier DB, which device actions do not update.
  The agent is polling a data source that is structurally blind to the thing it
  is waiting for.
- **The conversation is still alive.** The customer keeps replying through the
  loop in 40 of 49 traces (median 6 customer turns inside the looping stretch),
  so this is not a dead socket — the agent talks *and* re-polls.
- **Not a sampling artifact.** The reasoning probes loop at 73% (temp 0) and 71%
  (temp 1).

Telecom's scenarios prime the pattern: the customer is instructed that "if the
tool call does not return updated status information, you might need to perform
another tool call". The agent appears to adopt that stance itself and poll for a
change only its own write could produce.

## Where it concentrates

| scenario | looped rollouts | pass@1 **conditional on not looping** |
|---|---|---|
| `mms_issue` | 27 / 42 (64%) | 6/15 = 40% |
| `service_issue` | 15 / 53 (28%) | 34/38 = 89% |
| `mobile_data_issue` | 7 / 24 (29%) | 16/17 = 94% |
| **all** | **49 / 119 (41%)** | **56/70 = 80%** |

No looping rollout passed, by construction — a loop ends at the step cap, and τ2
skips evaluation entirely on a premature termination, recording reward 0.0 with
every check list null. So **50 of 160 rollouts were never scored against the
task criteria at all**. Unevaluated is not the same as failed.

## What is left when the loops are removed

63 scored failures = 49 loops + **14 that are not**. Of those 14:

- 13 ended cleanly on `user_stop` and *were* evaluated;
- most are near-misses — several sit at 2 of 11 or 2 of 12 criteria unmet, and
  τ2 multiplies over the reward basis, so nine satisfied checks still score 0;
- 9 of 14 are `mms_issue`;
- 10 of the 14 still repeat some call 3–8 times, and one `max_steps` trace
  repeats a call 9 times — just under the threshold.

Move the loop line to ≥5 repeats and the clean-failure count falls to about 6.
**The honest range for genuine capability failures is 6–14 of 63.**

## What this changes

1. **The 43% baseline is substantially a measure of loop propensity.** Any
   before/after must report *two* numbers: loop rate, and pass@1 among
   non-looping rollouts. The second is the capability claim, and it currently
   stands at 80% — far less headroom than 43% suggests.
2. **The frontier band is partly loop-propensity too.** A task lands in
   [0.25, 0.75] largely because it locked up on some trials and not others. The
   curriculum must be re-measured once looping is fixed, not before.
3. **The environment is bimodal underneath.** Conditional on not looping,
   `service_issue` (89%) and `mobile_data_issue` (94%) are near ceiling and give
   no gradient; `mms_issue` (40%) is the frontier. That is fortunate: `mms_issue`
   is 1,929 of the 2,151 pool tasks outside any curated split, so the curriculum
   is effectively inexhaustible — but 41 of the 74 `train` tasks would be spent
   on ceiling scenarios.
4. **The balanced `test` split loses power.** It is 15 `service_issue` +
   9 `mobile_data_issue` + 16 `mms_issue`. If two archetypes sit at ~90%, the
   paired delta must come from 16 tasks, and the 89% power figure in
   [eval_protocol.md](eval_protocol.md) — which assumes all 40 carry signal — is
   optimistic. An mms-heavy held-out eval drawn from the unused pool costs ~$24
   per arm at 40 × 8.
5. **Fix order.** A repeat-call guard in the scaffold is hours and free. Behaviour
   cloning on the non-looping successes targets the same failure offline. GRPO
   should be aimed at the residual capability gap, not at this — spending the run
   to fix something a scaffold guard also fixes makes the delta unreportable.
6. **Stage A needs a loop-rate gate**, alongside "lands near baseline" and "no
   malformed tool args". If self-hosted vLLM shows a materially different rate,
   part of this was a serving artifact.

## Caveats

- **The conditional rates are selection-biased upward.** Rollouts that avoided
  the loop may be the easier instances, so 80% is a ceiling on the estimate, not
  an unbiased one. `mms_issue`'s 40% rests on 15 rollouts.
- **The ≥10 threshold is a choice**, which is why the 6–14 range is quoted rather
  than a single clean-failure count.
- **One run, one stack.** The fp8 probe looped on 1 of 14 rollouts against bf16's
  7 of 12, which hints at a serving-side component — but n≈13 per arm and the
  backends differ, so that is a hypothesis for Stage A, not a finding.
- Loop rate in the earlier unpinned `baseline_telecom` was 62%, at step cap 200.
  A higher cap gives a loop more room; the cap is not the cause.

## Investigation: plumbing or model? (2026-08-20)

Hypotheses tested, in order of cheapness. Tooling: `taurl/loops.py` (detection +
taxonomy, regression-tested), `tau2probe/wire_audit.py` (serialization audit),
`tau2probe/replay_probe.py` (prefix replay against the pinned stack).

### Wire format: clean (H1 rejected)

Pushing looping and passing conversations through tau2's real
`to_litellm_messages`: every tool result goes as `role:"tool"` with a
`tool_call_id` matching a prior assistant call; no reasoning content is ever
replayed. Two nonstandard shapes exist — `content: null` on assistant tool-call
turns and a duplicate top-level `"name"` key in each tool_call — but they appear
identically in passing traces, and the replay probe's normalized-payload variant
(D below) shows they do not drive the loop.

(Related caveat: the `probe_reasoning_*` runs had `reasoning_effort` on while
tau2 drops reasoning between turns, which may inflate those probes' 73%/71%
loop rates. `pinned_telecom` did not use reasoning.)

### Prefix replay: the loop is a self-reinforcing attractor

15 looping conversations truncated just after the model has seen the *same*
tool result N times, 4 sampled continuations each against the same pinned
NextBit stack. Fraction that immediately re-issue the identical call:

| identical results already in context | repeat probability |
|---|---|
| 2 | **23%** (14/60) |
| 5 | **87%** (52/60) |
| 8 | **97%** (58/60) |

The recorded lock-ups are not a deterministic fault: at the first-repeat
decision point the model escapes 77% of the time. But each repetition in
context makes the next more likely, and by five identical calls the process is
essentially absorbed. This is the standard degenerate-repetition mechanism, and
it independently justifies the breaker threshold: at ≥5 consecutive the episode
is already lost with ~87–97% probability. (Selection caveat: these prefixes come
from traces that *did* eventually loop, so 23% is the entry rate conditional on
eventual absorption, an overestimate of the unconditional per-decision rate.)

What the model does in the 46 continuations that *don't* repeat: 17 pure-text
turns, 29 *different read calls*, **zero write actions**. Even off the exact
loop it stays in read-space at this decision point — consistent with the
passive-polling diagnosis rather than with a model that knows the fix and
occasionally slips.

### Interventions at the depth-2 decision point (n=60 each)

| variant | repeat rate | verdict |
|---|---|---|
| A — control | 23% | baseline |
| B — dedup note appended to the repeated result | **40%** | **backfires** — restating the call re-primes it |
| C — causal guidance in the system prompt | 22% | no effect |
| D — normalized payload (no null content, no duplicate `name`) | 30% | no effect → H1 dead |

Two practical lessons: the failure is **not prompt-fixable** with the obvious
paragraph (C), and the obvious harness "help" (B) makes it worse — any
repair-guard design must break the pattern (truncate/summarise the repetition)
rather than annotate it.

### The "infrastructure errors" have a signature too

Re-running the 41 excluded cells shows they are not provider 500s: tau2 raises
`AssistantMessage must have either content or tool_calls` — the backend returned
a **completely empty completion** — and the retries are task-correlated (the
same, mostly `mms_issue`, tasks fail again). **19 of the 41 re-failed the same
way**, so the repaired grid is 141 scored rollouts, not 160, and a run-level
pass@4 still does not exist: only 28 of 40 tasks reached 4 scored rollouts.
A harness retry-on-empty would recover these without touching semantics; it
should exist before any training run, since an empty completion mid-rollout
during GRPO would poison a whole group.

### Repaired-grid numbers

| | pre-repair | repaired |
|---|---|---|
| scored rollouts | 119 | **141** |
| pass@1 (macro) | 43.1% | **41.9%** |
| pass@2 | — | 59.2% |
| loop rate | 41% | **45%** |
| loops with a prior agent write | 4/49 | **5/63** |
| loops needing a customer device action | 38/49 | **51/63** |
| tasks at n=4 | 17/40 | 28/40 |

### Sampling parameters (depth-5 control = 87%)

Gemma 4 ships `top_k: 64, top_p: 0.95` in its `generation_config.json`; we had
been sending `temperature: 1.0` and nothing else, leaving truncation to each
host's undocumented default. Tested on NextBit:

| sampling | repeat probability at depth 5 | malformed tool JSON |
|---|---|---|
| unset (control) | 87% (52/60) | 0 |
| `top_p=0.95` (canonical) | **87%** (52/60) | 0 |
| `repetition_penalty=1.05` | not pursued (n=5) | — |

Canonical `top_p` changes nothing. `top_k=64` could not be tested here: NextBit
does not expose the parameter, and with `require_parameters: true` sending it
would make OpenRouter refuse the backend outright — so **our pinned stack is
structurally incapable of the vendor-recommended sampling config.**

`repetition_penalty` was deliberately abandoned rather than measured out. Even
if it suppressed the loop it is the wrong instrument: it penalises tokens by
prior occurrence, and agentic traces legitimately repeat tool names, customer
ids and JSON syntax constantly, so it risks corrupting exactly the structured
output the harness scores. And like a dedup guard it hides the deficiency in
the sampler instead of fixing it.

### Cross-stack loop rates: the serving stack dominates the ENTRY rate

Same 20 test tasks, same weights, same temperature — full detail and caveats in
[provider_loop_rates.md](provider_loop_rates.md):

| stack | precision | scored | loop rate | pass rate |
|---|---|---|---|---|
| **NextBit** (pinned) | bf16 | 72 | **32%** | 60% |
| DeepInfra | fp8 | 40 | 10% | 82% |
| **Novita** | bf16 | 34 | **6%** | **91%** |

NextBit vs Novita p = 0.003. Precision is not the axis — the two bf16 stacks
are the extremes. Google AI Studio, planned as the strongest independent check,
is infeasible: its quota for `gemma-4-26b` is 16k input tokens/min against
5–10k-token telecom turns.

### Removing the tool does not help — the loop migrates

The loop concentrates so heavily on `get_details_by_id` (55 of 63) that the
obvious question is whether that one tool is the problem: it always succeeds,
never mutates, and never advances the task. Tested directly by deleting it from
the agent's schema and re-running the four tasks that lock up in **100%** of
their baseline rollouts (4 trials each, `tau2probe/ablate_tool.py --mode drop`):

| | baseline | without `get_details_by_id` |
|---|---|---|
| loop rate | 100% (by selection) | **86%** (6/7 scored) |
| looped tool | `get_details_by_id` | **`get_customer_by_id` ×4, `get_customer_by_phone` ×2** |
| loops with a prior agent write | 0 | **0/6** |
| solved | 0 | 1/7 |

**The loop simply moved to the next available read.** Tool identity is
incidental; the failure is the polling behaviour. That rules out any tool-level
fix — blacklisting, reordering, description edits — and it means a dynamic
masker must **cascade** rather than mask a single tool.

(One rollout did solve the task without the tool at all, which is mildly
interesting given these tasks score 0 in every baseline rollout — but n=1 and
reward was never this test's metric.)

### Dynamic tool masking: promising, unproven

The intervention that fits the evidence: hide a tool *while* the agent is
looping on it (after 3 consecutive identical calls), restore it once the agent
does anything else, cascade to whatever it migrates to. It breaks the pattern
rather than annotating it — there is no restated call to re-prime from — and it
is surgical where `repetition_penalty` is global and reversible where deletion
is permanent. `tau2probe/loop_mask.py`.

First run on the same four tasks was **inconclusive because of the backend**:
of 6 completed cells, 5 were empty-completion deaths and 1 was a clean
`user_stop` **that solved the task**. Encouraging but n=1, and the empty
completions appeared at a higher per-cell rate than in the drop arm, which is
consistent with — though not proof of — the predicted out-of-distribution risk:
the model tries to call a tool it can no longer see, and the backend emits
nothing. Retest on Novita, and add a synthetic "tool temporarily unavailable"
result so the conversation stays well-formed instead of relying on the backend
to cope.

**If it works it is still training-time-only.** Masking rescues an episode, so
it deletes the gradient that teaches the policy not to loop; used in training it
must carry a small per-activation reward penalty (under the lexicographic gate
in [tiered_reward.md](tiered_reward.md)), and evaluation must always run
unmasked with loop rate reported. That is the only way to show the policy
learned something rather than leaning on the masker.

### Idea parked: rescuing degenerate all-fail tasks into the frontier

A task that scores 0 on every rollout gives GRPO no gradient. But a 0/n caused
by a **lock-up** says nothing about whether the model can do the task — so if
masking or ablation produces even one success there, the task moves into the
frontier band and becomes trainable.

Checked against the Novita grid (partial, n=2-3). Of 6 all-fail tasks, all
`mms_issue`:

| failure shape | tasks | verdict |
|---|---|---|
| `user_stop`, no loop (max consecutive 1-3) | **5** | genuine capability failure — the conversation finished and the end state was wrong. No tool intervention will manufacture a success. |
| `max_steps`, both rollouts looped, max consecutive **59** | **1** | a lock-up wearing a capability failure's clothes. Prime candidate. |

So the idea recovers roughly **1 task in 6** on a clean stack. On the old NextBit
grid, where 40% of rollouts were lock-ups, it would have been worth far more —
another way of saying the backend was hiding the real difficulty distribution.

**The catch that decides how such a success may be used.** A trajectory sampled
under a *masked action space* is not a draw from the unmasked policy. Putting it
into the GRPO objective as though it were on-policy is formally wrong; it needs
importance correction. Two uses avoid the problem entirely:

1. **As a discovery signal** — the masked rollout proves the task is reachable,
   so it enters the training set and is trained on **unmasked** with a larger
   group. Fully on-policy, no correction required.
2. **As SFT warm-start data**, which is where [rl_readiness.md](rl_readiness.md)
   already puts self-generated successes, and the same STaR/rationalization
   family as D25's `LLMGTAgent` route. **Decision deferred**: whether to seed
   with SFT at all is open, and this would be one input to it.

Cheaper alternative for the same goal, exactly on-policy: **sample more**. A task
solved 5% of the time yields at least one success 56% of the time at G=16; for
6 tasks that is ~96 rollouts (~$4.30 on Novita) against the cost of building and
validating masking infrastructure.

Revisit when the grid is complete at n=4 — if more all-fail tasks turn out to be
`max_steps`, the idea gets proportionally more valuable.

### Verdict

Two findings, and the second qualifies the first.

1. **The attractor is real and in the model.** Replay is unambiguous: 23% →
   87% → 97% repeat probability as identical results accumulate, unmoved by
   payload shape (30%), prompt guidance (22%) or canonical `top_p` (87%). The
   wire format is clean. It is a self-reinforcing repetition attractor aimed at
   a read that is structurally blind to the state being polled.
2. **But the rate at which trajectories ENTER it is dominated by the serving
   stack** — a 5× spread across three backends on identical weights, with our
   pinned backend the worst of them. So the headline 41.9% pass@1 is
   substantially a NextBit artifact, not a measurement of the model.

Consequences: the baseline and the backend choice both need redoing before any
training claim rests on them; the harness gets a **≥7-consecutive breaker**
(recalibrated — see below) and retry-on-empty-completion; and the model-side fix
stays training, not sampler tricks or repair guards.

**Breaker recalibration.** On the repaired grid (141 scored, 62 passes) passing
traces reach 6 consecutive identical calls, so the earlier ≥5 threshold is no
longer outcome-neutral — it would abort 2 passes. **≥7 aborts none and catches
61 of 79 failures**, identical coverage to ≥5. `taurl.loops.BREAKER_CONSECUTIVE`
is 7. Recalibrate whenever the trace pool grows.

## Reproducing it

`python -m viewer --all && open viewer/out/index.html`, then filter
`pinned_telecom` to failures — duplicate calls are flagged inline in the trace,
and the check panel shows which attempts were never evaluated at all. See
[../viewer/README.md](../viewer/README.md).

---

## The tool-boundary failure: hallucinating the customer's tools

*Found 2026-08-20 while inspecting `novita_telecom_n4/T-ed08#3`. Flagged for
write-up — it reframes the loop diagnosis and is the most interesting single
finding of the investigation.*

**66% of Novita rollouts (103/156) call a tool the agent does not have.** Every
one of them is a **customer device tool**:

| hallucinated tool | rollouts |
|---|---|
| `check_network_status` | 62 |
| `toggle_airplane_mode` | 34 |
| `toggle_data` | 32 |
| `set_network_mode_preference` | 23 |
| `reset_apn_settings` | 23 |
| `can_send_mms` | 14 |

τ2 telecom splits tools in two: the agent gets carrier-side tools
(`get_details_by_id`, `enable_roaming`, …) while the *customer* holds the device
tools. The agent must **ask the customer** to operate their own phone — which is
exactly how real support works. The agent's schema never contains these tools.

### Why the model does it: the manual names them in callable syntax

The agent's tech-support manual — which sits inside its system prompt —
documents the customer's tools by exact name, in second person addressed to the
customer, with call syntax:

> "**check_network_status** — Checks **your** phone's connection status…"
> "You can check if Airplane Mode is ON by using `check_status_bar()` or
> `check_network_status()`. If it is ON, guide the user to use
> `toggle_airplane_mode()` to turn it OFF."

So the prompt contains a mixture of tools the agent may call and tools it must
delegate, with **no syntactic marker separating them**. Discriminating the two
is part of the task, and the model frequently fails at it.

### The finding that reframes everything: 0% on NextBit

The same model on NextBit hallucinated a tool **zero times in 141 rollouts**.
That is not the model behaving differently — it is almost certainly
**constrained tool decoding**: NextBit appears to restrict generation to names
in the supplied schema, making a hallucinated name impossible to emit.

| | NextBit (constrained) | Novita (free-form) |
|---|---|---|
| agent wants a device tool | cannot emit it — must pick a *valid* tool | emits it, receives `Tool not found` |
| what happens next | hammers the nearest legal read (`get_details_by_id`) | reads the error, then asks the customer |
| result | **45% loop rate**, dead rollouts | **7% loop rate**, and **67 of 103 still pass** |

This finally explains the statistic that never fit: on NextBit, **51 of 63 loops
were on tasks whose solution required a customer device action**. The agent was
not confused about state — it was trying to *act on the phone*, could not
express that, and substituted the closest legal call, forever.

**Constrained decoding converted a recoverable mistake into an absorbing loop.**
A guardrail intended to guarantee valid tool calls produced a far worse failure
than letting the model be wrong and read the error.

### Why this matters beyond τ-bench

1. **A serving-side safety feature made behaviour worse, invisibly.** Nothing in
   the trace says "constrained decoding"; it looks like the model is stupid.
2. **The same defect wears different costumes per backend** — hallucination on
   one, looping on the other — so a benchmark number silently measures the
   interaction, not the model.
3. **Error messages are load-bearing.** The model recovers from `Tool not found`
   two-thirds of the time. Denying it the error denies it the recovery.
4. **It is highly learnable.** Recovery already happens at 67/103, so the RL
   gradient is clear — a good argument that this specific defect is RL-shaped.

### Is this a badly designed eval? Partly, yes.

Stated fairly, because the boundary *is* documented. The manual has a clear
header — "**# What the user can do on their device.** Here are the actions a
user is able to take on their device" — so the information is present and a
careful reader could infer it. The design problem is that the document then
undermines its own header, twice:

1. **Perspective whiplash.** Immediately under that header, every tool is
   described in second person addressed to the *customer*: "Checks **your**
   phone's status bar", "Checks **your** phone's connection status". The agent is
   told "these belong to the user" and then reads a page written as though
   speaking to the user.
2. **Direct instruction in the agent's voice.** Later sections say "**You** can
   check if Airplane Mode is ON by using `check_status_bar()` or
   `check_network_status()`" and "guide the user to use `toggle_airplane_mode()`".
   The first clause reads as an instruction to the reader — who is the agent.

The unambiguous ground truth (the tool schema) arrives through a *different
channel* — the API `tools` payload — than the prose, and nothing in the prose
flags which names appear in it. So the model must reconcile a schema it receives
structurally against a document that names non-schema tools in callable syntax
and sometimes addresses it in the second person.

**Why this is a measurement problem and not just a prose nit:**

- **It is a dominant score driver.** 66% of rollouts hit it. A benchmark
  intended to measure telecom troubleshooting is substantially measuring whether
  the model tracks a convention that the prompt actively muddles.
- **Its severity is decided by the serving stack, not the model.** Free-form
  decoding makes it a recoverable error (67 of 103 such rollouts still pass);
  constrained decoding makes it an absorbing loop (45% loop rate, near-total
  failure). A ~22-point swing in the headline number therefore turns on an
  interaction between a prompt-design choice and an undocumented backend
  behaviour. That is close to a validity failure: the score is not a property of
  the model.
- **It is silent.** Nothing in a trace says "the model was blocked from the
  action it wanted". On the constrained stack it simply looks incompetent.

The charitable reading is that the discrimination is intentional and part of the
task — real agents do work from documentation describing things they cannot do
themselves. That is defensible. But if it is intentional, it is *the* dominant
difficulty of the domain and should be stated as such, rather than emerging from
inconsistent second-person phrasing.

### Experiment: does clarifying the boundary fix it? (2026-08-20)

**Deviation tested, not adopted.** A single paragraph appended to the domain
policy, adding no telecom knowledge whatsoever -- only who owns which tools:

> IMPORTANT: You can only call the tools that appear in your own tool list. The
> device actions described in the technical support manual ... run ON THE
> CUSTOMER'S PHONE. You cannot call them and they are not available to you.
> Whenever the manual says to check or change something on the device, you must
> ask the customer to do it themselves and rely on what they report back.

Run on the **10 worst-affected tasks** from the Novita grid (those where nearly
every rollout hallucinated), 4 trials each, Novita, sampling pinned.
`tau2probe/ablate_tool.py --mode clarify`.

| | baseline | **+ tool_boundary** |
|---|---|---|
| rollouts that called a non-existent tool | **37/38 (97%)** | **4/35 (11%)** |
| pass rate | 25/38 (66%) | 19/35 (54%) |
| loop rate | 0% | 0% |
| avg messages/rollout | 76 | **61** |

**The clarification nearly eliminates the hallucination -- and does not improve
task success.** The 11.5-point pass-rate gap is the wrong way round and is not
significant (z=1.00, p=0.32); with ~36 rollouts per arm this experiment cannot
resolve differences below roughly 20 points.

**What that tells us, and it is not what we expected.** On a free-form backend
the hallucination is *largely harmless*: the model reads `Tool not found`,
adapts, and asks the customer. Removing the behaviour makes traces ~20% shorter
(76 -> 61 messages) without making them more successful. So on Novita, the
tool-boundary confusion is a **cost problem, not a capability problem**.

That reverses the natural reading of the earlier NextBit finding. The 22-point
gap between backends is not "hallucination destroys performance"; it is
"**constrained decoding** destroys performance, by denying the model the error
message it would otherwise recover from". The prompt ambiguity is the trigger;
the serving-side guardrail is what makes it fatal.

**Consequence for the eval-design critique above:** it stands, but narrowed.
The prompt genuinely induces a 97% error rate, and that is a real design defect
costing ~20% of every trace's length. It does *not* by itself distort the score
on a well-behaved backend. It distorts the score catastrophically on a
constrained one -- which is a statement about the interaction, and about how
little a benchmark number means without the serving stack attached.

**Decision: not adopted.** It buys no measurable capability, and it would cost
comparability with every published number. Re-test if we ever move to a backend
with constrained tool decoding, where the calculus is completely different.

### What we are not doing

Not patching the manual. Rewriting the domain policy to separate the tool lists
would likely fix most of it, but it is a benchmark change that breaks
comparability with every number we and everyone else has. The discrimination is
part of the task; the model should learn it.

### Open

- Confirm the constrained-decoding hypothesis directly (self-hosted vLLM with
  and without a tool-name grammar, same weights, same prompts).
- Does the loop rate on NextBit fall to Novita's level if the agent is *given*
  the device tools? Would isolate substitution-under-constraint from polling.
- Tag `hallucinated_tool_calls` and `too_many_errors` as first-class metrics.
