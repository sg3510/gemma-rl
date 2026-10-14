# RL notes: things we got wrong or had to think through

Methodological findings from this project, including corrections to claims made
earlier in it.

## 2026-09-12 — OpenRouter article and current GRPO readiness

**Follow-up prompt probe:** [clarify_zero8_probe.md](clarify_zero8_probe.md).
Three historical 0/4 MMS tasks were re-tested. One reached 3/8 with the original
prompt alone. The other two gave 0/6 → 1/8 and 0/7 → 0/8 under clarification;
three baseline episodes remained unfinished when stopped after ~30.5 minutes.
No observed frontier conversion (requires 2–6/8); one held-out success was
discovered. These are not confirmed 0/8 baselines, and the missing episodes
must not be counted as failures. Clarified arms had no detected loops.

I measured the backend effect on 2026-08-18–20 (D13, D30,
[provider_loop_rates.md](provider_loop_rates.md)). On 2026-09-07 Mo Moustafa
published [So you want to use OpenRouter?](https://mmoustafa.com/blog/so-you-want-to-use-openrouter/)
([Hacker News discussion](https://news.ycombinator.com/item?id=49621546)),
independently reporting the same pattern: substantial same-model differences
across providers, provider-specific tool parsing and empty completions, and
declared quantization as a poor predictor of quality. It corroborates the
backend investigation; it does not independently establish the cause of our
Gemma failures.

**Working approach:** use one pinned OpenRouter → Novita agent endpoint, no
provider fallbacks, with explicit sampling (temperature 1.0, top_p 0.95,
top_k 64). Keep the customer model and serving path fixed too. Multi-route
infrastructure exists, but route equivalence remains unproven; it need not be
on the critical path. Treat bf16 as provider-reported metadata, not proof of
the entire inference stack's precision or behavior. Defer further provider /
quantization investigation until controlled self-hosted comparisons can isolate
precision, template, parser and decoding constraints.

**Evals:** Novita telecom test is complete at 40×4, pass@1 64.4%, ~7% loops,
55% frontier; the frozen MMS pool sample is 20×4, pass@1 23.8%, pass@4 65%,
65% frontier and no tasks solved 4/4. The superseded NextBit full grid scored
41.9% with ~45% loops and 19/160 infrastructure losses. The backend switch
also pinned previously omitted sampling parameters, so the full-grid delta
does not isolate a single serving variable. Prompt clarification removed
loops without a demonstrated success-rate gain. Constrained decoding remains
a hypothesis, pending a controlled test.

**Ready to build and smoke-test GRPO, not to launch the full training run.**
The objective verifier, eval/trace tooling and evidence of mixed-reward tasks
exist. Still needed: trainable GPU weights and serving, a same-stack baseline,
multi-turn trainer integration with correct tokenization and agent-only loss
masks, policy-weight synchronization, and a four-task overfit gate. Saved API
traces are analysis / possible SFT material, not fresh on-policy GRPO batches.
Re-scan the training frontier at n=8 under whichever policy actually starts RL.

The latest tentative design is [design_doc_v1.md](design_doc_v1.md): optional
SFT warm start followed by binary-reward GRPO. Older readiness/run-plan and
protocol sections contain superseded settings. In particular, the latest
breaker proposal is ≥9 unproductive repeated calls, terminate-and-evaluate,
training only. Resolve the design's SFT gate on `test` versus its held-out-set
rule by using the monitor set for gate/checkpoint decisions. Final evaluation
should pair identical stacks and task IDs (40×8 test plus 20×8 MMS), with
per-task delta confidence intervals and separate transfer checks. Historical
power and cost estimates must be recalculated for the actual run setup.

---

## 1. "Expert traces are out of distribution" is the wrong argument

**What was claimed** (and is wrong): human-written expert trajectories cannot be
used because their phrasing is out of distribution for the model.

**Why it is wrong:** the model was pretrained on human text and instruction-tuned
on human-written targets. Producing human-like text is the thing it is *best*
at. Every SFT pipeline in existence trains on human-written completions. And in
our own setting, every trajectory already contains a non-Gemma interlocutor —
the customer simulator is Gemini. If "text from another distribution" were
disqualifying, the whole benchmark would be disqualified.

**What "in distribution" actually means in RL.** Not text style — the *policy's
action distribution given a state*, `π(a|s)`. The relevant question about an
expert trajectory is not "does this read like the model" but "what probability
does the current policy assign to these actions in these states".

That reframing produces the real constraints, and they are different in kind:

### 1a. The on-policy constraint is mathematical, not aesthetic

GRPO and PPO optimise a clipped importance-weighted objective. Each sampled
action carries a ratio

```
ρ = π_θ(a|s) / π_θ_old(a|s)
```

For a trajectory the policy never generated, `π_θ_old` is not merely small —
it is **undefined**. The trajectory was drawn from some other process (a human,
a stronger model, a ground-truth script), so there is no behaviour policy to
divide by. Substituting the current policy makes ρ ≡ 1 and silently converts the
objective into something else.

This is why you cannot concatenate expert data into an on-policy batch. It is an
estimator-validity problem, and it has nothing to do with how the text reads.

### 1b. Even with a valid ratio, clipping starves the gradient

Suppose you do have a behaviour policy — say an earlier checkpoint. If the expert
action has low probability under the current policy, ρ is large, clipping bites,
and the sample contributes almost nothing to the update. Off-policy data that is
*far* off-policy is not dangerous so much as **inert**.

**This is exactly why the standard recipe is SFT first, then RL.** SFT is
off-policy by construction and has no ratio to be invalid: it just raises
`π(expert action | state)` by maximum likelihood. Once those actions have
non-trivial probability, RL can actually reinforce them. The two stages are not
redundant — the first makes the second able to see the behaviour at all.

### 1c. The genuine risk with demonstrations is capability mismatch, not style

Train on demonstrations of reasoning the model cannot perform and it learns the
**surface form of correct answers without the underlying computation**. This is
well documented for imitation of stronger models, and it applies to *any*
stronger demonstrator — human, frontier model, or ground-truth script. The
failure mode is confident, well-formatted, wrong.

So the correct caveat about hand-written traces is not "the phrasing is alien".
It is:

1. they cannot enter an on-policy objective (1a);
2. if forced in off-policy, clipping makes them inert (1b);
3. SFT on them risks teaching form without substance (1c).

**Verdict for this project:** demonstrations remain useful, via SFT warm-start or
expert iteration, and the preferred source is still `LLMGTAgent` pointed at Gemma
— not because human phrasing is unusable, but because a model-generated
trajectory already has high probability under the policy, which addresses (1b)
directly.

---

## 2. What the customer simulator does and does not bake in

Covered in full in [simulator_distribution.md](simulator_distribution.md). The
short version, since it is adjacent to the above:

Only the **agent's** tokens carry loss; customer turns are masked as context. So
the simulator's phrasing is conditioned on but never produced. Masking protects
the *targets*, not the *inputs*, so the policy is still optimised **conditioned
on** one simulator's style — which scopes the claim rather than invalidating it.

Note how this differs from §1: the customer-simulator issue really is a
distribution question. The expert-trajectory issue is not — it is an estimator
question wearing a distribution costume.

---

## 3. RL here is likely sharpening, not teaching

| environment | pass@1 | pass@k | headroom |
|---|---|---|---|
| τ2 telecom | 52.9% | 83.7% (k=3) | **+30.8 pts** |
| τ1 retail | 65.9% | 88.7% (k=4) | +22.8 pts |

pass@k is what the model achieves given k attempts — capability it already has
but cannot deploy reliably. If RL mostly concentrates probability mass on what
the base model can already do, that gap is the realistic ceiling, and the job is
converting *can sometimes* into *does reliably*.

Which is not a disappointing result. It is why τ-bench's headline metric is
**pass^k** (all k attempts succeed) rather than pass@k: nobody ships an agent
that works one time in three.

---

## 4. Corrections made during this project

Kept deliberately, because the reasoning is more useful than a clean-looking
record.

| claim | correction |
|---|---|
| "fp8 beats bf16 by 31 points" | **Retracted.** Two backends could not serve tools at all and failed 100% of calls; that attrition moved both group means. Isolated per backend, all four working ones agree at 70–80%. |
| "telecom pass@1 is 35.0%" | **47.1%.** Our τ2 reader counted infrastructure errors as task failures — the same bug we had found in τ-bench's runner and fixed for τ1, then reproduced in our own code. |
| "the eval is underpowered, needs ~$97" | Used an *unpaired* test. The design is paired on the same tasks; 40 × 8 gives 89% power for +10 pts at ~$49. |
| "LoRA on a MoE is the highest risk" | Downgraded. vLLM ships `lora/layers/fused_moe.py`; the rating came from a general intuition rather than from this architecture. |
| "expert traces are out of distribution" | **Wrong argument.** See §1. The real constraints are estimator validity and clipping. |
