# Design doc v1 — SFT-then-GRPO on τ2 telecom (tentative)

*2026-08-21. First integrated design after the serving-stack and tool-boundary
investigations. Supersedes the staging in [rl_run_plan.md](rl_run_plan.md) where
they conflict; hosting, cost model and kill criteria there still apply.*

---

## 1. Settled parameters

Every knob below was either measured in this project or is Gemma's own shipped
default. Nothing is inherited silently.

| knob | value | why |
|---|---|---|
| policy | `google/gemma-4-26b-a4b-it` (26B MoE, ~4B active) | the model under test; `-it` is non-negotiable for tool calling |
| serving backend | **Novita** (bf16), pinned, `require_parameters` | D30: NextBit loops 32% vs Novita 6% on identical tasks; empty completions 19/160 vs ~0 |
| sampling | `temperature 1.0, top_p 0.95, top_k 64` — sent explicitly | Gemma's `generation_config.json`; before D30 these fell back to undocumented host defaults |
| customer simulator | `gemini-3.7-flash` @ temp 0, served direct (Gemini API) | D6 pins the model; direct = same Google serving minus proxy (all 6 OpenRouter endpoints are Google), 0.9s vs 1.8s, no global RPM cap |
| environment / split | **standard τ2 telecom, shipped splits: `train` 74 / `test` 40** | mms is 1,984 unique fault combinations from 13 atoms — varied enough that the standard split machinery holds; keeps every number comparable to published τ2 work |
| secondary eval | frozen 20-task mms pool sample (`results/_snapshots/mms_eval_extra.json`), n=4 baseline done: pass@1 23.8%, pass@4 65.0% | the test split over-weights easy archetypes ~19×; this is the pool-representative check |
| step cap | 140 | above every observed success (longest: 136 msgs) |
| rollout hygiene | loop breaker ≥9 *unproductive* consecutive identical agent calls, terminate-and-**evaluate** (not score-0); retry-on-empty-completion | breaker costs ~1% of successes for ~9% of tokens — training rollouts only, never eval; empty completions poisoned 12% of a NextBit grid |
| throughput | multi-route pools (`tau2probe/routes.py`): agent OpenRouter→Novita + Vercel→novita; customer direct + OpenRouter + Vercel; concurrency ~24 | 15.4× effective parallelism vs 4.2× single-route; every route in a pool is the same backend, so failover is a route change, not a stack change |
| metrics | pass@k (macro over tasks) headline; pass^k reported alongside; loop rate and hallucination rate on every run | micro/macro differ when trial counts are uneven, and macro is what the paired test uses |
| eval protocol | paired per-task deltas, same task ids / trials / stack / sampling / prompt in both arms; 40×8 gives 89% power for +10 pts | [eval_protocol.md](eval_protocol.md) |

### Set accounting — frozen 2026-08-21, disjointness asserted in code

| set | tasks | may be trained on? | touched when |
|---|---|---|---|
| `test` (shipped) | 40 | **never** | twice: before and after. Not during. |
| mms pool eval (`mms_eval_extra.json`) | 20 | **never** | twice: before and after |
| `small` (shipped) | 20 | never | smoke tests only, never for numbers |
| **monitor** (`monitor_tasks.json`, new) | 24 | **never** | freely during training: loss curves, checkpoint selection, reward-hacking checks |
| **trainable** (everything else, incl. the 74-task train split) | **2,181** | yes | SFT data gen, GRPO curriculum |

**On validation sets:** the classic early-stopping validation split has indeed
mostly disappeared from LLM fine-tuning — with a tiny LoRA and 1–2 epochs,
overfitting-in-the-classical-sense is not the live risk. But two of its jobs
survive under different names, and both need a set that is *neither* trained on
*nor* the test set:

1. **Checkpoint selection.** Picking the best of 30 GRPO checkpoints by test-set
   score is selection on the test set — the reported delta becomes biased
   upward. Checkpoints are picked on the monitor set; the test set is touched
   exactly twice.
2. **Reward-hacking surveillance during RL.** Training reward can rise while
   true capability does not (shaped-reward exploits, simulator quirks). The
   monitor set, evaluated every few steps with the *unshaped* binary reward, is
   the tripwire.

The monitor set is 24 tasks, stratified from the pool (so it is mms-heavy like
the pool, 22/24), and its per-task results never inform anything except *when
to stop and which checkpoint to keep*.

**Known caveat carried, not hidden:** the shipped `test` split is 38%
`service_issue` (2% of pool, ~87% pass) — it will overstate absolute capability.
That is why the mms pool sample rides along as the second eval. Both are frozen.

## 2. Pass 1 — generate SFT data with the prompt modification

The task supply is much larger than the 74-task train split: telecom ships
**2,285 tasks**, and after holding out `test` (40), the frozen mms eval sample
(20) and `small` (20), **~2,130 remain free for training data**. Every one is a
unique fault combination, so more tasks = more diversity, not repetition.

| tier | tasks × trials | rollouts | est. cost | est. successful traces |
|---|---|---|---|---|
| train split | 74 × 6 | 444 | ~$20 | ~250–290 (@ ~60% pass) |
| + pool draw, stratified by fault count | 326 × 4 | 1,304 | ~$59 | ~330–460 (@ 25–35% pass at pool difficulty) |
| **Pass 1 total** | **400 tasks** | **~1,750** | **~$79** | **~600–750 traces ≈ 0.8–0.9M target tokens** |

That takes the SFT set from "a nudge" to a respectable rejection-sampling
corpus, and — because the pool is harder — its successes are concentrated
exactly on the behaviours we want cloned: multi-fault diagnosis that the model
only sometimes pulls off. Scale further ($27 per +100 tasks × 4) only if the
SFT gate suggests the set is still thin.

Procedure: run everything with the *clarified* prompt (`--mode clarify`, the
one-line boundary statement) on the pinned stack, concurrency ~24 with the
multi-route pools. **Freeze the drawn pool task list to
`results/_snapshots/sft_pool_tasks.json` before running** — it is now a
training-set definition, and the split guard must know about it.

Why the clarified prompt for *generation*: it produces zero loops, 15–20%
shorter traces, and rescued 3 of 5 previously-unsolved test tasks — i.e. it
surfaces successful trajectories the shipped prompt rarely samples, in the
**unmodified action space** (unlike tool masking, which changes the action
space and taints the data).

Collect every **successful** trace. Expected yield at train-split difficulty:
roughly 55–65% of rollouts ≈ 250–290 traces, including — if the test-split
pattern holds — first-ever successes on several currently-unsolved tasks.

**Hard rule: SFT data comes from the `train` split (plus pool draws) only.**
The 3 rescued traces we already have are `test`-split tasks; they are evidence
the technique works, and they must never enter training. The split guard stays.

Also mine the same run for **efficiency pairs** (same task, two successes, ≥3
duplicate-call gap) — free DPO data for later, tagged by `taurl/loops.py`.

## 3. Pass 2 — SFT, done correctly

- **Objective:** cross-entropy on **agent tokens only**. Customer turns and tool
  results are masked out of the loss — they are context, not targets
  ([simulator_distribution.md](simulator_distribution.md)). verl ships the
  multiturn tokenization/masking machinery.
- **Template fidelity — exact tool-call formatting is a hard requirement.**
  Our saved traces hold tool calls as parsed JSON (`name` + `arguments` dict);
  the model never saw that. It emitted Gemma's native serialization, which the
  backend parsed. Training must invert this exactly:
  1. Re-render every trace through **Gemma's own chat template** (the
     tokenizer's, not a hand-rolled one): tool calls back to native
     `<|tool_call>`-style syntax, tool results into the template's tool-response
     turn, argument JSON serialized with the same key order and spacing the
     model produces.
  2. **Round-trip test:** template-render → parse back → must equal the
     original call. Any trace that fails is dropped, not patched.
  3. **Mask-alignment unit test:** the agent/context loss boundaries must land
     exactly on template token boundaries. Off-by-one here silently trains on
     fragments of tool results — the failure is invisible in loss curves and
     shows up only as hallucinated tool results at serve time. A fixed known
     trace with hand-verified token spans goes in `tests/`.
  4. The same rendering path is reused verbatim for GRPO's rollout
     tokenization, so train-time and RL-time token streams cannot drift.
  B7 established the inversion is possible from our saved JSON; this section
  makes it a tested contract rather than an intention.
- **Prompt at train time = the SHIPPED prompt**, even though the data was
  generated under the clarified one. The trajectories contain no reference to
  the boundary text, and training must condition on what evaluation will
  condition on. (Deviation risk logged: if SFT underperforms, re-test with the
  clarified prompt on both sides.)
- **Adapter:** attention-only LoRA r=16 (~10.8M params), per
  [moe_finetuning.md](moe_finetuning.md) — expert LoRA is ruled out by top-8-
  of-128 routing giving each expert ~6% of tokens.
- **Schedule:** 1–2 epochs, lr 1e-5 → cosine to 0, batch ~64k tokens. Small on
  purpose: this is a warm start, not the experiment.
- **Gate before proceeding:** re-run the 40×4 test-split eval on the SFT'd
  model, shipped prompt. Expect: loop rate ≈ 0–2% (the data has none), pass@1
  at least baseline (64%), ideally +2–5. If pass@1 *drops*, stop — the
  warm start is hurting and GRPO starts from base instead.

## 4. Pass 3 — GRPO on the SFT'd model

**Frontier selection first** (Stage B, unchanged): n=8 scan on the train split
under the *shipped* prompt with the SFT'd model — the frontier moves after SFT,
so it must be re-measured, not reused. Select tasks with pass@1 ∈ [0.25, 0.75];
supplement from the mms pool (stratified by fault count, 4–6 peak) if fewer
than ~32 tasks qualify.

Suggested starting hyperparameters (verl, colocated vLLM, RunPod H200):

| param | value | rationale |
|---|---|---|
| group size G | **8** | smallest G with usable advantage variance; loop-free traces (SFT + breaker) keep KV pressure low enough that 16 is the fallback if groups come back degenerate |
| tasks/step | 16, sampled from frontier, stratified by scenario and fault count | D18's stratification lesson |
| steps | 30 (first run), checkpoint every step | matches the costed plan; extend only if the curve is still moving |
| rollout sampling | temp 1.0, top_p 0.95, top_k 64 — identical to eval | the baseline was measured at these; divergence here contaminates the delta |
| advantage | group-normalised: (r − mean(r_G)) / std(r_G), whole-trajectory | standard GRPO; binary reward makes per-token credit meaningless anyway |
| loss masking | agent tokens only (same mask as SFT) | tool results and customer turns must carry no gradient |
| KL to reference | **β ≈ 1e-3** against the SFT'd policy | main brake on simulator-idiosyncrasy drift (D26); loosen to 3e-4 only if reward plateaus early |
| clipping | PPO-style ε = 0.2 | verl default; no reason measured to deviate |
| actor lr | 1e-6 (LoRA params) | conservative for an on-policy update on 10.8M params |
| reward | **Tier-0 binary only** for the first run | tiered bonus (λ·duplicates, μ·excess turns) is staged for run 2; one variable at a time |
| rollout guards | loop breaker ≥9 unproductive + retry-on-empty; **no tool masking, no prompt clarification, no repair guards** | GRPO must run in the exact action space and prompt that evaluation uses |
| gate (Stage C unchanged) | overfit 4 memorised tasks first | if it cannot, the loop is broken; do not scale |

**Evaluation (Stage E, unchanged in design):** paired 40×8 on the test split +
20×8 on the mms pool sample, shipped prompt, same stack; per-task deltas with a
CI. Then τ2 airline and τ1 retail for transfer, and the simulator-transfer
check (different customer model, same tasks).

## 5. Review — is this sound, and will it work?

**Techniques in use, named:**

| technique | where |
|---|---|
| pinned serving stack + explicit sampling | all runs (D13→D30) |
| paired within-model eval, per-task deltas | Stage E ([eval_protocol.md](eval_protocol.md)) |
| pass@k / pass^k separation, macro averaging | metrics throughout |
| capability-frontier curriculum (pass@1 ∈ [0.25,0.75]) | task selection |
| stratified sampling (scenario × fault count) | curriculum + eval supplement |
| **rejection-sampling SFT warm start** (train on own successes — STaR/RFT family) | Pass 1–2 |
| **prompt-assisted trajectory discovery** (clarified prompt to surface rare successes, standard prompt to train) | Pass 1 — the novel-ish bit; same family as easy-to-hard data collection |
| loss masking of non-policy tokens | SFT + GRPO |
| LoRA (attention-only, MoE-aware) | SFT + GRPO |
| GRPO: group-relative advantage, PPO clipping | Pass 3 |
| KL regularisation to reference | Pass 3 |
| episode truncation on absorbing failure (loop breaker) | rollouts only |
| action masking / tool ablation | **data-diagnosis only, never training** |
| tiered lexicographic reward (Tier-0 gate; Tier-1 shaped bonus) | staged for run 2; DPO on efficiency pairs staged for run 3 |
| persona diversity in the training mix | rollout sampling |
| held-out simulator transfer check | Stage E2 |
| cross-domain transfer eval (τ2 airline, τ1 retail) | Stage E |
| kill criteria, pre-registered | [rl_run_plan.md](rl_run_plan.md) |

**Best-practice check — honest gaps:**

1. **SFT before on-policy RL** is the standard ordering (it concentrates
   probability mass on formats/behaviours GRPO can then refine), and rejection
   sampling on own outputs avoids the imitation-of-foreign-style problem (D27).
   Sound.
2. **The biggest known risk is unchanged:** verl ↔ multi-turn AgentGymEnv
   integration. No design choice above removes it; Stage C's overfit-4-tasks
   gate is the control.
3. **Distribution shift, stated:** SFT data is generated under a different
   *prompt* than training conditions on. Mitigated (trajectories contain no
   boundary text) but not eliminated; the SFT gate catches it if it bites.
4. **Ceiling risk on the test split:** at 64% baseline with 13/40 tasks
   saturated, headline headroom is limited — the mms pool eval (23.8%) is where
   improvement has room to show. Expect the delta to be larger there.
5. **What we deliberately did NOT do:** repair guards, reward shaping in run 1,
   training through the clarified prompt, tool masking in RL — each would have
   manufactured improvement the eval could not attribute to the model.

**Do we expect improvement?** Yes, with bounded confidence:

- **SFT alone:** +2–6 pts on pass@1 is typical for rejection-sampling warm
  starts at this scale; loop rate should drop to ~0 since the data contains no
  loops. This alone would be a presentable result.
- **GRPO:** the preconditions are unusually good — 55–65% of tasks in the
  frontier band after re-scan, mixed groups nearly everywhere on the mms pool
  (pass^4 = 0%), objective verifier, on-policy loop. +5–15 pts on the trained
  distribution (mms-heavy) is the realistic band; the 40×8 paired design has
  89% power at +10.
- **What would count as failure and still be publishable:** improvement on
  telecom that does not transfer to airline/retail (narrow-overfit finding), or
  a gain that evaporates under a different customer simulator (simulator-
  overfit finding). Both are pre-registered outcomes, not embarrassments.

**Cost of the full pipeline as designed:** ~$79 (SFT data, 400 tasks) + ~$12
(SFT compute) + ~$8 (SFT gate eval) + ~$25 (frontier rescan) + ~$85 (GRPO run)
+ ~$60 (paired evals + transfer) ≈ **$270 end to end**, dominated as ever by
the customer simulator.
