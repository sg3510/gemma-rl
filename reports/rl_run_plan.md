# RL run plan

A staged plan to run GRPO on Gemma 4 26B A4B against τ2 telecom, on rented
GPUs. **Primary host: RunPod** (see host selection below); Modal kept for
short non-interactive jobs against its free credit. Every claim below was checked rather than assumed; the verification
column says how.

## Verified before planning

| assumption | status | how checked |
|---|---|---|
| vLLM can parse Gemma 4 tool calls | **yes** | `vllm/tool_parsers/gemma4_engine_tool_parser.py` + `gemma4_utils.py` exist upstream |
| SGLang can too | **yes** | `python/sglang/srt/function_call/gemma4_detector.py` |
| verl supports multi-turn tool-calling RL | **yes** | `examples/tutorial/agent_loop_get_started/`, plus documented SGLang multi-turn rollout |
| τ2 exposes an RL-shaped interface | **yes** | `tau2.gym.AgentGymEnv`, Gymnasium-compatible, runs customer + tools + reward |
| RunPod has the hardware, in stock | **yes** | queried live: **H200 141GB $3.59/hr, stock Medium**; H100 80GB $2.69/hr, Medium |
| Modal has it too | **yes** | H100 $3.95/hr (may upgrade to H200 free), H200 $4.54/hr |
| Modal can run a long job | **24 h max per call** | documented; longer runs need checkpoint + retry, which we want anyway |
| Modal free credits | **$30/mo** (Starter) | ≈ 7.6 H100-hours — enough for Stage A |

## Host selection: RunPod primary

The decision axis is **serverless vs. a persistent box**, and it turns on one
fact: 50 GB of weights must load before anything runs.

- **Modal does not bill for thinking.** Between runs you pay nothing. But every
  invocation reloads the model unless you configure memory snapshotting.
- **A pod keeps the model resident.** You iterate against a warm vLLM in seconds,
  but you pay while reading logs and editing code.

Stage C is explicitly *"wire the loop and prove it overfits 4 tasks"* — many
short debug cycles against a loaded model. That is pod-shaped work.

| option | $/hr | 1.2 h run | +8 h dev |
|---|---|---|---|
| **RunPod H200 141G** | **$3.59** | $4.31 | $28.72 |
| RunPod H100 80G | $2.69 | $3.23 | $21.52 |
| Modal H100 (→H200) | $3.95 | $4.74 | $31.60 (upper bound; idle is free) |

**Chosen: RunPod H200 at $3.59/hr.** 21% under Modal's H200, in stock, per-second
billing, network volumes for checkpoints, and a plain SSH box so verl and vLLM
install exactly as documented — which matters because the verl agent-loop
integration is already the highest-risk piece.

**Why not the cheaper H100 at $2.69.** 80 GB leaves ~30 GB after bf16 weights for
activations, KV cache and LoRA. Rollout concurrency is our main cost lever, and
concurrency is KV-cache-bound, so the 141 GB card likely finishes *sooner* and
costs less in total despite the higher hourly rate. To be confirmed in Stage A.

**Rejected:** Vast.ai (cheapest at ~$1.49, but interruptible instances against a
run holding 50 GB for hours is a false economy); Together's fine-tuning API
(SFT/DPO only — an online RL reward needs the environment in the loop, which no
managed fine-tuning service can express); hyperscalers (IAM, quotas, VPC,
image-building for no benefit at this scale).

**Modal retained** for Stage A, which is short and non-interactive — a good use
of the $30/month free credit.

## Architecture

```
RunPod pod (1x H200 141GB)
├── vLLM serving the CURRENT policy
│     --enable-auto-tool-choice --tool-call-parser gemma4  (bf16)
├── rollout workers  ── N concurrent episodes
│     └── tau2 AgentGymEnv  ── customer sim (Gemini 3.7 Flash, external API)
│                            └── tools + DB + reward
└── GRPO trainer (LoRA)  → weights pushed back to vLLM each step
                          → checkpoints to a network volume
```

The customer simulator stays an external API call. That is the one component we
deliberately do not move, because it is pinned (D6) and changing it would break
comparability with every baseline we have.

## Costs

Measured inputs: ~12 customer calls per rollout at ~1.5 s each; ~15 agent turns
at ~1.2 s each with local vLLM (3.8B active params). So ~36 s per rollout, of
which **half is the GPU idling on customer-API latency**.

| concurrency | 30 steps (G=8, 16 tasks/step) | GPU |
|---|---|---|
| 16 | 2.4 hr | $9.48 |
| **32** | **1.2 hr** | **$4.74** |
| 64 | 0.6 hr | $2.37 |

Rollout concurrency is the main cost lever: at low concurrency you are renting
an idle H200.

| line | cost |
|---|---|
| GPU, 30 steps @ concurrency 32 (RunPod H200) | $4.31 |
| customer-sim API, 3,840 rollouts | $30.72 |
| eval before + after (40 × 8, paired, 89% power) | $49.00 |
| **first full run** | **~$84** |

The GPU is the cheapest line. Budget attention accordingly.

## Stages

**Stage A — serve and re-baseline.** *(~$10, half a day)*
Stand up vLLM on Modal with the gemma4 tool parser, bf16. Re-run the τ2 telecom
baseline against it. **This becomes the real "before".** If the "before" came
from a rented API and the "after" from our own server, part of the delta is a
change of server — the entire lesson of D13/D14. Gate: the new baseline should
land near the pinned pass@1, show no `malformed_tool_args`, **and show a
tool-call loop rate comparable to the pinned stack's** (measure with
`python -m taurl.loops`) — if the loop rate moves materially between serving
stacks, part of the baseline was a serving artifact, and that has to be settled
before any training delta can be attributed to the model. See
[tool_call_looping.md](tool_call_looping.md).

**Stage B — frontier scan on the train split.** *(~$25)*
n=8 over the 74-task telecom train split, **stratified by scenario** — the full
pool is 87% `mms_issue` at 9% success, so uniform sampling yields degenerate
groups (D18). Output: the task list with per-task pass@1 in [0.25, 0.75].

**Stage A2 — rollout hygiene.** *(free, before any training)*
Two harness fixes that are prerequisites, not niceties:
**(a) loop breaker** — terminate at 7 consecutive identical tool calls, score 0
(`tau2probe/loop_breaker.py`); saves 12–55% of tokens and stops long looping
trajectories from capping group size.
**(b) retry-on-empty-completion** — some backends return a completely empty
assistant message, which tau2 turns into a dead rollout; on NextBit this killed
19 of 160 cells. During GRPO an empty completion mid-rollout would poison a
whole group.

**Stage B2 — Tier-1 quality metrics.** *(free)*
Pure analysis of traces we already hold. Implement exact-duplicate-call and
excess-turn detection, and report the quality spread across successes. Feeds a
small shaped bonus *among* successes in Stage D, and the comparison UI in
Stage F. See [tiered_reward.md](tiered_reward.md).

**Stage C — wire the loop and prove it on 4 tasks.** *(~$5)*
verl GRPO + `AgentGymEnv`. Do not train anything yet; verify one full step:
rollouts collected, rewards attached, advantages computed group-relative, tool
tokens masked out of the loss, LoRA weights reaching vLLM. Gate: reward on 4
memorised tasks goes up. If it cannot overfit 4 tasks, the loop is wrong.

**Stage D — the real run.** *(~$85)*
30 steps, G=8, 16 tasks/step from the frontier set. Checkpoint to a Volume every
step so the 24-hour cap and any preemption are survivable.

**Stage E2 — simulator-transfer check.** *(~$10)*
Re-evaluate the trained policy against a **different** customer model on the same
tasks. If the gain survives it reflects telecom competence; if it evaporates we
optimised for one simulator. See [simulator_distribution.md](simulator_distribution.md).

**Stage E — evaluate, paired.** *(~$49, already in the total)*
Same task ids, trial counts, customer, temperature, backend and step cap as
Stage A. Test on **per-task deltas**, report a confidence interval on the delta
(D20). Then evaluate on τ2 airline and τ1 retail for transfer — telecom has only
three scenario archetypes, so a telecom-only gain is a weak claim.

**Stage F — human preference, optional.** *(cost = labelling time)*
Trace-comparison UI, Tier-2 labels (politeness, question quality) on pairs the
trained policy generates, then DPO or a preference model against the
Tier-0-only baseline. This is the step that makes the project RLHF rather than
RLVR. Measured first: Tier 1 auto-ranks ~95% of same-task comparisons, so human
effort goes to dimensions Tier 1 cannot see, not to arbitrating efficiency.

## Risks, honestly

| risk | severity | mitigation |
|---|---|---|
| **Adapter placement on a MoE** — *downgraded from high after checking; see [moe_finetuning.md](moe_finetuning.md)* | medium | vLLM ships `lora/layers/fused_moe.py`, so serving works. Plan of record is **attention-only LoRA r=16** (10.8M params, 100% gradient signal). Expert LoRA is ruled out: top-8-of-128 routing gives each expert ~6% of tokens. Settled empirically in Stage C. |
| **verl ↔ AgentGymEnv integration** — verl's agent loop expects its own interface, not Gymnasium | high | budget real time here; it is the main engineering task |
| **GPU idle on customer-API latency** (50% of wall-clock) | medium | high rollout concurrency; consider batching episodes |
| **pod interruption / 24 h Modal cap** | low | checkpoint to a network volume every step |
| **Cold start** loading 26B of weights | low | persistent pod keeps it resident; Modal needs memory snapshotting |
| **Reward hacking** — the agent finds a DB state that hashes correct without doing the task | medium | inspect trajectories of newly-successful tasks by hand |

## Kill criteria

State these now, so we do not rationalise later.

- Stage C cannot overfit 4 tasks → the loop is broken; stop and fix, do not scale.
- Stage A baseline lands far from 35% → the serving stack is not equivalent;
  stop and reconcile before spending on training.
- Stage E delta is positive but its confidence interval includes zero → report it
  as **inconclusive**. Our evaluation has 89% power for +10 points and 39% for
  +5. If the true effect is small, this design cannot prove it, and we say so.
- Improvement on telecom with none on airline or retail → report as
  **narrow/overfit**, which is a legitimate finding, not a failure.
