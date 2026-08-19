# Modal RL plan

A staged plan to run GRPO on Gemma 4 26B A4B against τ2 telecom, on rented
Modal GPUs. Every claim below was checked rather than assumed; the verification
column says how.

## Verified before planning

| assumption | status | how checked |
|---|---|---|
| vLLM can parse Gemma 4 tool calls | **yes** | `vllm/tool_parsers/gemma4_engine_tool_parser.py` + `gemma4_utils.py` exist upstream |
| SGLang can too | **yes** | `python/sglang/srt/function_call/gemma4_detector.py` |
| verl supports multi-turn tool-calling RL | **yes** | `examples/tutorial/agent_loop_get_started/`, plus documented SGLang multi-turn rollout |
| τ2 exposes an RL-shaped interface | **yes** | `tau2.gym.AgentGymEnv`, Gymnasium-compatible, runs customer + tools + reward |
| Modal has the hardware | **yes** | H100 SXM5 $3.95/hr, H200 SXM 141GB $4.54/hr, A100 80GB $2.50/hr |
| Modal may upgrade H100 → H200 free | **yes** | stated in Modal's GPU guide; 141GB fits our whole stack on one card |
| Modal can run a long job | **24 h max per call** | documented; longer runs need checkpoint + retry, which we want anyway |
| Modal free credits | **$30/mo** (Starter) | ≈ 7.6 H100-hours — more than stage 3 needs |

## Architecture

```
Modal container (1x H100/H200)
├── vLLM serving the CURRENT policy
│     --enable-auto-tool-choice --tool-call-parser gemma4  (bf16)
├── rollout workers  ── N concurrent episodes
│     └── tau2 AgentGymEnv  ── customer sim (Gemini 3.7 Flash, external API)
│                            └── tools + DB + reward
└── GRPO trainer (LoRA)  → weights pushed back to vLLM each step
                          → checkpoints to a Modal Volume
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
| GPU, 30 steps @ concurrency 32 | $4.74 |
| customer-sim API, 3,840 rollouts | $30.72 |
| eval before + after (40 × 8, paired, 89% power) | $49.00 |
| **first full run** | **~$85** |

The GPU is the cheapest line. Budget attention accordingly.

## Stages

**Stage A — serve and re-baseline.** *(~$10, half a day)*
Stand up vLLM on Modal with the gemma4 tool parser, bf16. Re-run the τ2 telecom
baseline against it. **This becomes the real "before".** If the "before" came
from a rented API and the "after" from our own server, part of the delta is a
change of server — the entire lesson of D13/D14. Gate: the new baseline should
land near 35% and show no `malformed_tool_args`.

**Stage B — frontier scan on the train split.** *(~$25)*
n=8 over the 74-task telecom train split, **stratified by scenario** — the full
pool is 87% `mms_issue` at 9% success, so uniform sampling yields degenerate
groups (D18). Output: the task list with per-task pass@1 in [0.25, 0.75].

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
| **24-hour function cap** | low | checkpoint + retry, standard Modal pattern |
| **Cold start** loading 26B of weights | low | Modal Volume + memory snapshotting |
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
