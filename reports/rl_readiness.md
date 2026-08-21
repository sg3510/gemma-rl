# What we have, and what stands between us and an RL run

## Have

| asset | state |
|---|---|
| Environment with an objective verifier | τ2 ships **`AgentGymEnv`**, a Gymnasium env where you act as the agent and τ2 runs the user simulator, tools, DB and reward |
| Pinned, reproducible serving stack | NextBit bf16, temp 1.0, fixed customer, step cap 140 — all recorded per run. **Reproducible but wrongly chosen**: NextBit loops 5× more than Novita and loses 12% of cells to empty completions; sampling params were never pinned at all. See [provider_loop_rates.md](provider_loop_rates.md) |
| Baselines across 5 environments | τ1 retail/airline, τ2 retail/airline/telecom |
| Curriculum signal | per-scenario difficulty ladder inside telecom (9% / 44% / 57%) |
| Analysis tooling | pass@k vs pass^k, frontier selection, failure taxonomy, resumable sweeps |
| 901 successful trajectories | full message structure, saved to disk |

## The trap: saved traces do **not** get us GRPO

901 successful demonstrations sounds like a training set. It is not, for GRPO.

GRPO is **on-policy**: at every step you sample a fresh group of G rollouts
*from the policy you are currently training*, score them, and compute advantage
relative to the group mean. Trajectories generated weeks earlier by a different
policy — through an API that never returned token ids or logprobs — cannot enter
that objective. We checked: our traces carry `role`, `content`, `tool_calls`
and `reward`, and **no logprobs, no token ids**.

What the traces *are* good for:

- **Curriculum construction** — which tasks sit in the frontier band. This is
  their main job and they do it well.
- **Failure analysis** — the mechanical-vs-capability split.
- **An optional SFT warm-start** — 901 successful trajectories are a legitimate
  behavioural-cloning set to initialise from before RL. Worth trying, but it is
  a *different* technique, not a shortcut to GRPO.

## Gaps, in order of how hard they are to close

### 1. Trainable weights on real GPUs — the blocker

| configuration | memory |
|---|---|
| bf16 weights | 50.4 GB |
| 4-bit (QLoRA) weights | 12.6 GB |
| LoRA adapters + optimizer (r=16) | ~0.5 GB |
| activations + KV cache | ~10–25 GB |
| separate vLLM for rollouts | +50 GB (avoidable by colocating) |

- **bf16 LoRA with colocated rollouts:** 2× H100 80GB, or 1× H200 141GB.
- **4-bit QLoRA:** fits 1× A100 80GB, but trains in a precision we do not serve,
  reintroducing exactly the train/serve mismatch D13b warned about.
- **M3 Pro 36GB:** inference only. Not a training machine.

### 2. Self-hosted inference with the right tool parser

vLLM with `--enable-auto-tool-choice --tool-call-parser gemma4`. Needed twice
over: to generate on-policy rollouts, and to make the before/after comparison
mean anything. Without it we are back to a benchmark score that is a property of
someone else's server (B6, D14).

### 3. The user simulator is a cost *and* latency tax inside the loop

Every rollout needs ~12 sequential customer-simulator calls, and they are
strictly interleaved with agent turns — so they sit on the critical path.

| run shape | rollouts | customer-sim API cost |
|---|---|---|
| 30 steps × 32 tasks × G=8 | 7,680 | ~$61 |
| 100 steps × 32 tasks × G=8 | 25,600 | ~$205 |
| 100 steps × 74 tasks × G=8 | 59,200 | ~$474 |

That is API spend *on top of* GPU rental, and the serialisation means wall-clock
is dominated by round trips rather than by GPU throughput. Mitigations worth
testing: a cheaper/local customer model (but D6 pinned it, so changing it breaks
comparability), aggressive rollout parallelism, or τ2's `solo_mode` where
available.

### 4. Frontier task set at n=8 on the **train** split

Everything measured so far is on test splits, at n=2–4. The curriculum needs
n=8 on train, stratified by scenario (telecom's pool is 87% `mms_issue` at 9%
success — sampling it uniformly would produce mostly degenerate groups).
Estimated ~$45 for telecom train at n=8.

### 5. A GRPO trainer wired to a multi-turn environment

`verl` or TRL. Multi-turn agentic RL is not the default path in either — the
standard recipes assume single-turn prompt→completion. The `AgentGymEnv`
interface makes this tractable but it is still real integration work: rollout
collection, masking tool-result tokens out of the loss, and group-relative
advantage over whole trajectories rather than single completions.

## Rough order of operations

1. Rent GPUs; stand up vLLM bf16 with the gemma4 tool parser.
2. Re-measure the baseline on that stack (it becomes the real "before").
3. n=8 frontier scan on the telecom **train** split, stratified by scenario.
4. Wire `AgentGymEnv` to a GRPO trainer; verify the loop on a handful of tasks.
5. Train on the frontier set; evaluate on held-out telecom **and** on τ2 airline
   and τ1 retail to test whether anything transfers.

Step 2 is the one people skip and then cannot defend: if the "before" number
came from a rented API and the "after" came from our own vLLM, the delta is
partly a change of server.
