# Why we train a MoE, and how

Gemma 4 26B A4B is a mixture-of-experts model: **25.2B total parameters, 3.8B
active per token, 30 layers, 128 experts, top-8 routing**. Google also ships a
dense **Gemma 4 31B**. This documents why we picked the MoE, and why its
fine-tuning story is sound rather than merely tolerable.

---

## Why the MoE, not the dense 31B

| | 26B A4B (MoE) | 31B (dense) |
|---|---|---|
| active params | 3.8B | 31B — **8.2× the FLOPs/token** |
| bf16 weights | 50.4 GB | 62.0 GB |
| τ2 telecom (Artificial Analysis) | **44% — in band** | 60–65% — too easy |
| our measured τ2 telecom | **35% (pinned stack)** | not measured |
| GPU cost, 30 GRPO steps | **$4.74** | $21.70 |

Three independent reasons, in order of weight:

**1. It is the only one in the difficulty band.** The entire environment-selection
exercise was about finding a setup where the model succeeds *sometimes* —
roughly 20–50%. The dense 31B sits at 60–65% on telecom, which is the same "too
easy" verdict that eliminated retail. Choosing it would land us back at the
start, having spent the selection work for nothing.

**2. Sparsity is what makes the rollout loop viable.** With 3.8B active
parameters, generation takes ~18 s per rollout against ~18 s of customer-API
latency — roughly balanced. Dense 31B pushes generation to ~147 s, so it
dominates wall-clock, and rollout throughput is precisely what sets the GPU
bill. The MoE is not merely cheaper; it is what keeps generation off the
critical path.

**3. It fits comfortably.** 50.4 GB of bf16 weights plus LoRA, activations and a
colocated vLLM fit on a single 141 GB H200. 62 GB is tighter for no benefit.

### The honest counterargument

Dense models have a far more battle-tested LoRA story. That was, for a while,
our highest-severity risk. It turned out to be smaller than it looked — see
below — but it was a real reason to consider the dense model, and we should say
so rather than pretend the choice was obvious.

There is also one scenario where 31B genuinely wins: telecom's `mms_issue`
family sits at **9%** for 26B A4B — too hard, mostly degenerate rollout groups —
and **1,984 of the 2,285 tasks** live there. A stronger model would plausibly
put that pool in band, converting the largest and currently-unusable slice into
training data. That remains our documented fallback if the MoE path stalls.

---

## Why fine-tuning a MoE is fine

### The serving question is already solved upstream

The design that would have broken is colocated rollouts: the trainer must push
updated adapters into the vLLM instance generating rollouts. vLLM ships
**`vllm/lora/layers/fused_moe.py`** — LoRA over fused MoE layers is explicitly
implemented. Similarly, both vLLM (`gemma4_engine_tool_parser.py`) and SGLang
(`gemma4_detector.py`) parse Gemma 4's native tool-call syntax.

So the MoE risk was never "can it be served"; it is "where do adapters go".

### Where adapters go: the architecture decides

| strategy | trainable | gradient signal per token | verdict |
|---|---|---|---|
| **attention only** (q,k,v,o), r=16 | **10.8M** (0.04%) | **100% of tokens** | **our default** |
| router / gate, full fine-tune | 10.8M | 100% of tokens | high-leverage ablation |
| all 128 experts, r=16 | 908M (3.6%) | ~6% per expert | sample-starved |
| all 128 experts, r=4 | 227M | ~6% per expert | still sample-starved |

**Expert LoRA is the wrong tool here, and memory is not the reason** — 908M
adapters is only 1.8 GB. The problem is the routing: with top-8 of 128 experts,
each expert sees about **6% of tokens**. Across a few thousand RL rollouts most
expert adapters would receive almost no gradient. That is acceptable for SFT
over a large corpus and poor for RL, where rollouts are the scarce resource and
each one costs a customer-simulator round trip.

**Attention-only LoRA inverts that**: 10.8M parameters, every token contributes,
and it is the best-supported configuration in every framework. For a run of a
few thousand rollouts, signal density matters far more than parameter count.

### The MoE-specific opportunity

**Router tuning** is the interesting option a dense model cannot offer. For the
same ~10.8M parameters it changes *which* experts fire rather than what they
compute — recomposing existing capability instead of learning new capability,
which is a good match for a model that already knows how to use these tools and
mainly fails at applying them consistently.

The risk is routing collapse or load imbalance, which needs an auxiliary
load-balancing loss. Planned as an ablation after the attention-only baseline,
not as the default.

### Other options considered

| approach | why not first |
|---|---|
| **QLoRA** (4-bit base + bf16 adapters) | Drops weights 50 GB → 13 GB, but reintroduces the train/serve precision mismatch we spent the project eliminating. An H200 fits bf16 anyway. |
| **Prefix / prompt tuning** | Touches no weights, sidesteps MoE entirely, cheapest possible probe. Limited capacity. Worth keeping as a fallback if adapters misbehave. |
| **DoRA / rsLoRA** | Drop-in LoRA variants, marginal gains, identical MoE considerations. |
| **Freeze experts, train attention + layernorms** | Attention LoRA plus norms — slightly more capacity for near-zero extra cost. A reasonable variant of the default. |
| **Full fine-tune** | 25.2B parameters. Not on one GPU, not on this budget. |

---

## Revised risk position

We originally logged "LoRA on a MoE" as the highest-severity risk in the Modal
plan. That was based on a general intuition about MoE tooling immaturity rather
than on this model's architecture or the state of vLLM. Having checked both:

- **Downgraded from high to medium.** Serving is implemented upstream; the
  adapter-placement question has a clear answer driven by the routing sparsity.
- **What remains**: adapter placement is now a *tuning* decision rather than a
  feasibility one, and Stage C of the Modal plan (overfit 4 tasks, ~$5) is where
  it gets settled empirically before any real spend.

**Plan of record:** attention-only LoRA at r=16, bf16, colocated vLLM. Router
tuning as the first ablation if attention-only underfits.
