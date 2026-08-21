# rlhf-toy — τ-retail baseline & RL training frontier for Gemma 4 26B A4B

Measures where `google/gemma-4-26b-a4b-it` actually lands on τ-bench retail, and
turns per-task success rates into an RL curriculum.

The design decisions behind all of this — and the bugs found in τ-bench along the
way — are logged in **[rl_product_managers.md](rl_product_managers.md)**.

## Status

**Environment chosen: τ2 telecom.** Baseline complete on a fully pinned serving
stack. Next step is training; see [reports/rl_readiness.md](reports/rl_readiness.md).

> **Re-baselined 2026-08-20 on Novita.** The old NextBit numbers were
> substantially a *serving artifact* — 45% of rollouts ended in a tool-call loop
> and 19 of 160 cells died on empty completions. On Novita (same weights, same
> tasks, sampling now pinned explicitly) the loop rate is 7% and no cells are
> lost. See [reports/provider_loop_rates.md](reports/provider_loop_rates.md),
> [reports/tool_call_looping.md](reports/tool_call_looping.md), D30–D31.

| environment | tasks x trials | pass@1 | frontier tasks | stack |
|---|---|---|---|---|
| **τ2 telecom (Novita)** | 40 x 4 | **64.4%** | **55%** | pinned, sampling explicit |
| τ2 telecom, **mms pool sample** | 20 x 4 | **23.8%** | **65%** | pinned, sampling explicit |
| τ2 telecom (NextBit, superseded) | 40 x 4 | 41.9% | 56% | pinned |
| τ2 airline | 20 x 2 | 45.0% ±7.9% | 30%* | pinned |
| τ1 retail | 115 x 4 | 74.0% ±2.4% | 50% | unpinned |
| τ1 airline | 50 x 4 | 48.7% ±4.0% | — | unpinned |

\* measured at n=2, which systematically undercounts; see the decision log.

**The balanced `test` split is not representative of the task pool.** It is 38%
`service_issue` (2% of the pool, 87% pass) and 40% `mms_issue` (87% of the pool,
36% pass). A pool-drawn mms sample gives **pass@1 23.8%, pass@4 65.0%, 65%
frontier and zero tasks solved 4/4** — a far better RL target than the split's
64% with 12 of 40 tasks already saturated. Curriculum and eval should be drawn
from the pool, not the curated split.

"Frontier tasks" = share producing mixed rollout groups, i.e. usable GRPO
signal. Unpinned numbers predate the serving-stack fix and are **not**
comparable with pinned ones.

Telecom also ships a difficulty ladder for free:

| scenario | pool | pass@1 | frontier |
|---|---|---|---|
| `service_issue` | 47 | 57% | 87% |
| `mobile_data_issue` | 254 | 44% | 67% |
| `mms_issue` | 1,984 | 9% | 38% |

Sample the pool **stratified** — it is 87% `mms_issue` at 9% success, so uniform
sampling yields mostly all-zero groups.

## Setup

```bash
git clone --recurse-submodules <this repo> && cd rlhf-toy
uv venv --python 3.12
uv pip install -e . && uv pip install --no-deps -e vendor/tau-bench
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
```

`vendor/tau-bench` is a submodule pinned to `59a200c`, kept byte-identical to
upstream. Every change we need is monkey-patched at runtime in `taurl/patches.py`
so `git diff` against Sierra's repo stays empty and "we ran the real benchmark"
is an auditable claim.

## Usage

```bash
# smoke test the harness (dev split, never used for numbers)
python -m taurl.run_sweep --name smoke --split dev --tasks 3 --trials 1 --temperature 0.0

# headline baseline on the held-out test split
python -m taurl.run_sweep --name baseline_test --split test --trials 4 --tier paid

# frontier scan on the train split
python -m taurl.run_sweep --name frontier_train --split train --tasks 150 --trials 8 --tier paid

# analysis (all free — reads saved trajectories)
python -m taurl.metrics  results/baseline_test --title "Baseline (test)" --csv reports/baseline.csv
python -m taurl.triage   results/frontier_train --split train
python -m taurl.failures results/baseline_test
```

Sweeps resume: re-run the same command and only missing `(task, trial)` cells
execute. Rollouts that crashed are retried rather than counted as failures.

```bash
# trace viewer: per-attempt outcomes, traces, and ideal-vs-actual checks
python -m viewer --all && open viewer/out/index.html
```

See [viewer/README.md](viewer/README.md).

## pass@k vs pass^k

τ-bench reports **pass^k** (*all* k rollouts succeed). We need **pass@k**
(*at least one* of k succeeds). For a task solved 3 times out of 8:
pass@8 = 100%, pass^8 = 0%. Both are computed in `taurl/metrics.py` and every
report labels which is which.

The RL frontier criterion is per-task **pass@1** (`c/n`) in `[0.25, 0.75]` —
not pass@8, which is 1.0 for any task with a single success and so cannot
discriminate between tasks.

## Layout

| path | what |
|---|---|
| `taurl/config.py` | every knob that affects a measured number |
| `taurl/patches.py` | runtime patches over τ-bench (retries, temps, costs, step cap) |
| `taurl/run_sweep.py` | resumable sweep runner |
| `taurl/metrics.py` | pass@k / pass^k |
| `taurl/triage.py` | difficulty levels + frontier selection |
| `taurl/failures.py` | mechanical-vs-capability failure taxonomy |
| `viewer/` | static trace viewer over any eval run (τ2 and τ1 adapters) |

## The pinned stack

> **Superseded 2026-08-20.** NextBit turns out to be the worst of the three
> working backends for τ2 telecom — 32% tool-call-loop rate against Novita's 6%
> on identical tasks, and 19 of 160 cells lost to empty completions. Plan of
> record is to re-baseline on **Novita** with sampling parameters pinned
> explicitly. See [reports/provider_loop_rates.md](reports/provider_loop_rates.md)
> and D30 in the decision log. The table below records what *was* pinned.

A benchmark score for a hosted open-weights model is not a property of the
model. It is a property of the model *and* the backend that served it. We
measured ~30-point swings between OpenRouter backends for the same model id, and
one backend (Google's) corrupts tool calls outright. So everything is pinned and
recorded per run:

| knob | value |
|---|---|
| backend | OpenRouter → **NextBit only**, no fallbacks, `require_parameters` — **superseded, see [provider_loop_rates.md](reports/provider_loop_rates.md)** |
| precision | bf16 (matches what we would fine-tune in) |
| agent temperature | 1.0 (pass@k is meaningless greedy) |
| customer | `gemini-3.7-flash` @ temp 0, fixed since day one |
| step cap | 140 (τ2) / 30 (τ1) |
| task split | `test`, held out |

Full reasoning: [reports/leaderboard_vs_ours.md](reports/leaderboard_vs_ours.md).

## Reports

| file | what |
|---|---|
| [rl_product_managers.md](rl_product_managers.md) | decision log — every call and why |
| [reports/leaderboard_vs_ours.md](reports/leaderboard_vs_ours.md) | why our numbers differ from published leaderboards |
| [reports/rl_readiness.md](reports/rl_readiness.md) | what stands between us and a training run |
| [reports/eval_protocol.md](reports/eval_protocol.md) | how we differ from Sierra's protocol, and the power analysis |
| [reports/moe_finetuning.md](reports/moe_finetuning.md) | why we train a MoE and where the adapters go |
| [reports/design_doc_v1.md](reports/design_doc_v1.md) | **current plan**: SFT-then-GRPO design, all settled params, technique review |
| [reports/rl_run_plan.md](reports/rl_run_plan.md) | staged plan for the GRPO run (RunPod primary; superseded by design_doc_v1 where they conflict) |
| [reports/tool_call_looping.md](reports/tool_call_looping.md) | the dominant telecom failure is a tool-call loop, not capability |
| [reports/provider_loop_rates.md](reports/provider_loop_rates.md) | backends disagree 5× on loop rate; our pinned one is the worst |
| [reports/tiered_reward.md](reports/tiered_reward.md) | tiered reward design — the RLHF half |
| [reports/simulator_distribution.md](reports/simulator_distribution.md) | what training against a simulated customer bakes in |
| [reports/rl_notes.md](reports/rl_notes.md) | RL methodology notes, and every correction we made |

## Caveat on comparability

The simulated customer is `google/gemini-3.7-flash` at temperature 0, not
τ-bench's default `gpt-4o`. Our numbers are "Gemma 4 26B A4B on τ-retail under
this harness", not a leaderboard entry.
