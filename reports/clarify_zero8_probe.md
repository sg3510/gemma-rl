# Does prompt clarification turn observed zero-success tasks into frontier tasks?

## Result: one discovered success, no observed frontier conversion

| candidate | historical original | fresh original | clarified | interpretation |
|---|---|---|---|---|
| 1 (8 faults, Hard persona) | 0/4 | **0/6**, 2 unfinished | **1/8** | a successful demonstration, below the 2/8 frontier cutoff |
| 2 (6 faults, default persona) | 0/4 | **3/8** | not run | ordinary resampling already exposes frontier signal |
| 3 (6 faults, Hard persona) | 0/4 | **0/7**, 1 unfinished | **0/8** | no discovered success |

The strict 0/8 → frontier question remains unresolved: baseline execution was
stopped after approximately 30.5 minutes with three episodes still outstanding
(running for roughly 21–30 minutes). These are missing, never scored failures.
There were 21 scored original-prompt episodes and 16 scored clarified episodes.
The clarified arm's one infrastructure-failed cell was recovered by resuming
the run; repeated empty assistant completions affected both arms.

Neither clarified task met the prespecified 2–6/8 frontier criterion. Candidate
1's trial 4 passed both environment assertions (MMS availability and the required
2 GB data refill); it still emitted invalid customer-tool calls, so this is not
evidence that boundary discrimination is fixed. Its held-out success is for
diagnosis only, never SFT training data.

Loop counts were 1/6 and 1/7 on the two scored original-prompt groups versus
0/8 and 0/8 with clarification (the separate resampling control had 1/8).
These small, incomplete, selected samples support trace discovery as a possible
use of clarification, not a causal claim of increased success probability or
a reliable curriculum-conversion method. The single success could still be
sampling variation. Missing long episodes also preclude an unbiased latency
comparison.

Counts and exact scored trial IDs: [clarify_zero8_results.json](clarify_zero8_results.json).
All launched local processes have stopped. Saved runs can resume missing cells.

## Protocol

Started 2026-09-12. Exploratory diagnostic, not a training or headline eval run.

No saved telecom run had eight scored attempts per task. Select the first three
lexicographically sorted task IDs with 0/4 in `mms_eval_extra`; freeze them in
`results/_snapshots/clarify_zero8_candidates.json`. These are held-out tasks:
none of their traces may enter SFT or GRPO training.

1. Run eight fresh original-prompt attempts per candidate.
2. For candidates with eight scored failures, run eight clarified attempts.
3. Report counts per task. At n=8, 2–6 successes meets our empirical frontier
   band [0.25, 0.75]; one success demonstrates reachability but falls below it.
   Infrastructure errors do not count as failures or toward eight attempts.

Fixed settings: Gemma 4 26B A4B IT, OpenRouter → Novita only, no fallbacks,
temperature 1, top_p 0.95, top_k 64; Gemini 3.7 Flash customer direct at
temperature 0; 140 steps, full tool set, no masking or loop breaker.
Clarification appends `You can't call a customer's tools but you can ask them
to use them.` inside a `tool_boundary` block; all other settings match.

Original-prompt run: `zero8_plain_20260912`, eight trials, concurrency six.
Logs: `results/zero8_plain_20260912.log`; traces are under
`vendor/tau2-bench/data/simulations/zero8_plain_20260912/`.

Execution amendment: after 17 scored original-prompt episodes, candidates 1
and 3 were both 0/5, while candidate 2 was 3/7. Start their clarified arms
concurrently with the remaining baseline attempts to avoid serial waiting;
qualification as a 0/8 rescue still depends on the completed baseline.
The two IDs are frozen in `clarify_zero8_tested.json`; clarified run name is
`zero8_clarify_20260912`, eight trials each, concurrency six. This amendment
was made before viewing any clarified-arm outcomes.

Selection on zero observed successes does not establish zero underlying
success probability. Any rescue here is exploratory: confirm promising tasks
with additional independent original/clarified arms before claiming a causal
frontier shift. Prior 0/4 results are selection data, not pooled into the fresh
eight-attempt comparison.

## Frozen task IDs

1. `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]`
2. `[mms_issue]airplane_mode_on|bad_network_preference|break_apn_mms_setting|break_app_both_permissions|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]`
3. `[mms_issue]airplane_mode_on|bad_wifi_calling|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card[PERSONA:Hard]`
