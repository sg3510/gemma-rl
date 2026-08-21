# Trace viewer

Static HTML for reading eval runs: which attempt succeeded, what the verifier
actually checked, and what the model did on the way there.

```bash
python -m viewer --all                                    # every run found on disk
python -m viewer vendor/tau2-bench/data/simulations/pinned_telecom
python -m viewer results/baseline_test -o /tmp/site
open viewer/out/index.html
```

One self-contained page per run plus an index. No server, no fetch, no build
step — the payload is inlined, so a page works over `file://` and can be handed
to somebody as a single file. `pinned_telecom` (160 rollouts, 26 MB of raw
trace) comes out at 3.8 MB.

## What it shows

**Grid.** Rows are tasks grouped by scenario, columns are trials. `✓` solved,
`✗` failed, `!` infrastructure error (not scored — needs a re-run), `·` never
run. A purple bar marks a frontier task: per-task pass@1 in [0.25, 0.75], the
same criterion as `taurl/triage.py`, which is the curriculum-selection view.

**Trace.** Messages with tool calls expanded, customer-side tool calls marked
separately (telecom lets the customer act on their own phone), and exact
duplicate calls flagged — the Tier-1 waste signal from
[../reports/tiered_reward.md](../reports/tiered_reward.md).

**Ideal end state.** τ2's reward is a checklist, not a database diff, so the
panel puts the task's `evaluation_criteria` next to what the verifier found,
check by check. Criteria outside the task's `reward_basis` are marked
*diagnostic only* — telecom grades on `ENV_ASSERTION`, so `db_check` is
routinely False on a trace that scored 1.0, and reading that as a failure is a
mistake the panel is built to prevent.

**Side by side.** Two attempts of the same task in parallel columns, sharing the
check panel. This is the comparison surface Stage F needs for Tier-2 preference
labelling.

**Index.** All runs with headline metrics, plus a paired per-task delta between
any two runs that share tasks — the test `../reports/eval_protocol.md` commits
us to. It warns when the two runs' serving stacks differ, because then part of
the delta is a change of server rather than of policy (D13/D14).

## Conventions worth knowing

- **Missing cells are rendered.** A run that planned 40 × 4 shows 160 cells even
  if 41 died on infrastructure and were never retried. `pinned_telecom` has 17
  of 40 tasks at full depth, which is why its pass@4 is not yet a run-level
  number — the grid is also the retry work list.
- **pass@k carries its eligible-task count.** pass@4 over the third of tasks
  that have four scored rollouts is a different claim from pass@4 over the run.
- **Macro and micro pass@1 are both shown.** Macro (mean of per-task rates) is
  what the paired before/after analysis uses; micro (successes / scored
  rollouts) is the README headline. They differ by 4 points on `pinned_telecom`
  because trial counts are uneven.
- **Unevaluated ≠ failed.** τ2 skips evaluation entirely when a simulation ends
  early, recording reward 0.0 with every check list null. 50 of 160 rollouts in
  `pinned_telecom` are step-cap terminations that were never scored, and the
  page says so rather than showing ten unmet checks.
- **Infrastructure errors leave the denominator**, matching
  `taurl.metrics.is_errored`. A model-fault crash (malformed tool JSON) stays a
  genuine failure.

## Adding another eval

Write a module with `detect(path) -> bool` and `load(path, name=None) -> Run`,
lowering the format into `viewer/model.py`, and list it in
`viewer/adapters/__init__.py`. Nothing else changes: metrics, grid, trace and
diff all work off the normalised `Run`. `tau2.py` and `tau1.py` are the two
worked examples — a future trained-policy eval, or a run from a different
harness, is a third.

| file | what |
|---|---|
| `model.py` | the normalised shapes every adapter targets |
| `adapters/` | one module per eval format, plus the registry |
| `splits.py` | split membership from τ2's own `split_tasks.json` |
| `metrics.py` | pass@k (via `taurl.metrics`), frontier, Tier-1 quality |
| `payload.py` | Run → the JSON the page reads |
| `render.py` + `assets/` | page templates, CSS, and the two vanilla-JS bundles |
