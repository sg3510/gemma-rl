"""Resumable sweep runner over tau-bench.

Upstream's `run()` has a checkpoint file but no resume: restarting re-runs every
task from scratch. Free-tier sweeps *will* die partway, so without resume we
would repay for everything already done.

Design: one tau-bench invocation per trial, each writing into its own
`results/<sweep>/trial_<i>/` directory. That gives us

  * clean (task_id, trial) identity -- upstream stamps `trial=0` on every result
    when `num_trials=1`, so the directory is what distinguishes rollouts;
  * cell-level resume -- we scan what exists and pass only the missing task ids;
  * crash isolation -- a rollout that raised is re-run rather than being counted
    as a genuine 0.0 (see `metrics.is_errored`).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from tau_bench.types import RunConfig

from taurl import config, patches
from taurl.metrics import format_summary, load_results, scan_cells, summarize



def plan_remaining(
    sweep_dir: Path, task_ids: list[int], trials: int
) -> dict[int, list[int]]:
    """Return {trial_index: [task_ids still to run]}."""
    cells, errored = scan_cells(sweep_dir)
    done = {k for k in cells if k not in errored}
    plan = {}
    for trial in range(trials):
        missing = [t for t in task_ids if (t, f"trial_{trial}") not in done]
        if missing:
            plan[trial] = missing
    return plan


def check_split_hygiene(env: str, split: str, name: str) -> None:
    """Guard the held-out split.

    Retail has a real 500-task train pool, so its test split is sacred: exactly
    one sweep may touch it. Airline has *no* train split at all, so its test
    split is the only thing to measure -- we allow it but say so loudly, because
    anything we later train on airline would be training on the eval.
    """
    if split not in config.SPLIT_SIZES[env]:
        raise SystemExit(
            f"{env} has no '{split}' split (available: "
            f"{sorted(config.SPLIT_SIZES[env])})."
        )
    if split != "test":
        return
    if env == "retail" and name != "baseline_test":
        raise SystemExit(
            f"Refusing to run sweep '{name}' on retail's held-out test split. "
            "Only the single 'baseline_test' sweep may touch it."
        )
    if env == "airline":
        print(
            "NOTE: airline ships a test split only. This is an EVALUATION "
            "measurement; it cannot become an RL training set without an "
            "author-defined split."
        )


def run_sweep(
    name: str,
    split: str,
    task_ids: list[int],
    trials: int,
    tier: config.Tier,
    temperature: float,
    env: str = "retail",
    dry_run: bool = False,
) -> Path:
    check_split_hygiene(env, split, name)

    patches.apply(tier)
    config.require_api_key()

    sweep_dir = config.RESULTS_DIR / name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "sweep": name,
        "env": env,
        "split": split,
        "num_tasks": len(task_ids),
        "trials": trials,
        "tier": tier.name,
        "agent_model": tier.agent_model,
        "user_model": config.USER_MODEL,
        "user_model_alias": config.USER_MODEL_ALIAS,
        "agent_temperature": temperature,
        "user_temperature": config.USER_TEMPERATURE,
        "max_num_steps": config.MAX_NUM_STEPS,
        "provider": config.PROVIDER,
    }
    (sweep_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    plan = plan_remaining(sweep_dir, task_ids, trials)
    todo = sum(len(v) for v in plan.values())
    total = len(task_ids) * trials
    print(f"\n=== sweep '{name}' [{env}/{split}] ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"  rollouts: {total - todo}/{total} already done, {todo} to run")
    if dry_run:
        print("  (dry run -- nothing executed)")
        return sweep_dir
    if not todo:
        print("  nothing to do")
        return sweep_dir

    started = time.time()
    for trial, missing in sorted(plan.items()):
        print(f"\n--- trial {trial}: {len(missing)} tasks ---")
        from tau_bench.run import run as tau_run

        tau_run(
            RunConfig(
                model_provider=config.PROVIDER,
                user_model_provider=config.PROVIDER,
                model=tier.agent_model,
                user_model=config.USER_MODEL_ALIAS,
                num_trials=1,
                env=env,
                agent_strategy="tool-calling",
                temperature=temperature,
                task_split=split,
                task_ids=missing,
                log_dir=str(sweep_dir / f"trial_{trial}"),
                max_concurrency=tier.max_concurrency,
                user_strategy="llm",
            )
        )

    elapsed = time.time() - started
    spend = patches.spend()
    print(f"\n=== sweep '{name}' finished in {elapsed / 60:.1f} min ===")
    print(
        f"  spend: agent ${spend['agent']:.3f} + user ${spend['user']:.3f} "
        f"= ${spend['total']:.3f}"
    )

    tasks = load_results(sweep_dir)
    print()
    print(format_summary(summarize(tasks), f"sweep '{name}' ({env}/{split})"))
    _, errored = scan_cells(sweep_dir)
    if errored:
        print(f"WARNING: {len(errored)} rollouts crashed and were excluded.")
        print("Re-run the same command to retry them.")
    return sweep_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="sweep name -> results/<name>/")
    ap.add_argument("--env", choices=["retail", "airline"], default="retail")
    ap.add_argument("--split", choices=["dev", "test", "train"], default="dev")
    ap.add_argument("--tasks", type=int, help="run the first N task ids")
    ap.add_argument("--task-ids", type=int, nargs="+", help="explicit task ids")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--tier", choices=["free", "paid"], default=None)
    ap.add_argument(
        "--max-concurrency",
        type=int,
        help="override the tier default (useful when two sweeps share the "
             "provider's rate limit)",
    )
    ap.add_argument("--temperature", type=float, default=config.AGENT_TEMPERATURE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sizes = config.SPLIT_SIZES[args.env]
    if args.split not in sizes:
        raise SystemExit(
            f"{args.env} has no '{args.split}' split (available: {sorted(sizes)})."
        )
    if args.task_ids:
        task_ids = args.task_ids
    else:
        n = args.tasks or sizes[args.split]
        task_ids = list(range(min(n, sizes[args.split])))

    tier = config.get_tier(args.tier)
    if args.max_concurrency:
        tier = dataclasses.replace(tier, max_concurrency=args.max_concurrency)

    run_sweep(
        name=args.name,
        split=args.split,
        task_ids=task_ids,
        trials=args.trials,
        tier=tier,
        temperature=args.temperature,
        env=args.env,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
