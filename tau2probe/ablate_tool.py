"""Ablation: remove the looped tool and see whether the loop moves or dies.

    .venv-tau2/bin/python -m tau2probe.ablate_tool --drop get_details_by_id \
        --tasks-file results/_snapshots/loopy_tasks.json --trials 4

The loop concentrates on `get_details_by_id`: a generic id-lookup read that
always succeeds, never mutates, and never advances the task. Two competing
explanations:

  (a) that specific tool is an attractor -- always-available, always-successful,
      zero-consequence, so repetition is cheap;
  (b) the agent is passively polling for a state change it did not cause, and
      the identity of the tool is incidental.

Under (a), removing the tool should end the lock-ups. Under (b), the loop should
migrate to another read (`get_customer_by_id`, `get_data_usage`, ...).

The ablated tasks may become genuinely unsolvable without the dropped tool --
that is fine and expected. The metric is NOT reward, it is whether the trace
still ends in a repetition lock-up and on which tool.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.tasks import Task
from tau2.registry import registry
from tau2.runner import run_tasks
from tau2.runner.batch import _run_save_paths
from tau2.data_model.simulation import TextRunConfig

REPO = Path(__file__).resolve().parent.parent

CLARIFICATION = """

<tool_boundary>
You can't call a customer's tools but you can ask them to use them.
</tool_boundary>
"""

MODEL = "openrouter/google/gemma-4-26b-a4b-it"
LLM_ARGS = {
    "temperature": 1.0,
    "extra_body": {
        "provider": {
            "only": ["NextBit"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    },
}


def make_factory(drop: set[str]):
    """An llm_agent whose tool list is missing the named tools."""

    def factory(tools, domain_policy, **kwargs):
        kept = [t for t in tools if t.name not in drop]
        dropped = sorted({t.name for t in tools} & drop)
        if dropped:
            print(f"[ablation] dropped tools: {dropped}; {len(kept)} remain")
        kwargs.pop("task", None)
        kwargs.pop("audio_native_config", None)
        kwargs.pop("audio_taps_dir", None)
        return LLMAgent(tools=kept, domain_policy=domain_policy, **kwargs)

    return factory


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["drop", "mask", "breaker", "plain", "clarify"],
                    default="drop",
                    help="drop = remove the tool for the whole run (static "
                         "ablation); mask = hide it only while looping "
                         "(dynamic, cascading, reversible)")
    ap.add_argument("--drop", nargs="+", default=["get_details_by_id"])
    ap.add_argument("--mask-after", type=int, default=3)
    ap.add_argument("--restore-after", type=int, default=2)
    ap.add_argument("--no-cascade", action="store_true")
    ap.add_argument("--tasks-file", type=Path, required=True,
                    help="JSON list of task ids")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--save-to", default="ablate_no_details")
    ap.add_argument("--max-steps", type=int, default=140)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--provider", default="NextBit")
    ap.add_argument("--sampling", type=json.loads, default=None)
    ap.add_argument("--customer", choices=["openrouter", "aistudio", "failover"],
                    default="openrouter",
                    help="Where to serve the customer simulator. OpenRouter "
                         "throttles gemini-3.7-flash globally under high demand "
                         "(model_limit_rpm, shared across all their users), which "
                         "caps concurrency at ~12 and is the real throughput "
                         "bottleneck -- the agent backend never rate-limited. "
                         "The direct Gemini API has far higher Tier-2 quotas and "
                         "half the latency (0.9s vs 1.8s per call).")
    args = ap.parse_args()

    if args.mode == "clarify":
        # PROTOCOL DEVIATION, deliberate and measured. The shipped manual names
        # the customer's device tools in callable syntax inside the agent's own
        # prompt ("You can check ... by using check_network_status()"), and 66%
        # of rollouts call one. This appends a single unambiguous statement of
        # the tool boundary and changes nothing else, to test whether the
        # hallucination is caused by that ambiguity.
        from tau2.agent.llm_agent import LLMAgent as _A
        agent_name = "llm_agent_clarified"

        def factory(tools, domain_policy, **kw):
            for k in ("task", "audio_native_config", "audio_taps_dir"):
                kw.pop(k, None)
            return _A(tools=tools, domain_policy=domain_policy + CLARIFICATION, **kw)

        MASK_LOG = None
    elif args.mode == "breaker":
        from tau2probe.loop_breaker import make_factory as breaker_factory
        agent_name = "llm_agent_breaker"
        factory = breaker_factory()
        MASK_LOG = None
    elif args.mode == "plain":
        from tau2.agent.llm_agent import LLMAgent as _A
        agent_name = "llm_agent_plain"
        factory = lambda tools, domain_policy, **kw: _A(
            tools=tools, domain_policy=domain_policy,
            **{k: v for k, v in kw.items()
               if k not in ("task", "audio_native_config", "audio_taps_dir")})
        MASK_LOG = None
    elif args.mode == "mask":
        from tau2probe.loop_mask import MASK_LOG, make_factory as mask_factory
        agent_name = "llm_agent_loopmask"
        factory = mask_factory(mask_after=args.mask_after,
                               restore_after=args.restore_after,
                               cascade=not args.no_cascade)
    else:
        agent_name = "llm_agent_ablated"
        factory = make_factory(set(args.drop))
        MASK_LOG = None
    registry.register_agent_factory(factory, name=agent_name)

    task_ids = json.loads(args.tasks_file.read_text())
    domain = "telecom"
    all_tasks = registry.get_tasks_loader(domain)(None)
    tasks: list[Task] = [t for t in all_tasks if t.id in set(task_ids)]
    missing = set(task_ids) - {t.id for t in tasks}
    if missing:
        raise SystemExit(f"task ids not found: {missing}")
    print(f"{len(tasks)} tasks x {args.trials} trials, dropping {args.drop}")

    llm_args = json.loads(json.dumps(LLM_ARGS))
    llm_args["extra_body"]["provider"]["only"] = [args.provider]
    if args.sampling:
        llm_args.update(args.sampling)
    print(f"[run] provider={args.provider} llm_args={json.dumps(llm_args)}")

    if args.customer == "failover":
        from tau2probe.customer_failover import DIRECT, install
        key = os.environ.get("AISTUDIO_GOOGLE_KEY")
        if key:
            os.environ.setdefault("GEMINI_API_KEY", key)
        install()
        user_llm = DIRECT  # start on the higher-quota path, fall back to proxy
    elif args.customer == "aistudio":
        key = os.environ.get("AISTUDIO_GOOGLE_KEY")
        if not key:
            raise SystemExit("AISTUDIO_GOOGLE_KEY not set")
        os.environ.setdefault("GEMINI_API_KEY", key)
        user_llm = "gemini/gemini-3.7-flash"
    else:
        user_llm = "openrouter/google/gemini-3.7-flash"
    print(f"[run] customer={user_llm} concurrency={args.concurrency}")

    config = TextRunConfig(
        domain=domain,
        agent=agent_name,
        llm_agent=MODEL,
        llm_args_agent=llm_args,
        user="user_simulator",
        llm_user=user_llm,
        llm_args_user={"temperature": 0.0},
        num_trials=args.trials,
        max_steps=args.max_steps,
        max_errors=10,
        max_concurrency=args.concurrency,
        save_to=args.save_to,
        auto_resume=True,
    )
    # run_tasks does NOT derive a save path from config.save_to -- only the CLI
    # layer does that (via _run_save_paths). Passing it explicitly is the
    # difference between keeping the traces and losing the whole run.
    _, save_dir, save_path, results_format = _run_save_paths(config)
    print(f"[ablation] saving to {save_path}")
    run_tasks(config, tasks, save_path=save_path, save_dir=save_dir,
              results_format=results_format)
    if args.customer == "failover":
        from tau2probe.customer_failover import stats
        print(f"[run] customer routing: {stats()}")
    print(f"[ablation] wrote {save_path}")
    if MASK_LOG is not None:
        log_path = REPO / "results" / f"{args.save_to}_mask_events.json"
        log_path.write_text(json.dumps(MASK_LOG, indent=1))
        print(f"[ablation] {len(MASK_LOG)} mask activations -> {log_path}")


if __name__ == "__main__":
    main()
