"""Prefix-replay probe: is the loop reproducible, and what breaks it?

    .venv-tau2/bin/python -m tau2probe.replay_probe --prefixes 15 --samples 4

For each looping conversation in the snapshot, truncate the agent-visible
history right after the SECOND identical tool result -- the model has now seen
the same answer twice, and in the recorded trace it called a third time -- and
sample continuations under four conditions against the same pinned stack:

  A  control              exactly what tau2 sent
  B  dedup note           second result carries a note: state unchanged, an
                          action is required to change it
  C  causal guidance      system prompt + read-tools-don't-mutate paragraph
  D  normalized payload   content:"" instead of null, no duplicate "name" key,
                          no explicit tool_calls:null   (the H1 test)

Metric per variant: fraction of sampled continuations that re-issue the exact
looped call. Resumable; one JSON result per (sim, variant, sample) under
results/loop_probe/.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import litellm

from tau2.agent.base_agent import is_valid_agent_history_message
from tau2.data_model.simulation import Results
from tau2.registry import registry
from tau2.utils.llm_utils import to_litellm_messages

from tau2probe.wire_audit import normalize_payload

REPO = Path(__file__).resolve().parent.parent
SNAPSHOT = REPO / "results/_snapshots/pinned_telecom_prerepair.json"
OUT_DIR = REPO / "results/loop_probe"

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

DEDUP_NOTE = (
    "\n\n[note: this exact tool call was already made and returned an identical "
    "result. The state has not changed, and it will not change until an action "
    "is taken that changes it.]"
)
GUIDANCE = (
    "\n\nImportant: read-only tools (get_customer_*, get_details_by_id, "
    "get_bills_for_customer, get_data_usage) never change any state. Calling "
    "the same read-only tool again with the same arguments returns the same "
    "result. If a reading shows an undesired state, do not re-read it -- work "
    "out which action changes that state (one of your own write tools, or a "
    "step the customer must perform on their device) and take it, then re-check "
    "once afterwards."
)

VARIANTS = ["A", "B", "C", "D"]


def call_signature(name: str, arguments) -> str:
    return f"{name}::{json.dumps(arguments, sort_keys=True)}"


def find_loops(results: Results, threshold: int = 10):
    """(sim, looped_signature) for every looping simulation."""
    out = []
    for sim in results.simulations:
        if str(sim.termination_reason).endswith("infrastructure_error"):
            continue
        sigs = []
        for msg in sim.messages:
            for tc in getattr(msg, "tool_calls", None) or []:
                sigs.append(call_signature(tc.name, tc.arguments))
        counts = Counter(sigs)
        if counts and counts.most_common(1)[0][1] >= threshold:
            out.append((sim, counts.most_common(1)[0][0]))
    return out


def build_prefix_payload(sim, looped_sig: str, system_content: str, depth: int = 2):
    """litellm payload ending just after the depth-th identical tool result."""
    view = [m for m in sim.messages if is_valid_agent_history_message(m)]
    seen = 0
    cut = None
    pending: str | None = None
    for i, msg in enumerate(view):
        calls = getattr(msg, "tool_calls", None) or []
        for tc in calls:
            if call_signature(tc.name, tc.arguments) == looped_sig:
                pending = tc.id
        if msg.role == "tool" and pending and getattr(msg, "id", None) == pending:
            seen += 1
            pending = None
            if seen == depth:
                cut = i
                break
    if cut is None:
        return None
    payload = to_litellm_messages(view[: cut + 1])
    return [{"role": "system", "content": system_content}] + payload


def apply_variant(payload: list[dict], variant: str) -> list[dict]:
    payload = json.loads(json.dumps(payload))  # deep copy
    if variant == "B":
        payload[-1]["content"] = str(payload[-1].get("content") or "") + DEDUP_NOTE
    elif variant == "C":
        payload[0]["content"] += GUIDANCE
    elif variant == "D":
        payload = [payload[0]] + normalize_payload(payload[1:])
    return payload


def _bad_json(raw) -> bool:
    """A repetition penalty can corrupt structured output -- watch for it."""
    if not isinstance(raw, str):
        return False
    try:
        json.loads(raw)
        return False
    except json.JSONDecodeError:
        return True


def response_signatures(response) -> list[str]:
    msg = response.choices[0].message
    out = []
    for tc in msg.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            args = tc.function.arguments
        out.append(call_signature(tc.function.name, args))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefixes", type=int, default=15)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    ap.add_argument("--depth", type=int, default=2,
                    help="cut after the Nth identical result (2 = decision point "
                         "right after the first repeat; higher = deeper in the loop)")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--sampling", type=json.loads, default=None,
                    help='JSON merged into llm_args, e.g. \'{"top_p":0.95}\'. '
                         "Gemma 4 ships top_k=64, top_p=0.95 in its "
                         "generation_config; we have been sending temperature only.")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--provider", default=None, help="pin one OpenRouter backend")
    ap.add_argument("--tag", default="", help="prefix for result filenames")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = Results.load(args.snapshot)

    llm_args = json.loads(json.dumps(LLM_ARGS))
    if args.provider:
        llm_args["extra_body"]["provider"]["only"] = [args.provider]
    if args.sampling:
        llm_args.update(args.sampling)
    print(f"model={args.model} llm_args={json.dumps(llm_args)}")

    # Same system prompt tau2 built for these runs: agent instruction + policy.
    from tau2.agent.llm_agent import LLMAgent

    env = registry.get_env_constructor("telecom")()
    tools = env.get_tools()
    agent = LLMAgent(tools=tools, domain_policy=env.get_policy(),
                     llm=args.model, llm_args=llm_args)
    system_content = agent.system_prompt
    tool_schemas = [t.openai_schema for t in tools]

    loops = find_loops(results)[: args.prefixes]
    variants = args.variants
    print(f"{len(loops)} looping prefixes at depth {args.depth}; {len(variants)} "
          f"variants x {args.samples} samples = "
          f"{len(loops)*len(variants)*args.samples} calls")

    tag = (args.tag + "_" if args.tag else "") + ("" if args.depth == 2 else f"d{args.depth}_")
    tally: dict[str, list[int]] = {v: [0, 0] for v in variants}  # [repeats, total]
    for sim, looped_sig in loops:
        base = build_prefix_payload(sim, looped_sig, system_content, args.depth)
        if base is None:
            print(f"  {sim.id[:8]}: fewer than {args.depth} results, skipped")
            continue
        for variant in variants:
            payload = apply_variant(base, variant)
            for k in range(args.samples):
                out_file = OUT_DIR / f"{tag}{sim.id[:8]}_{variant}_{k}.json"
                if out_file.exists():
                    rec = json.loads(out_file.read_text())
                else:
                    resp = litellm.completion(
                        model=args.model, messages=payload, tools=tool_schemas,
                        num_retries=3, **llm_args,
                    )
                    sigs = response_signatures(resp)
                    rec = {
                        "sim": sim.id, "task": sim.task_id, "variant": variant,
                        "sampling": args.sampling, "model": args.model,
                        "sample": k, "looped_sig": looped_sig,
                        "response_sigs": sigs,
                        "repeats": looped_sig in sigs,
                        "content": resp.choices[0].message.content,
                        "n_tool_calls": len(resp.choices[0].message.tool_calls or []),
                        "bad_json": sum(
                            1 for tc in (resp.choices[0].message.tool_calls or [])
                            if _bad_json(tc.function.arguments)
                        ),
                    }
                    out_file.write_text(json.dumps(rec, indent=1))
                tally[variant][0] += rec["repeats"]
                tally[variant][1] += 1
        done = sum(t[1] for t in tally.values())
        print(f"  {sim.id[:8]} done ({done} calls so far)")

    print(f"\n| variant (depth {args.depth}) | re-issued the looped call | rate |")
    print("|---|---|---|")
    names = {"A": "A control", "B": "B dedup note", "C": "C causal guidance",
             "D": "D normalized payload"}
    for v in variants:
        r, n = tally[v]
        rate = f"{r/n:.0%}" if n else "n/a"
        print(f"| {names[v]} | {r}/{n} | {rate} |")


if __name__ == "__main__":
    main()
